import os
import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete_if_cache

WORKING_DIR = "./ZGF_Family_Letters"

setup_logger("lightrag", level="INFO", log_file_path=f"{WORKING_DIR}/lightrag.log")

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "deepseek-v3-241226",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        **kwargs,
    )

# ollama pull nomic-embed-text
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await ollama_embed(
        texts, embed_model="nomic-embed-text", host="http://localhost:11434"
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = len(embedding[0])
    return embedding_dim


async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


async def initialize_rag() -> LightRAG:
    await test_funcs()
    
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance
        rag: LightRAG = await initialize_rag()

        with open("./ZGF_Family_Letters.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # # Perform naive search
        # print(
        #     await rag.aquery(
        #         "What are the top themes in this story?", param=QueryParam(mode="naive")
        #     )
        # )

        # # Perform local search
        # print(
        #     await rag.aquery(
        #         "What are the top themes in this story?", param=QueryParam(mode="local")
        #     )
        # )

        # # Perform global search
        # print(
        #     await rag.aquery(
        #         "What are the top themes in this story?",
        #         param=QueryParam(mode="global"),
        #     )
        # )

        # Perform hybrid search
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
