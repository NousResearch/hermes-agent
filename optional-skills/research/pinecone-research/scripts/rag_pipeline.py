"""Pinecone RAG pipeline — index documents and query with retrieval-augmented generation.

Usage:
    export PINECONE_API_KEY="your-key"
    export OPENAI_API_KEY="your-key"
    python rag_pipeline.py --index-name agent-memory --action index --docs-dir ./docs
    python rag_pipeline.py --index-name agent-memory --action query --query "How does X work?"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def get_pinecone_client():
    """Initialize Pinecone client from environment."""
    try:
        from pinecone import Pinecone
    except ImportError:
        print("Error: pinecone-client not installed. Run: pip install pinecone-client", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    return Pinecone(api_key=api_key)


def ensure_index(pc, index_name: str, dimension: int = 1536):
    """Create the index if it doesn't exist."""
    from pinecone import ServerlessSpec

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created index: {index_name}")
    else:
        print(f"Index already exists: {index_name}")
    return pc.Index(index_name)


def load_documents(docs_dir: str) -> list[dict]:
    """Load text files from a directory as documents."""
    docs = []
    docs_path = Path(docs_dir)
    if not docs_path.is_dir():
        print(f"Error: {docs_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    for filepath in sorted(docs_path.rglob("*.txt")):
        text = filepath.read_text(encoding="utf-8").strip()
        if text:
            docs.append({
                "id": str(filepath.relative_to(docs_path)),
                "text": text,
                "metadata": {"source": str(filepath.name)},
            })
    return docs


def index_documents(index, docs: list[dict], batch_size: int = 100):
    """Embed and upsert documents into Pinecone."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("Error: langchain-openai not installed. Run: pip install langchain-openai", file=sys.stderr)
        sys.exit(1)

    embeddings = OpenAIEmbeddings()
    vectors = []

    for doc in docs:
        embedding = embeddings.embed_query(doc["text"])
        vectors.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {**doc["metadata"], "text": doc["text"][:1000]},
        })

    # Batch upsert
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    print(f"Total vectors indexed: {len(vectors)}")


def query_index(index, query: str, top_k: int = 5):
    """Embed a query and retrieve similar documents from Pinecone."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("Error: langchain-openai not installed. Run: pip install langchain-openai", file=sys.stderr)
        sys.exit(1)

    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(query)

    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    print(f"\nQuery: {query}")
    print(f"Top {top_k} results:\n")
    for match in results["matches"]:
        score = match["score"]
        source = match["metadata"].get("source", "unknown")
        text_preview = match["metadata"].get("text", "")[:200]
        print(f"  [{score:.4f}] {source}")
        print(f"    {text_preview}...")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Pinecone RAG pipeline")
    parser.add_argument("--index-name", required=True, help="Pinecone index name")
    parser.add_argument("--action", choices=["index", "query", "stats"], required=True)
    parser.add_argument("--docs-dir", help="Directory of .txt files to index")
    parser.add_argument("--query", help="Query string for retrieval")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    pc = get_pinecone_client()
    index = ensure_index(pc, args.index_name)

    if args.action == "index":
        if not args.docs_dir:
            parser.error("--docs-dir required for index action")
        docs = load_documents(args.docs_dir)
        if not docs:
            print("No .txt documents found.", file=sys.stderr)
            sys.exit(1)
        index_documents(index, docs)
    elif args.action == "query":
        if not args.query:
            parser.error("--query required for query action")
        query_index(index, args.query, top_k=args.top_k)
    elif args.action == "stats":
        stats = index.describe_index_stats()
        print(f"Total vectors: {stats['total_vector_count']}")
        for ns, info in stats.get("namespaces", {}).items():
            print(f"  Namespace '{ns}': {info['vector_count']} vectors")


if __name__ == "__main__":
    main()
