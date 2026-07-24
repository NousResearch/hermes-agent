"""Pinecone memory manager — namespace-based session memory for agents.

Provides helpers for storing and retrieving agent conversation memory
using Pinecone namespaces. Each session gets its own namespace for isolation,
with cross-session search available via the global namespace.

Usage:
    export PINECONE_API_KEY="your-key"
    export OPENAI_API_KEY="your-key"
    python memory_manager.py --index-name agent-memory --action store \
        --session-id sess-001 --text "User discussed project architecture"
    python memory_manager.py --index-name agent-memory --action recall \
        --query "architecture decisions"
    python memory_manager.py --index-name agent-memory --action cleanup \
        --session-id sess-001
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time


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


def get_embeddings():
    """Get the embedding model."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("Error: langchain-openai not installed. Run: pip install langchain-openai", file=sys.stderr)
        sys.exit(1)
    return OpenAIEmbeddings()


def store_memory(index, session_id: str, text: str, metadata: dict | None = None):
    """Store a memory entry in the session namespace."""
    embeddings = get_embeddings()
    vector = embeddings.embed_query(text)

    doc_id = hashlib.sha256(f"{session_id}:{text}:{time.time()}".encode()).hexdigest()[:16]
    entry_metadata = {
        "text": text[:1000],
        "session_id": session_id,
        "timestamp": int(time.time()),
    }
    if metadata:
        entry_metadata.update(metadata)

    index.upsert(
        vectors=[{"id": doc_id, "values": vector, "metadata": entry_metadata}],
        namespace=session_id,
    )
    print(f"Stored memory [{doc_id}] in namespace '{session_id}'")
    return doc_id


def recall_memories(index, query: str, session_id: str | None = None, top_k: int = 5):
    """Recall memories matching a query, optionally scoped to a session."""
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    kwargs = {"vector": query_vector, "top_k": top_k, "include_metadata": True}
    if session_id:
        kwargs["namespace"] = session_id

    results = index.query(**kwargs)

    print(f"\nRecalling memories for: {query!r}")
    if session_id:
        print(f"Scoped to session: {session_id}")
    print(f"Found {len(results['matches'])} results:\n")

    for match in results["matches"]:
        score = match["score"]
        text = match["metadata"].get("text", "")[:200]
        sess = match["metadata"].get("session_id", "unknown")
        ts = match["metadata"].get("timestamp", 0)
        print(f"  [{score:.4f}] session={sess} time={ts}")
        print(f"    {text}")
        print()

    return results


def cleanup_session(index, session_id: str):
    """Delete all vectors in a session namespace."""
    index.delete(delete_all=True, namespace=session_id)
    print(f"Cleaned up namespace '{session_id}'")


def show_stats(index):
    """Show index statistics."""
    stats = index.describe_index_stats()
    print(f"Total vectors: {stats['total_vector_count']}")
    namespaces = stats.get("namespaces", {})
    if namespaces:
        print(f"Namespaces ({len(namespaces)}):")
        for ns, info in sorted(namespaces.items()):
            print(f"  '{ns}': {info['vector_count']} vectors")
    else:
        print("No namespaces found.")


def main():
    parser = argparse.ArgumentParser(description="Pinecone agent memory manager")
    parser.add_argument("--index-name", required=True, help="Pinecone index name")
    parser.add_argument(
        "--action",
        choices=["store", "recall", "cleanup", "stats"],
        required=True,
    )
    parser.add_argument("--session-id", help="Session namespace ID")
    parser.add_argument("--text", help="Text to store as memory")
    parser.add_argument("--query", help="Query for recall")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    pc = get_pinecone_client()
    index = pc.Index(args.index_name)

    if args.action == "store":
        if not args.session_id or not args.text:
            parser.error("--session-id and --text required for store action")
        store_memory(index, args.session_id, args.text)
    elif args.action == "recall":
        if not args.query:
            parser.error("--query required for recall action")
        recall_memories(index, args.query, session_id=args.session_id, top_k=args.top_k)
    elif args.action == "cleanup":
        if not args.session_id:
            parser.error("--session-id required for cleanup action")
        cleanup_session(index, args.session_id)
    elif args.action == "stats":
        show_stats(index)


if __name__ == "__main__":
    main()
