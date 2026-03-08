#!/usr/bin/env python3
"""
Demo: Cognitive Memory System with real embeddings.

Tests the full pipeline: embed -> store -> recall -> contradiction -> forget.
"""

import os
import sys
import tempfile

# Gemini embedding model via litellm
EMBEDDING_MODEL = "gemini/gemini-embedding-001"


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY env var required")
        sys.exit(1)

    # --- Setup ---
    from cognitive_memory.store import CognitiveStore
    from cognitive_memory.embeddings import LiteLLMEmbedder
    from cognitive_memory.recall import RecallEngine, RecallConfig
    from cognitive_memory.encoding import encode
    from cognitive_memory.extraction import ForgettingManager

    db_path = os.path.join(tempfile.mkdtemp(), "demo_cognitive.db")
    store = CognitiveStore(db_path=db_path)
    embedder = LiteLLMEmbedder(model=EMBEDDING_MODEL, api_key=api_key)
    engine = RecallEngine(store=store, embedder=embedder)
    forgetting = ForgettingManager(store=store)

    print("=" * 60)
    print("COGNITIVE MEMORY SYSTEM - LIVE DEMO")
    print("=" * 60)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"DB: {db_path}")
    print()

    # --- 1. Store memories ---
    print("[1] STORING MEMORIES")
    print("-" * 40)

    memories_to_store = [
        ("User prefers dark mode in all IDEs", "/user", 0.8),
        ("The project uses Python 3.11 with FastAPI backend", "/project", 0.7),
        ("API keys are stored in HashiCorp Vault, never in .env files", "/project/security", 0.9),
        ("The team follows conventional commits: feat:, fix:, refactor:", "/project/conventions", 0.7),
        ("User's timezone is UTC+3, based in Istanbul", "/user", 0.6),
        ("PostgreSQL 15 is the primary database", "/project/infra", 0.7),
        ("Redis is used for caching with 5 minute TTL", "/project/infra", 0.6),
        ("The CI pipeline runs on GitHub Actions with pytest", "/project/ci", 0.7),
    ]

    for content, scope, importance in memories_to_store:
        result = engine.add_and_recall(
            content=content,
            scope=scope,
            importance=importance,
        )
        encoding = encode(content)
        print(f"  #{result['memory_id']:2d} [{', '.join(encoding.categories):<25s}] {content[:60]}")
        if result["related"]:
            for r in result["related"][:2]:
                print(f"       -> related: {r.memory.content[:50]} (sim={r.similarity:.3f})")

    print(f"\n  Total memories: {store.count()}")
    print()

    # --- 2. Semantic Recall ---
    print("[2] SEMANTIC RECALL")
    print("-" * 40)

    queries = [
        "What database does the project use?",
        "How should I format commit messages?",
        "Where are secrets stored?",
        "What does the user prefer?",
        "Tell me about the infrastructure",
    ]

    for query in queries:
        results = engine.recall(query, limit=3)
        print(f"\n  Q: {query}")
        if results:
            for r in results:
                print(f"    -> [{r.score:.3f}] {r.memory.content[:65]}")
                print(f"       (sim={r.similarity:.3f}, reasons={r.match_reasons})")
        else:
            print("    -> No results")

    print()

    # --- 3. Contradiction Detection ---
    print("[3] CONTRADICTION DETECTION")
    print("-" * 40)

    contradictions = [
        "The project uses MySQL, not PostgreSQL",
        "API keys should be stored in .env files for convenience",
        "User prefers light mode",
    ]

    for text in contradictions:
        # Find similar memories first
        try:
            emb = embedder.embed_text(text)
            similar = store.search_similar(emb, threshold=0.5, limit=3)
        except Exception as e:
            print(f"  Embedding failed: {e}")
            continue

        encoding = encode(text, candidates=similar)
        print(f"\n  New: {text}")
        print(f"  Categories: {encoding.categories}")
        print(f"  Importance: {encoding.importance:.3f}")
        if encoding.contradictions:
            for c in encoding.contradictions:
                print(f"  !! CONTRADICTION (confidence={c.confidence:.3f}): {c.existing_memory.content[:60]}")
        else:
            print("  No contradictions detected")

    print()

    # --- 4. Scope-based Operations ---
    print("[4] SCOPE OPERATIONS")
    print("-" * 40)

    project_memories = store.get_all_active("/project")
    user_memories = store.get_all_active("/user")
    print(f"  /project scope: {len(project_memories)} memories")
    print(f"  /user scope: {len(user_memories)} memories")

    print()

    # --- 5. Status ---
    print("[5] MEMORY STATUS")
    print("-" * 40)
    print(f"  Active: {store.count()}")
    print(f"  Total (incl. forgotten): {store.count(include_forgotten=True)}")
    print(f"  Embedding dimensions: {embedder.dimensions}")

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    store.close()


if __name__ == "__main__":
    main()
