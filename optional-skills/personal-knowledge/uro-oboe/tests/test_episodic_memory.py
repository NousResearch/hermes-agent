"""
Tests for Episodic Memory Skill

Run with: python -m pytest tests/test_episodic_memory.py -v
Or directly: python tests/test_episodic_memory.py
"""

import sys
import os

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from episodic_memory import (
    episodic_store,
    episodic_recall_fuzzy,
    episodic_fetch,
    episodic_search_fts,
    episodic_stats,
)


def test_basic_store_and_fetch():
    """Test basic store and fetch operations."""
    print("Testing basic store and fetch...")

    # Store a memory
    result = episodic_store(
        text="Test memory for unit testing",
        tags=["test", "unit"],
        metadata={"source": "test_suite", "version": 1}
    )

    assert "id" in result
    assert result["text"] == "Test memory for unit testing"
    assert result["tags"] == ["test", "unit"]
    assert result["metadata"]["source"] == "test_suite"
    memory_id = result["id"]
    print(f"  Stored memory ID: {memory_id}")

    # Fetch it back
    fetched = episodic_fetch([memory_id])
    assert len(fetched["memories"]) == 1
    mem = fetched["memories"][0]
    assert mem["id"] == memory_id
    assert mem["text"] == "Test memory for unit testing"
    assert mem["tags"] == ["test", "unit"]
    assert mem["metadata"]["source"] == "test_suite"
    assert mem["vector"] is not None
    assert len(mem["vector"]) == 384
    print(f"  Fetched memory: {mem['text'][:50]}...")
    print("  ✓ Basic store and fetch passed")


def test_fuzzy_recall():
    """Test fuzzy recall with binary HNSW."""
    print("\nTesting fuzzy recall...")

    # Store multiple memories
    ids = []
    texts = [
        "User prefers Japanese language for responses",
        "Python pytest configuration with xdist plugin",
        "Novel writing project about cyberpunk themes",
        "Ollama local LLM running on 6GB GPU",
        "Hermes custom provider for Gemini 3.7 Flash",
    ]

    for text in texts:
        result = episodic_store(text, tags=["test"])
        ids.append(result["id"])

    print(f"  Stored {len(ids)} memories")

    # Fuzzy recall
    results = episodic_recall_fuzzy("Japanese language preference", k=3, noise_ratio=0.0)
    assert len(results["results"]) > 0
    print(f"  Found {len(results['results'])} results")

    # Check that the Japanese memory is in top results
    top_id = results["results"][0]["id"]
    fetched = episodic_fetch([top_id])
    assert "Japanese" in fetched["memories"][0]["text"]
    print(f"  Top result: {fetched['memories'][0]['text'][:50]}...")
    print("  ✓ Fuzzy recall passed")


def test_noise_ratio():
    """Test noise_ratio parameter mixes in low-score results."""
    print("\nTesting noise_ratio...")

    # Store memories with different relevance
    episodic_store("Completely unrelated content about cooking recipes", tags=["noise_test"])
    episodic_store("Another unrelated memory about gardening tips", tags=["noise_test"])
    episodic_store("Target memory about machine learning embeddings", tags=["noise_test", "target"])

    # Search with noise_ratio=0 (no noise)
    results_clean = episodic_recall_fuzzy("machine learning embeddings", k=2, noise_ratio=0.0, tags=["noise_test"])
    print(f"  Clean results (noise_ratio=0): {len(results_clean['results'])}")

    # Search with noise_ratio=0.5 (50% noise)
    results_noisy = episodic_recall_fuzzy("machine learning embeddings", k=4, noise_ratio=0.5, tags=["noise_test"])
    print(f"  Noisy results (noise_ratio=0.5): {len(results_noisy['results'])}")

    # With noise, we should get more results (including lower scoring ones)
    assert len(results_noisy["results"]) >= len(results_clean["results"])
    print("  ✓ Noise ratio parameter works")


def test_tag_filtering():
    """Test tag filtering in fuzzy recall."""
    print("\nTesting tag filtering...")

    episodic_store("Memory with tag A only", tags=["tag_a"])
    episodic_store("Memory with tag B only", tags=["tag_b"])
    episodic_store("Memory with both tags", tags=["tag_a", "tag_b"])

    # Filter by tag_a
    results = episodic_recall_fuzzy("memory", k=10, tags=["tag_a"])
    for r in results["results"]:
        fetched = episodic_fetch([r["id"]])
        assert "tag_a" in fetched["memories"][0]["tags"]

    # Filter by tag_b
    results = episodic_recall_fuzzy("memory", k=10, tags=["tag_b"])
    for r in results["results"]:
        fetched = episodic_fetch([r["id"]])
        assert "tag_b" in fetched["memories"][0]["tags"]

    print("  ✓ Tag filtering passed")


def test_fts_search():
    """Test full-text search with FTS5."""
    print("\nTesting FTS5 full-text search...")

    # Use unique strings to avoid collisions with existing data
    import uuid
    unique1 = f"fts_test_fox_{uuid.uuid4().hex[:8]}"
    unique2 = f"fts_test_cats_{uuid.uuid4().hex[:8]}"

    episodic_store(f"The quick brown {unique1} jumps over the lazy dog", tags=["fts_test"])
    episodic_store(f"A completely different sentence about {unique2}", tags=["fts_test"])

    results = episodic_search_fts(unique1, k=5)
    assert len(results["memories"]) == 1
    assert unique1 in results["memories"][0]["text"]
    print(f"  Found: {results['memories'][0]['text']}")

    results = episodic_search_fts(unique2, k=5)
    assert len(results["memories"]) == 1
    assert unique2 in results["memories"][0]["text"]
    print(f"  Found: {results['memories'][0]['text']}")

    print("  ✓ FTS5 search passed")


def test_stats():
    """Test statistics endpoint."""
    print("\nTesting statistics...")

    stats = episodic_stats()
    assert "total_memories" in stats
    assert "index_size" in stats
    assert "embedding_dim" in stats
    assert "embedding_model" in stats
    assert stats["embedding_dim"] == 384
    assert stats["embedding_model"] == "all-MiniLM-L6-v2"
    print(f"  Stats: {stats}")
    print("  ✓ Statistics passed")


def test_persistence():
    """Test that data persists across instances."""
    print("\nTesting persistence...")

    # Store a memory
    result = episodic_store("Persistence test memory", tags=["persist"])
    memory_id = result["id"]

    # Create a new memory instance (simulating restart)
    from episodic_memory import EpisodicMemory
    new_memory = EpisodicMemory()

    # Fetch using new instance
    fetched = new_memory.fetch([memory_id])
    assert len(fetched["memories"]) == 1
    assert fetched["memories"][0]["text"] == "Persistence test memory"
    print(f"  Persisted memory: {fetched['memories'][0]['text']}")

    # Test recall with new instance
    results = new_memory.recall_fuzzy("persistence test", k=5)
    assert len(results["results"]) > 0
    print("  ✓ Persistence passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Episodic Memory Tests")
    print("=" * 60)

    try:
        test_basic_store_and_fetch()
        test_fuzzy_recall()
        test_noise_ratio()
        test_tag_filtering()
        test_fts_search()
        test_stats()
        test_persistence()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)