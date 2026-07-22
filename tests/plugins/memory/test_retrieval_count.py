"""Tests for holographic memory retrieval_count increment (#17899).

retrieval_count was never incremented because FactRetriever.search()
bypassed store.search_facts(), the only place that bumped the counter.
"""

import pytest

from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test_memory.db")
    return MemoryStore(db_path=db, default_trust=0.5, hrr_dim=128)


@pytest.fixture
def retriever(store):
    return FactRetriever(store=store, hrr_dim=128, hrr_weight=0.0)


class TestRetrievalCount:
    """retrieval_count should be incremented when facts are retrieved."""

    def test_search_increments_count(self, store, retriever):
        """After searching, retrieval_count should go from 0 to >= 1."""
        fact_id = store.add_fact("Python is a programming language")

        # Verify starting at 0
        row = store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        assert row["retrieval_count"] == 0

        # Search and find it
        results = retriever.search("Python", min_trust=0.0)
        assert len(results) >= 1

        # Verify incremented
        row = store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        assert row["retrieval_count"] >= 1

    def test_multiple_searches_accumulate(self, store, retriever):
        """Multiple searches should keep incrementing."""
        store.add_fact("Rust is memory safe")

        retriever.search("Rust", min_trust=0.0)
        retriever.search("Rust", min_trust=0.0)
        retriever.search("Rust", min_trust=0.0)

        row = store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content LIKE '%Rust%'"
        ).fetchone()
        assert row["retrieval_count"] >= 3

    def test_unrelated_search_no_increment(self, store, retriever):
        """Searching for something else shouldn't increment unrelated facts."""
        fact_id = store.add_fact("Haskell is purely functional")
        store.add_fact("Rust is memory safe")

        retriever.search("Rust", min_trust=0.0)

        row = store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        assert row["retrieval_count"] == 0
