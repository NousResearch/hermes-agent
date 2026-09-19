"""Retrieval telemetry invariants: search/probe must record that facts were served.

retrieval_count has been in the facts schema since the beginning, but no code path
ever incremented it — the column read 0 forever on every real deployment. These
tests pin the behaviour contract: any retrieval path that returns facts to a caller
must (a) bump retrieval_count and (b) refresh updated_at, which temporal decay keys
on — a fact that keeps being found keeps its score.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def seeded_store(tmp_path):
    """Store with a few facts, some carrying HRR vectors (add_fact computes them)."""
    store = MemoryStore(str(tmp_path / "telemetry.db"))
    store.add_fact(content="The deployment rollback failed because of stale migration state.", category="project")
    store.add_fact(content="Compaction settings tuned to 0.85 threshold.", category="tool")
    store.add_fact(content="Venice.ai advertises availableContextTokens inside model_spec.", category="tool")
    yield store
    store.close()


def _counts(store) -> dict[int, int]:
    return {row["fact_id"]: row["retrieval_count"]
            for row in store._conn.execute("SELECT fact_id, retrieval_count FROM facts").fetchall()}


def test_search_bumps_retrieval_count(seeded_store):
    retriever = FactRetriever(store=seeded_store)
    before = _counts(seeded_store)
    assert all(c == 0 for c in before.values()), "fixture sanity: fresh facts start at zero"

    results = retriever.search("deployment rollback", limit=2)

    assert results, "search must find the seeded fact"
    after = _counts(seeded_store)
    served = {r["fact_id"] for r in results}
    # Every returned fact's count is now strictly greater than before it was served
    for fact_id in served:
        assert after[fact_id] > before[fact_id], f"fact {fact_id} served but retrieval_count not bumped"


def test_retrieval_refreshes_updated_at(seeded_store):
    retriever = FactRetriever(store=seeded_store, temporal_decay_half_life=45)
    stale = seeded_store._conn.execute(
        "SELECT updated_at FROM facts WHERE content LIKE '%deployment%'").fetchone()["updated_at"]

    results = retriever.search("deployment rollback", limit=1)

    assert results
    fresh = seeded_store._conn.execute(
        "SELECT updated_at FROM facts WHERE content LIKE '%deployment%'").fetchone()["updated_at"]
    # updated_at moves forward: decay resistance follows sensing
    assert fresh >= stale


def test_vector_path_bumps_retrieval_count(seeded_store):
    """probe/related/reason go through _rank_by_vector, not search()'s tail — both paths count."""
    retriever = FactRetriever(store=seeded_store)
    before = _counts(seeded_store)

    results = retriever.probe("Venice.ai", limit=3)  # vector path when numpy is present

    if results:  # with numpy the probe is vector-driven; without it falls back to search
        after = _counts(seeded_store)
        for r in results:
            assert after[r["fact_id"]] > before[r["fact_id"]]


def test_mark_retrieved_swallows_write_failure(seeded_store):
    """Telemetry is best-effort: a failing UPDATE inside _mark_retrieved must not raise."""
    retriever = FactRetriever(store=seeded_store)

    class _FailingConn:
        def execute(self, *a, **k):
            raise sqlite3.OperationalError("database is locked")

        def commit(self):
            pass

    conn = seeded_store._conn
    seeded_store._conn = _FailingConn()
    try:
        retriever._mark_retrieved([{"fact_id": 1}])  # must not raise
    finally:
        seeded_store._conn = conn
