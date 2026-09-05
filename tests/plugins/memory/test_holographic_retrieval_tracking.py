"""retrieval_count must reflect facts actually returned to the caller.

Invariant: every query path that surfaces facts to the agent (search, probe,
related, reason) increments retrieval_count for the returned rows. The column
feeds trust-weight maintenance and retention decisions — with no writer, every
fact reads 0 forever, so actively-used facts are indistinguishable from
never-read ones and retention deletes the wrong rows.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"))
    for i in range(8):
        s.add_fact(
            content=f"deploy target {i} setting alpha beta gamma option {i % 7}",
            category="fact" if i % 2 else "preference",
            tags=f"entity_{i % 5} deploy",
        )
    yield s
    s.close()


def _counts(s: MemoryStore, ids: list[int]) -> list[int]:
    ph = ",".join("?" * len(ids))
    return [r["retrieval_count"] for r in
            s._conn.execute(f"SELECT retrieval_count FROM facts WHERE fact_id IN ({ph}) ORDER BY fact_id", ids)]


def test_search_bumps_retrieval_count(store):
    r = FactRetriever(store=store)
    results = r.search("deploy target setting")
    assert results, "FTS path returned nothing — fixture broken"
    ids = [f["fact_id"] for f in results]
    assert all(c >= 1 for c in _counts(store, ids)), \
        "search() results still read retrieval_count = 0 — facts are invisible to retention"


def test_probe_related_reason_bump_retrieval_count(store):
    r = FactRetriever(store=store)
    for method, call in (("probe", lambda: r.probe("entity_1")),
                         ("related", lambda: r.related("entity_1")),
                         ("reason", lambda: r.reason(["entity_1", "alpha"]))):
        results = call()
        assert results, f"{method}() returned nothing — fixture broken"
        ids = [f["fact_id"] for f in results]
        assert all(c >= 1 for c in _counts(store, ids)), \
            f"{method}() results still read retrieval_count = 0 — vector paths invisible to retention"