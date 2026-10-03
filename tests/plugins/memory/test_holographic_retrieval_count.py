"""Tests that retrieval actually increments facts.retrieval_count (#126181).

retrieval_count was declared in the schema and selected by list_facts() and
FactRetriever, but no code path incremented it — the metric read 0 forever.
The fix batches one UPDATE per retrieval at the two result exits (search()
and _rank_by_vector(), which serves probe/related/reason) and degrades to a
no-op when the handle can't write (read-only database, cross-process lock
timeout) instead of failing the search.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    """Each test starts and ends with an empty shared-connection registry."""
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


DEPLOY_FACT = "The Thursday deployment rollback failed because of stale migration state."


@pytest.fixture
def store_and_retriever(tmp_path):
    store = MemoryStore(str(tmp_path / "memory_store.db"))
    store.add_fact(content=DEPLOY_FACT, category="project")
    store.add_fact(content='John Doe owns the "Aurora Project" release calendar.', category="people")
    yield store, FactRetriever(store=store)
    store.close()


def _count(store: MemoryStore, content: str) -> int:
    return store._one("SELECT retrieval_count FROM facts WHERE content = ?", (content,))["retrieval_count"]


def test_search_increments_retrieval_count(store_and_retriever):
    store, retriever = store_and_retriever
    results = retriever.search("what happened with the deployment rollback")
    assert len(results) >= 1
    assert _count(store, DEPLOY_FACT) == 1
    retriever.search("what happened with the deployment rollback")
    assert _count(store, DEPLOY_FACT) == 2


def test_search_without_hits_leaves_counts_at_zero(store_and_retriever):
    store, retriever = store_and_retriever
    assert retriever.search("quantum teapot ledger") == []
    assert _count(store, DEPLOY_FACT) == 0


def test_probe_vector_path_increments_retrieval_count(store_and_retriever):
    store, retriever = store_and_retriever
    results = retriever.probe("John Doe")
    assert len(results) >= 1
    assert _count(store, 'John Doe owns the "Aurora Project" release calendar.') == 1


def test_record_retrievals_dedupes_and_counts_rows(store_and_retriever):
    store, _ = store_and_retriever
    fact_id = store._one("SELECT fact_id FROM facts WHERE content = ?", (DEPLOY_FACT,))["fact_id"]
    assert store.record_retrievals([fact_id, fact_id]) == 1
    assert _count(store, DEPLOY_FACT) == 1
    assert store.record_retrievals([fact_id, 999999]) == 1  # unknown id matches no row
    assert _count(store, DEPLOY_FACT) == 2


def test_record_retrievals_empty_input_is_noop(store_and_retriever):
    store, _ = store_and_retriever
    assert store.record_retrievals([]) == 0
    assert store.record_retrievals(None) == 0


def test_record_retrievals_degrades_on_unwritable_handle(store_and_retriever, monkeypatch):
    """A read-only handle or lock timeout must not raise out of the search path."""
    store, retriever = store_and_retriever

    def _readonly(*_args, **_kwargs):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(store, "_write", _readonly)
    assert store.record_retrievals([1, 2]) == 0
    results = retriever.search("what happened with the deployment rollback")  # search still succeeds
    assert len(results) >= 1
