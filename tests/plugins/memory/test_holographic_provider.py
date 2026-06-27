"""Regression tests for the holographic memory provider FTS search."""

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore, _sanitize_fts5_query


def test_sanitize_fts5_query_quotes_hyphenated_terms():
    assert _sanitize_fts5_query("pve-01") == '"pve-01"'
    assert _sanitize_fts5_query('"pve-01"') == '"pve-01"'


def test_store_search_facts_matches_hyphenated_terms(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db")
    store.add_fact(
        "PVE-01 hardware: i5-13500T, IP 10.20.90.00",
        category="hardware",
        tags="pve-01,homelab",
    )

    results = store.search_facts("pve-01", category="hardware", limit=10)

    assert len(results) == 1
    assert results[0]["content"].startswith("PVE-01 hardware")


def test_retriever_search_matches_hyphenated_terms(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db")
    retriever = FactRetriever(store)
    store.add_fact(
        "PVE-01 hardware: i5-13500T, IP 10.20.90.00",
        category="hardware",
        tags="pve-01,homelab",
    )

    results = retriever.search("pve-01", category="hardware", limit=10)

    assert len(results) == 1
    assert results[0]["content"].startswith("PVE-01 hardware")
