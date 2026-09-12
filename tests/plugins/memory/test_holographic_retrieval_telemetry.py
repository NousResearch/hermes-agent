"""Regression tests for holographic retrieval usage telemetry."""

from __future__ import annotations

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.store import MemoryStore


def _provider(tmp_path) -> HolographicMemoryProvider:
    provider = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    provider.initialize(session_id="telemetry-test")
    return provider


def _store(provider: HolographicMemoryProvider) -> MemoryStore:
    assert provider._store is not None
    return provider._store


def _retrieval_count(provider: HolographicMemoryProvider, fact_id: int) -> int:
    row = _store(provider)._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    return row["retrieval_count"]


def test_prefetch_counts_injected_facts_and_reports_recall_status(tmp_path):
    provider = _provider(tmp_path)
    try:
        first = _store(provider).add_fact("H59 requires a live check before advising action.")
        second = _store(provider).add_fact("H59 alerts after a sustained 12-hour elevation.")

        block = provider.prefetch("What is the H59 alert workflow?")

        assert block
        assert _retrieval_count(provider, first) == 1
        assert _retrieval_count(provider, second) == 1
        status = provider.recall_status()
        assert status is not None
        assert status.provider_label == "Holographic"
        assert status.count == 2
    finally:
        provider.shutdown()


def test_manual_search_counts_returned_facts_once_per_retrieval(tmp_path):
    provider = _provider(tmp_path)
    try:
        matched = _store(provider).add_fact("Use an H59 live check first.")
        unmatched = _store(provider).add_fact("The garden watering schedule is weekly.")

        response = provider.handle_tool_call(
            "fact_store", {"action": "search", "query": "H59 live check"}
        )

        assert '"count": 1' in response
        assert _retrieval_count(provider, matched) == 1
        assert _retrieval_count(provider, unmatched) == 0
    finally:
        provider.shutdown()


def test_empty_prefetch_clears_prior_recall_status(tmp_path):
    provider = _provider(tmp_path)
    try:
        _store(provider).add_fact("H59 requires a live check before advising action.")
        assert provider.prefetch("H59 live check")
        assert provider.recall_status() is not None

        assert provider.prefetch("unrelated xylophone taxonomy") == ""
        assert provider.recall_status() is None
    finally:
        provider.shutdown()
