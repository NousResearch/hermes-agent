"""Regression coverage for holographic fact retrieval counters."""

from __future__ import annotations

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic import holographic as hrr


def _make_provider(tmp_path):
    provider = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    provider.initialize(session_id="test-session")
    return provider


def _call_fact_store(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _counts_by_id(provider):
    return {
        fact["fact_id"]: fact["retrieval_count"]
        for fact in provider._store.list_facts(limit=20)
    }


def test_search_increments_retrieval_count_for_returned_facts(tmp_path):
    provider = _make_provider(tmp_path)
    try:
        kept = _call_fact_store(
            provider,
            action="add",
            content='"Hermes" records holographic memory retrieval counts.',
            category="tool",
        )["fact_id"]
        other = _call_fact_store(
            provider,
            action="add",
            content='"Cafe24" deployment cache behavior is unrelated.',
            category="tool",
        )["fact_id"]

        result = _call_fact_store(
            provider,
            action="search",
            query="Hermes holographic retrieval",
            limit=1,
        )

        assert [fact["fact_id"] for fact in result["results"]] == [kept]
        assert _counts_by_id(provider) == {kept: 1, other: 0}

        _call_fact_store(
            provider,
            action="search",
            query="Hermes holographic retrieval",
            limit=1,
        )
        assert _counts_by_id(provider) == {kept: 2, other: 0}
    finally:
        provider.shutdown()


@pytest.mark.parametrize(
    ("args"),
    [
        {"action": "probe", "entity": "Hermes", "limit": 2},
        {"action": "related", "entity": "Hermes", "limit": 2},
        {"action": "reason", "entities": ["Hermes", "Memory"], "limit": 2},
    ],
)
def test_structural_retrieval_increments_returned_fact_counts(tmp_path, args):
    if not hrr._HAS_NUMPY:
        pytest.skip("structural holographic retrieval requires numpy")

    provider = _make_provider(tmp_path)
    try:
        _call_fact_store(
            provider,
            action="add",
            content='"Hermes" and "Memory" should update retrieval counters.',
            category="tool",
        )
        _call_fact_store(
            provider,
            action="add",
            content='"Hermes" and "Cron" share operational context.',
            category="tool",
        )

        result = _call_fact_store(provider, **args)
        returned_ids = {fact["fact_id"] for fact in result["results"]}

        assert returned_ids
        counts = _counts_by_id(provider)
        for fact_id in returned_ids:
            assert counts[fact_id] == 1
    finally:
        provider.shutdown()
