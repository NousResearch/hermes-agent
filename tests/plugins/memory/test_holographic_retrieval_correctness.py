"""Regression tests for Holographic retrieval telemetry and trust filtering."""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def test_vector_retrieval_signatures_preserve_positional_limit() -> None:
    expected = {
        "probe": ["self", "entity", "category", "limit", "min_trust"],
        "related": ["self", "entity", "category", "limit", "min_trust"],
        "reason": ["self", "entities", "category", "limit", "min_trust"],
        "contradict": [
            "self",
            "category",
            "threshold",
            "limit",
            "min_trust",
        ],
    }

    for method_name, parameter_names in expected.items():
        method = getattr(FactRetriever, method_name)
        assert list(inspect.signature(method).parameters) == parameter_names


def test_public_retrieval_paths_record_returned_fact_usage(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "retrieval_usage.db"))
    try:
        fact_id = store.add_fact(
            '"Shared Entity" owns the durable alpha configuration.',
            category="general",
        )
        retriever = FactRetriever(store=store)

        result_sets = [
            retriever.search("Shared Entity", limit=10),
            retriever.probe("Shared Entity", category="general", limit=10),
            retriever.related("Shared Entity", category="general", limit=10),
            retriever.reason(["Shared Entity"], category="general", limit=10),
        ]

        for expected_count, results in enumerate(result_sets, start=1):
            assert [result["fact_id"] for result in results] == [fact_id]
            assert results[0]["retrieval_count"] == expected_count

        stored_count = store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()["retrieval_count"]
        assert stored_count == 4
    finally:
        store.close()


def test_vector_retrieval_paths_honor_min_trust(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "retrieval_trust.db"))
    try:
        trusted_id = store.add_fact(
            '"Shared Entity" owns the durable alpha configuration.',
            category="general",
        )
        low_trust_id = store.add_fact(
            '"Shared Entity" owns the stale alpha configuration.',
            category="general",
        )
        assert store.update_fact(low_trust_id, trust_delta=-0.4)
        retriever = FactRetriever(store=store)

        result_sets = [
            retriever.probe(
                "Shared Entity", category="general", min_trust=0.3, limit=10
            ),
            retriever.related(
                "Shared Entity", category="general", min_trust=0.3, limit=10
            ),
            retriever.reason(
                ["Shared Entity"], category="general", min_trust=0.3, limit=10
            ),
        ]

        for results in result_sets:
            assert {result["fact_id"] for result in results} == {trusted_id}

        assert retriever.contradict(min_trust=0.3, threshold=0.0, limit=10) == []
    finally:
        store.close()


def test_provider_passes_trust_threshold_to_every_retrieval_action() -> None:
    class RecordingRetriever:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def record(*args, **kwargs):
                self.calls.append((name, kwargs))
                return []

            return record

    provider = HolographicMemoryProvider(
        config={"min_trust_threshold": 0.7}
    )
    recorder = RecordingRetriever()
    untyped_provider: Any = provider
    untyped_provider._store = object()
    untyped_provider._retriever = recorder

    actions = [
        {"action": "probe", "entity": "Shared Entity"},
        {"action": "related", "entity": "Shared Entity"},
        {"action": "reason", "entities": ["Shared Entity"]},
        {"action": "contradict"},
    ]
    for args in actions:
        response = json.loads(provider._handle_fact_store(args))
        assert response == {"results": [], "count": 0}

    assert [name for name, _ in recorder.calls] == [
        "probe",
        "related",
        "reason",
        "contradict",
    ]
    assert all(kwargs["min_trust"] == 0.7 for _, kwargs in recorder.calls)
