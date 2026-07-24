"""Behavior contracts for bounded, authority-aware Hindsight recall output."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider
from plugins.memory.hindsight.recall_postprocess import (
    format_recall_results,
    normalize_max_results,
    rank_and_deduplicate,
)


def _result(text, tags=()):
    return SimpleNamespace(text=text, tags=list(tags))


def _client_with_results(results):
    client = MagicMock()
    client.arecall = AsyncMock(return_value=SimpleNamespace(results=list(results)))
    client.aclose = AsyncMock()
    return client


def _provider(tmp_path, monkeypatch, **overrides):
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config.update(overrides)
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    provider = HindsightMemoryProvider()
    provider.initialize(
        session_id="test-session", hermes_home=str(tmp_path), platform="cli"
    )
    return provider


def test_exact_and_atomic_name_variants_collapse_to_authoritative_result():
    rows = [
        _result("User's name is Alice.", ["session:old"]),
        _result("The user's name is Alice", ["stale:never"]),
        _result("Alice is the user", ["session:new"]),
    ]

    kept = rank_and_deduplicate(rows, max_results=7)

    assert [item.text for item in kept] == ["The user's name is Alice"]
    assert kept[0].authority == "authoritative"


def test_distinct_facts_survive_and_keep_server_order_within_authority_tier():
    rows = [
        _result("Alice lives in Manhattan", ["session:one"]),
        _result("Alice prefers concise responses", ["session:two"]),
        _result("Alice likes dark mode", ["session:three"]),
    ]

    kept = rank_and_deduplicate(rows, max_results=3)

    assert [item.text for item in kept] == [row.text for row in rows]


def test_authoritative_results_rank_before_session_evidence_then_cap():
    rows = [
        _result("Session detail one", ["session:one"]),
        _result("Canonical profile", ["stale:never"]),
        _result("Session detail two", ["session:two"]),
    ]

    kept = rank_and_deduplicate(rows, max_results=2)

    assert [item.text for item in kept] == ["Canonical profile", "Session detail one"]


@pytest.mark.parametrize("value", [0, 21, -1, "bad", True, None])
def test_invalid_max_results_fails_closed(value):
    with pytest.raises(ValueError, match="between 1 and 20"):
        normalize_max_results(value)


def test_formatter_labels_provenance_and_conflict_rule():
    items = rank_and_deduplicate(
        [
            _result("Canonical profile", ["stale:never"]),
            _result("Session detail", ["session:one"]),
        ],
        max_results=7,
    )

    text = format_recall_results(items, numbered=True)

    assert "evidence, not canonical authority" in text
    assert "1. [authoritative] Canonical profile" in text
    assert "2. [session-derived] Session detail" in text


def test_provider_defaults_to_seven_results(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)

    assert provider._recall_max_results == 7


def test_explicit_recall_uses_shared_postprocessor(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch, recall_max_results=2)
    provider._client = _client_with_results([
        _result("Duplicate", ["session:one"]),
        _result("Duplicate.", ["session:two"]),
        _result("Distinct", ["session:three"]),
        _result("Dropped by cap", ["session:four"]),
    ])

    payload = json.loads(
        provider.handle_tool_call("hindsight_recall", {"query": "test"})
    )["result"]

    assert payload.count("Duplicate") == 1
    assert "Distinct" in payload
    assert "Dropped by cap" not in payload


def test_prefetch_uses_shared_postprocessor(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch, recall_max_results=2)
    provider._client = _client_with_results([
        _result("Duplicate", ["session:one"]),
        _result("Duplicate.", ["session:two"]),
        _result("Distinct", ["session:three"]),
        _result("Dropped by cap", ["session:four"]),
    ])

    provider.queue_prefetch("test")
    provider._prefetch_thread.join(timeout=3)
    text = provider.prefetch("test")

    assert text.count("Duplicate") == 1
    assert "Distinct" in text
    assert "Dropped by cap" not in text
