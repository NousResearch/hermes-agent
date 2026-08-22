"""Tests for the ByteRover memory provider config gates."""

import threading

import pytest

from agent.memory_manager import MemoryManager
from plugins.memory.byterover import ByteRoverMemoryProvider


def test_auto_extract_false_skips_sync_turn(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.sync_turn("please remember this detail", "acknowledged")

    assert calls == []
    assert provider._sync_thread is None


def test_timeout_query_defaults_to_ten_seconds():
    provider = ByteRoverMemoryProvider({})

    assert provider._timeout_query == 10.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1.0), (1.5, 1.5), ("2.5", 2.5), (0.01, 0.01), (3600, 3600.0)],
)
def test_valid_timeout_query_values_are_propagated(value, expected):
    provider = ByteRoverMemoryProvider({"timeout_query": value})

    assert provider._timeout_query == expected


@pytest.mark.parametrize(
    "value",
    [True, False, "invalid", 0, -1, float("nan"), float("inf"), float("-inf"), 3600.1],
)
def test_invalid_timeout_query_falls_back_to_ten_seconds(value, caplog):
    provider = ByteRoverMemoryProvider({"timeout_query": value})

    assert provider._timeout_query == 10.0
    assert "Invalid memory.byterover.timeout_query" in caplog.text
    assert "using default 10.0 seconds" in caplog.text


def test_timeout_query_reaches_prefetch_and_tool_query(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"timeout_query": "17.5"})
    provider.initialize("session-1")

    def fake_run_brv(*args, **kwargs):
        calls.append((args, kwargs))
        return {"success": True, "output": "relevant memory content"}

    monkeypatch.setattr("plugins.memory.byterover._run_brv", fake_run_brv)

    assert provider.prefetch("a sufficiently long query")
    assert provider._tool_query({"query": "a sufficiently long query"})
    assert [kwargs["timeout"] for _, kwargs in calls] == [17.5, 17.5]


@pytest.mark.parametrize(
    ("prefetch_timeout", "timeout_query", "expected_deadline"),
    [(8.0, 10.0, 8.0), (12.0, 10.0, 10.0)],
    ids=["prefetch-8-query-10", "prefetch-12-query-10"],
)
def test_prefetch_all_returns_context_with_minimum_effective_deadline(
    prefetch_timeout,
    timeout_query,
    expected_deadline,
    monkeypatch,
):
    calls = []
    join_timeouts = []
    provider = ByteRoverMemoryProvider({"timeout_query": timeout_query})
    provider.initialize("session-1")
    manager = MemoryManager(external_prefetch_timeout=prefetch_timeout)
    manager.add_provider(provider)

    original_join = threading.Thread.join

    def record_join(thread, timeout=None):
        join_timeouts.append(timeout)
        return original_join(thread, timeout)

    def fake_run_brv(*args, **kwargs):
        calls.append((args, kwargs))
        return {"success": True, "output": "relevant memory content"}

    monkeypatch.setattr("agent.memory_manager.threading.Thread.join", record_join)
    monkeypatch.setattr("plugins.memory.byterover._run_brv", fake_run_brv)

    context = manager.prefetch_all("a sufficiently long query")

    assert context == "## ByteRover Context\nrelevant memory content"
    assert join_timeouts == [prefetch_timeout]
    assert [kwargs["timeout"] for _, kwargs in calls] == [timeout_query]
    assert min(manager._external_prefetch_timeout, provider._timeout_query) == expected_deadline


def test_prefetch_keeps_runtime_docstring():
    assert ByteRoverMemoryProvider.prefetch.__doc__ is not None
    assert "default 10s" in ByteRoverMemoryProvider.prefetch.__doc__


def test_timeout_query_schema_registers_default_and_bounds():
    provider = ByteRoverMemoryProvider({})

    schema = next(
        field for field in provider.get_config_schema()
        if field["key"] == "timeout_query"
    )

    assert schema["default"] == 10.0
    assert schema["type"] == "number"
    assert schema["minimum"] == 0.01
    assert schema["maximum"] == 3600.0


