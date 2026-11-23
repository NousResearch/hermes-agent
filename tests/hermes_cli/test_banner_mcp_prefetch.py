"""The banner reads the configured MCP servers ahead of time, once, and falls back to reading them itself."""

import sys
import types
from concurrent.futures import Future

import pytest

from hermes_cli import banner


@pytest.fixture
def status_calls(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(get_mcp_status=lambda configured=None: calls.append(configured) or ["status"])
    monkeypatch.setitem(sys.modules, "tools.mcp_tool_discovery", fake)
    monkeypatch.setattr(banner, "_mcp_servers_prefetch", None)
    return calls


def _ready(value):
    future = Future()
    future.set_result(value)
    return future


def test_prefetched_servers_are_passed_to_the_status_read_once(status_calls, monkeypatch):
    monkeypatch.setattr(banner, "_mcp_servers_prefetch", _ready({"docs": {"url": "http://x"}}))
    assert banner._probe_mcp_status() == ["status"]
    assert status_calls == [{"docs": {"url": "http://x"}}]
    banner._probe_mcp_status()  # a later caller never gets the stale config
    assert status_calls[-1] is None


def test_a_failed_prefetch_falls_back_to_reading_the_servers_in_the_status_call(status_calls, monkeypatch):
    failed = Future()
    failed.set_exception(RuntimeError("plugin discovery failed"))
    monkeypatch.setattr(banner, "_mcp_servers_prefetch", failed)
    assert banner._probe_mcp_status() == ["status"]
    assert status_calls == [None]


def test_without_a_prefetch_the_status_call_reads_the_servers_itself(status_calls):
    assert banner._probe_mcp_status() == ["status"]
    assert status_calls == [None]
