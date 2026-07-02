"""Regression tests for MCP discovery wait before init_agent tool snapshot (#57170)."""

from __future__ import annotations

import inspect
import threading
import time
import types

from agent import agent_init
from run_agent import AIAgent


def test_init_agent_waits_for_mcp_discovery_before_tool_snapshot():
    """init_agent must wait for MCP discovery before snapshotting tools."""
    src = inspect.getsource(agent_init.init_agent)
    wait_idx = src.index("wait_for_mcp_discovery")
    snap_idx = src.index("agent.tools = _ra().get_tool_definitions")
    assert wait_idx < snap_idx


def test_init_agent_calls_wait_for_mcp_discovery(monkeypatch):
    """Behavioral check: init_agent invokes wait_for_mcp_discovery()."""
    order: list[str] = []

    monkeypatch.setattr(
        "hermes_cli.mcp_startup.wait_for_mcp_discovery",
        lambda timeout=None: order.append("wait"),
    )

    def _get_tools(**_kwargs):
        order.append("snapshot")
        return []

    import run_agent

    monkeypatch.setattr(run_agent, "get_tool_definitions", _get_tools)
    monkeypatch.setattr(
        run_agent,
        "OpenAI",
        lambda **_kwargs: types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **_k: None))
        ),
    )

    agent = object.__new__(AIAgent)
    agent_init.init_agent(
        agent,
        base_url="http://test/v1",
        api_key="k",
        provider="openai",
        model="test-model",
        quiet_mode=True,
    )

    assert order[:2] == ["wait", "snapshot"]


def test_slow_mcp_thread_lands_before_snapshot_when_init_agent_waits(monkeypatch):
    """Slow background MCP discovery should complete before tool snapshot."""
    import run_agent

    registered = threading.Event()

    def _slow_discover():
        time.sleep(0.08)
        registered.set()

    thread = threading.Thread(target=_slow_discover, daemon=True)
    thread.start()

    import hermes_cli.mcp_startup as mcp_startup

    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", thread)

    def _get_tools(**_kwargs):
        assert registered.is_set(), "tool snapshot ran before MCP discovery finished"
        return []

    monkeypatch.setattr(run_agent, "get_tool_definitions", _get_tools)
    monkeypatch.setattr(
        run_agent,
        "OpenAI",
        lambda **_kwargs: types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **_k: None))
        ),
    )

    agent = object.__new__(AIAgent)
    agent_init.init_agent(
        agent,
        base_url="http://test/v1",
        api_key="k",
        provider="openai",
        model="test-model",
        quiet_mode=True,
    )

    assert not thread.is_alive()
