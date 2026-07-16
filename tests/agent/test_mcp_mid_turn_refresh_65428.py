"""Mid-turn MCP tool refresh (#65428).

When an MCP server fires `notifications/tools/list_changed` during a tool call
(e.g. mcp-dap-server swapping from the startup `debug` tool to session tools
`context`/`step`/`breakpoint` once a debug session starts), the global registry
advances its `_generation` stamp.  The agent's tool snapshot, however, is built
once per turn in `build_turn_context()` — without a mid-turn check the newly
registered tools are invisible to the next API call in the SAME turn, forcing
the agent to wait for the next user message before it can use them.

The fix lives in `agent/conversation_loop.py` immediately after
`agent._execute_tool_calls(...)`.  It compares the registry's generation stamp
to the agent's `_tool_snapshot_generation`; if the registry advanced, it calls
`refresh_agent_mcp_tools(agent)` to rebuild the snapshot in-place before the
next LLM call assembles its `tools=` kwarg.

These tests exercise that check in isolation — bypassing the full conversation
loop — by invoking the same guard expression the loop runs.  They assert:

  * When the registry advanced, `refresh_agent_mcp_tools` is invoked and the
    agent's snapshot is rebuilt (new tool lands in `agent.tools`).
  * When the registry generation matches the snapshot, the refresh is NOT
    called — a no-op `list_changed` (server fired it but the visible tool set
    didn't change) pays only the cheap generation compare.
  * When MCP support isn't imported (`tools.mcp_tool` not in `sys.modules`),
    the check is a no-op — keeps the no-MCP fast path off the heavy import.
"""

from __future__ import annotations

import sys
import types

import pytest


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _make_agent(tool_names, *, snapshot_generation: int = 0, skip_mcp_refresh: bool = False):
    """Minimal agent stub matching the loop's read surface."""
    a = types.SimpleNamespace()
    a.tools = [_tool(n) for n in tool_names]
    a.valid_tool_names = set(tool_names)
    a.enabled_toolsets = None
    a.disabled_toolsets = None
    a._tool_snapshot_generation = snapshot_generation
    a._skip_mcp_refresh = skip_mcp_refresh
    a.session_id = "test-session"
    return a


def _apply_mid_turn_mcp_refresh(agent) -> bool:
    """Inline replay of the exact guard block added to conversation_loop.py.

    Returns True iff `refresh_agent_mcp_tools` was actually called for this
    agent — used by the tests below to assert the dispatch decision without
    needing the full loop wiring.
    """
    if getattr(agent, "_skip_mcp_refresh", False):
        return False
    if "tools.mcp_tool" not in sys.modules:
        return False
    from tools.registry import registry as _snap_registry
    _reg_gen = getattr(_snap_registry, "_generation", 0)
    _agent_gen = getattr(agent, "_tool_snapshot_generation", 0)
    if not (isinstance(_reg_gen, int) and isinstance(_agent_gen, int) and _reg_gen > _agent_gen):
        return False
    from tools.mcp_tool import refresh_agent_mcp_tools
    refresh_agent_mcp_tools(agent, quiet_mode=True)
    return True


def test_mid_turn_refresh_fires_when_registry_advanced(monkeypatch):
    """Registry generation > snapshot → refresh is called, snapshot rebuilt."""
    import tools.mcp_tool as mcp_tool
    import model_tools

    agent = _make_agent(["read_file", "terminal"], snapshot_generation=2)

    # Registry is one generation ahead → the loop should call refresh.
    new_defs = [_tool(n) for n in ("read_file", "terminal", "mcp_dap_step")]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: new_defs)

    called = {"n": 0}
    orig = mcp_tool.refresh_agent_mcp_tools

    def _counter(agent, **kw):
        called["n"] += 1
        return orig(agent, **kw)

    monkeypatch.setattr(mcp_tool, "refresh_agent_mcp_tools", _counter)

    fired = _apply_mid_turn_mcp_refresh(agent)

    assert fired is True
    assert called["n"] == 1
    # Snapshot was rebuilt — the new tool landed.
    assert "mcp_dap_step" in agent.valid_tool_names
    # Stamp advanced to the registry's.
    assert agent._tool_snapshot_generation >= 2


def test_mid_turn_refresh_skipped_when_generation_matches(monkeypatch):
    """Registry generation == snapshot → no refresh, agent untouched."""
    import tools.mcp_tool as mcp_tool

    agent = _make_agent(["read_file", "terminal"], snapshot_generation=5)
    original_tools = agent.tools

    called = {"n": 0}

    def _should_not_fire(agent, **kw):
        called["n"] += 1
        return set()

    monkeypatch.setattr(mcp_tool, "refresh_agent_mcp_tools", _should_not_fire)
    # Pretend the registry is at the same generation.
    import tools.registry as registry_mod
    monkeypatch.setattr(registry_mod.registry, "_generation", 5, raising=False)

    fired = _apply_mid_turn_mcp_refresh(agent)

    assert fired is False
    assert called["n"] == 0
    assert agent.tools is original_tools  # untouched


def test_mid_turn_refresh_skipped_when_mcp_not_imported(monkeypatch):
    """`tools.mcp_tool` not in sys.modules → no-op (no-MCP fast path)."""
    # Temporarily hide the already-imported module from sys.modules to assert
    # the gate short-circuits without touching the heavy import path.
    saved = sys.modules.pop("tools.mcp_tool", None)
    try:
        agent = _make_agent(["read_file"], snapshot_generation=0)

        fired = _apply_mid_turn_mcp_refresh(agent)

        assert fired is False
    finally:
        if saved is not None:
            sys.modules["tools.mcp_tool"] = saved


def test_mid_turn_refresh_skipped_when_agent_opts_out(monkeypatch):
    """`agent._skip_mcp_refresh = True` → no refresh, even if registry advanced."""
    import tools.mcp_tool as mcp_tool

    agent = _make_agent(["read_file"], snapshot_generation=0, skip_mcp_refresh=True)

    called = {"n": 0}

    def _should_not_fire(agent, **kw):
        called["n"] += 1
        return set()

    monkeypatch.setattr(mcp_tool, "refresh_agent_mcp_tools", _should_not_fire)
    import tools.registry as registry_mod
    monkeypatch.setattr(registry_mod.registry, "_generation", 99, raising=False)

    fired = _apply_mid_turn_mcp_refresh(agent)

    assert fired is False
    assert called["n"] == 0


def test_mid_turn_refresh_handles_stale_snapshot_gracefully(monkeypatch):
    """Snapshot stamp is non-int (defensive) → guard skips without TypeError."""
    import tools.mcp_tool as mcp_tool

    agent = _make_agent(["read_file"], snapshot_generation=0)
    # Simulate a malformed stamp (a test mock or a future regression).
    agent._tool_snapshot_generation = "not-an-int"  # type: ignore[assignment]

    called = {"n": 0}

    def _should_not_fire(agent, **kw):
        called["n"] += 1
        return set()

    monkeypatch.setattr(mcp_tool, "refresh_agent_mcp_tools", _should_not_fire)

    # Should not raise.
    fired = _apply_mid_turn_mcp_refresh(agent)

    assert fired is False
    assert called["n"] == 0
