"""Tests for GH-84772: MCP tools must be directly dispatchable after lazy load.

With tool_search progressive disclosure active, MCP/plugin tools are hidden
behind the tool_search/tool_describe/tool_call bridge:
``agent.valid_tool_names`` only holds the post-assembly surface (core tools +
the bridge). The conversation-loop dispatch gate therefore rejected direct
``mcp__<server>__<tool>`` calls with "Tool does not exist", while
``tool_call(name=...)`` kept working — a desync between the two dispatch
paths.

Fix: the dispatch gate unions the session's pre-assembly catalog (the same
source the bridge reads, ``skip_tool_search_assembly=True``), so the main
dispatch and the bridge share one source of truth by construction.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_runtime_helpers import dispatch_valid_tool_names
from agent.conversation_loop import _invalid_tool_name_error_content

import model_tools


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_CORE_TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read a file"}}
_MCP_TOOL = {
    "type": "function",
    "function": {"name": "mcp__automem__store_memory", "description": "Store a memory"},
}
_MCP_TOOL_2 = {
    "type": "function",
    "function": {"name": "mcp__automem__recall", "description": "Recall memories"},
}

# Post-assembly surface: core tools + the three bridge tools. MCP tools are
# NOT here — that is exactly the buggy shape the dispatch gate used to see.
_POST_ASSEMBLY_NAMES = {
    "read_file",
    "tool_search",
    "tool_describe",
    "tool_call",
}

# Pre-assembly catalog: core + MCP tools (no bridge).
_PRE_ASSEMBLY_CATALOG = [_CORE_TOOL, _MCP_TOOL, _MCP_TOOL_2]


def _fake_agent(**overrides) -> SimpleNamespace:
    """Minimal agent stand-in with the attributes the gate reads."""
    base = dict(
        valid_tool_names=set(_POST_ASSEMBLY_NAMES),
        enabled_toolsets=None,
        disabled_toolsets=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# model_tools.get_session_tool_names — the unified pre-assembly source
# ---------------------------------------------------------------------------

def test_get_session_tool_names_uses_pre_assembly_catalog():
    """get_session_tool_names reads the same catalog the bridge reads."""
    seen = {}

    def _fake_get_definitions(**kwargs):
        seen.update(kwargs)
        return list(_PRE_ASSEMBLY_CATALOG)

    with patch.object(model_tools, "get_tool_definitions", side_effect=_fake_get_definitions):
        names = model_tools.get_session_tool_names(
            enabled_toolsets=["mcp-automem"],
            disabled_toolsets=["desktop"],
        )

    assert "mcp__automem__store_memory" in names
    assert "mcp__automem__recall" in names
    assert "read_file" in names
    # The single source of truth for both dispatch paths:
    assert seen.get("skip_tool_search_assembly") is True
    assert seen.get("quiet_mode") is True
    assert seen.get("enabled_toolsets") == ["mcp-automem"]
    assert seen.get("disabled_toolsets") == ["desktop"]


def test_get_session_tool_names_empty_on_failure():
    """A catalog failure degrades to an empty set, never an exception."""
    with patch.object(model_tools, "get_tool_definitions", side_effect=RuntimeError("boom")):
        names = model_tools.get_session_tool_names()
    assert names == set()


# ---------------------------------------------------------------------------
# model_tools.has_deferrable_tools — cheap gate
# ---------------------------------------------------------------------------

def test_has_deferrable_tools_from_last_assembly():
    """True only when the last assembly actually deferred tools."""
    with patch.object(model_tools, "_last_tool_search_assembly", SimpleNamespace(activated=True)):
        assert model_tools.has_deferrable_tools() is True
    with patch.object(model_tools, "_last_tool_search_assembly", SimpleNamespace(activated=False)):
        assert model_tools.has_deferrable_tools() is False
    with patch.object(model_tools, "_last_tool_search_assembly", None):
        with patch("tools.tool_search.load_config") as fake_load:
            fake_load.return_value = SimpleNamespace(enabled="on")
            assert model_tools.has_deferrable_tools() is True
            fake_load.return_value = SimpleNamespace(enabled="off")
            assert model_tools.has_deferrable_tools() is False


# ---------------------------------------------------------------------------
# dispatch_valid_tool_names — the gate's universe
# ---------------------------------------------------------------------------

def test_dispatch_valid_tool_names_unions_deferred_catalog():
    """With deferral active, the gate accepts direct mcp__ tool calls."""
    agent = _fake_agent()
    with patch.object(model_tools, "has_deferrable_tools", return_value=True), patch.object(
        model_tools,
        "get_session_tool_names",
        return_value={"mcp__automem__store_memory", "mcp__automem__recall"},
    ):
        names = dispatch_valid_tool_names(agent)

    # Direct dispatch of the deferred tool now validates...
    assert "mcp__automem__store_memory" in names
    assert "mcp__automem__recall" in names
    # ...while the post-assembly surface (core + bridge) is preserved.
    assert _POST_ASSEMBLY_NAMES <= names


def test_dispatch_valid_tool_names_skips_union_without_deferral():
    """No deferral → post-assembly names only; no extra catalog read."""
    agent = _fake_agent()
    with patch.object(model_tools, "has_deferrable_tools", return_value=False), patch.object(
        model_tools, "get_session_tool_names", return_value={"mcp__automem__store_memory"}
    ) as fake_session:
        names = dispatch_valid_tool_names(agent)

    assert names == _POST_ASSEMBLY_NAMES
    fake_session.assert_not_called()


def test_dispatch_valid_tool_names_falls_back_on_error():
    """A gate-computation failure never breaks dispatch (pre-fix behavior)."""
    agent = _fake_agent()
    with patch.object(model_tools, "has_deferrable_tools", side_effect=RuntimeError("boom")):
        names = dispatch_valid_tool_names(agent)
    assert names == _POST_ASSEMBLY_NAMES


def test_dispatch_valid_tool_names_scopes_to_session_toolsets():
    """The union is scoped to the session's enabled/disabled toolsets."""
    agent = _fake_agent(enabled_toolsets=["mcp-automem"], disabled_toolsets=["desktop"])
    seen = {}

    def _fake_session_names(enabled_toolsets=None, disabled_toolsets=None):
        seen["enabled_toolsets"] = enabled_toolsets
        seen["disabled_toolsets"] = disabled_toolsets
        return {"mcp__automem__store_memory"}

    with patch.object(model_tools, "has_deferrable_tools", return_value=True), patch.object(
        model_tools, "get_session_tool_names", side_effect=_fake_session_names
    ):
        names = dispatch_valid_tool_names(agent)

    assert "mcp__automem__store_memory" in names
    assert seen["enabled_toolsets"] == ["mcp-automem"]
    assert seen["disabled_toolsets"] == ["desktop"]


# ---------------------------------------------------------------------------
# Error listing — deferred tools must appear in "Available tools"
# ---------------------------------------------------------------------------

def test_invalid_tool_error_lists_deferred_tools():
    """The model-facing error now advertises the deferred catalog too."""
    agent = _fake_agent()
    with patch.object(model_tools, "has_deferrable_tools", return_value=True), patch.object(
        model_tools,
        "get_session_tool_names",
        return_value={"mcp__automem__store_memory", "mcp__automem__recall"},
    ):
        names = dispatch_valid_tool_names(agent)

    content = _invalid_tool_name_error_content("totally_bogus_tool", names)
    assert "Tool 'totally_bogus_tool' does not exist" in content
    assert "mcp__automem__store_memory" in content
    assert "mcp__automem__recall" in content
    assert "tool_call" in content


# ---------------------------------------------------------------------------
# Bridge path — tool_call via tool_search bridge still dispatches MCP tools
# ---------------------------------------------------------------------------

def test_tool_call_bridge_still_dispatches_mcp_tools():
    """tool_call(name=...) unwraps to the underlying MCP tool (regression)."""
    dispatched: list[str] = []

    def _fake_resolve_underlying_call(args):
        return "mcp__automem__store_memory", {"memory": "hello"}, None

    with patch("tools.tool_search.is_bridge_tool", side_effect=lambda n: n == "tool_call"), patch(
        "tools.tool_search.resolve_underlying_call", side_effect=_fake_resolve_underlying_call
    ), patch(
        "tools.tool_search.scoped_deferrable_names",
        return_value=frozenset({"mcp__automem__store_memory"}),
    ), patch("tools.tool_search.validate_deferred_call_args", return_value=None), patch.object(
        model_tools, "get_tool_definitions", return_value=list(_PRE_ASSEMBLY_CATALOG)
    ), patch.object(
        model_tools.registry, "dispatch", side_effect=lambda name, args, **kw: dispatched.append(name) or '{"ok": true}'
    ):
        result = model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "mcp__automem__store_memory", "arguments": {"memory": "hello"}},
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    # The bridge unwrapped to the real tool and the main registry dispatched
    # it — the same registry the direct path now reaches (#84772).
    assert dispatched == ["mcp__automem__store_memory"]
    assert '"ok": true' in result
