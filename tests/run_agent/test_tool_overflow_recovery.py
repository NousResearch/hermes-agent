"""Tests for session-scoped tool-overflow recovery."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_agent(*tool_names: str) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "gpt-5-mini"
    agent.log_prefix = ""
    agent.tools = [_tool(name) for name in tool_names]
    agent.valid_tool_names = set(tool_names)
    agent._session_disabled_toolsets = set()
    agent._vprint = lambda *a, **k: None
    agent.clarify_callback = None
    return agent


def test_recover_from_tool_overflow_uses_selected_group():
    agent = _make_agent(
        "mcp_arr_stack_a",
        "mcp_arr_stack_b",
        "browser_open",
        "browser_click",
        "terminal",
    )
    toolset_map = {
        "mcp_arr_stack_a": "mcp-arr_stack",
        "mcp_arr_stack_b": "mcp-arr_stack",
        "browser_open": "browser",
        "browser_click": "browser",
        "terminal": "terminal",
    }

    with patch("run_agent.get_toolset_for_tool", side_effect=toolset_map.get):
        browser_label = agent._tool_overflow_group_label("browser", 2)
        agent.clarify_callback = lambda _question, _choices: browser_label
        changed = agent._recover_from_tool_overflow(max_tools=3, actual_tools=5)

    assert changed is True
    assert agent.valid_tool_names == {"mcp_arr_stack_a", "mcp_arr_stack_b", "terminal"}
    assert "browser" in agent._session_disabled_toolsets
    assert len(agent.tools) == 3


def test_recover_from_tool_overflow_prefers_mcp_group_without_prompt():
    agent = _make_agent(
        "mcp_arr_stack_a",
        "mcp_arr_stack_b",
        "mcp_arr_stack_c",
        "browser_open",
        "browser_click",
        "terminal",
    )
    toolset_map = {
        "mcp_arr_stack_a": "mcp-arr_stack",
        "mcp_arr_stack_b": "mcp-arr_stack",
        "mcp_arr_stack_c": "mcp-arr_stack",
        "browser_open": "browser",
        "browser_click": "browser",
        "terminal": "terminal",
    }

    with patch("run_agent.get_toolset_for_tool", side_effect=toolset_map.get):
        changed = agent._recover_from_tool_overflow(max_tools=4, actual_tools=6)

    assert changed is True
    assert agent.valid_tool_names == {"browser_open", "browser_click", "terminal"}
    assert "mcp-arr_stack" in agent._session_disabled_toolsets
    assert len(agent.tools) == 3
