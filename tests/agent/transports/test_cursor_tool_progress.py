"""Tests for Cursor SDK tool progress name resolution."""

from __future__ import annotations

from agent.transports.cursor_sdk_session import CursorSDKSession


def test_completion_prefers_started_tool_name_over_generic_default():
    session = CursorSDKSession()
    session._notify_tool_started(
        call_id="call-1",
        name="shell",
        args={"command": "git status"},
    )
    session._notify_tool_completed(call_id="call-1", name="tool")
    assert not session._active_tool_calls


def test_pop_falls_back_to_single_active_tool():
    session = CursorSDKSession()
    session._notify_tool_started(call_id="call-abc", name="Read", args={"path": "foo.py"})
    session._notify_tool_completed(call_id="", name="")
    assert not session._active_tool_calls


def test_display_tool_name_strips_mcp_prefix():
    assert CursorSDKSession._display_tool_name("mcp_hermes-tools_terminal") == "terminal"
    assert CursorSDKSession._display_tool_name("mcp_notion_notion_search") == "notion_search"


def test_parse_mcp_wrapper_resolves_inner_tool_name():
    _id, name, args = CursorSDKSession._parse_tool_event(
        call_id="call-mcp-1",
        name="mcp",
        tool_call={
            "toolName": "ideas_list",
            "serverName": "hermes-tools",
            "args": {"board": "default"},
        },
    )
    assert _id == "call-mcp-1"
    assert name == "ideas_list"
    assert args.get("board") == "default"


def test_parse_mcp_wrapper_from_nested_payload():
    _id, name, _ = CursorSDKSession._parse_tool_event(
        call_id="call-mcp-2",
        name="mcp",
        tool_call={
            "mcpToolCall": {"name": "web_search", "server": "hermes-tools"},
        },
    )
    assert name == "web_search"


def test_partial_tool_call_upgrades_generic_name():
    session = CursorSDKSession()
    session._notify_tool_started(
        call_id="call-1",
        name="mcp",
        tool_call={"toolName": "ideas_show"},
    )
    session._notify_tool_started(
        call_id="call-1",
        name="mcp",
        tool_call={"toolName": "ideas_show", "args": {"id": "x"}},
    )
    assert session._active_tool_calls["call-1"]["name"] == "ideas_show"
