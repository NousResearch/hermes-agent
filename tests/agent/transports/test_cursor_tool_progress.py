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
