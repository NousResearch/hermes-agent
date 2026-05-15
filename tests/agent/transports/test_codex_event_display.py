"""Tests for codex_event_display.make_progress_bridge — codex item/* events
into Hermes' progress_callback channel.

Drives the bridge against realistic notification shapes (captured from
codex 0.130.0 for commandExecution; synthetic but schema-accurate for
the other item types).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.transports.codex_event_display import (
    _INTERNAL_MCP_SERVER,
    make_progress_bridge,
)


# ----------------------------------------------------------------------
# Fixtures: realistic codex notification shapes
# ----------------------------------------------------------------------

COMMAND_EXEC_STARTED = {
    "method": "item/started",
    "params": {
        "item": {
            "type": "commandExecution",
            "id": "f8a75c66-a89e-4fd7-8bcf-2d58e664fa9e",
            "command": "/bin/bash -lc 'ls /tmp'",
            "cwd": "/tmp",
            "source": "userShell",
        },
    },
}

COMMAND_EXEC_COMPLETED = {
    "method": "item/completed",
    "params": {
        "item": {
            "type": "commandExecution",
            "id": "f8a75c66-a89e-4fd7-8bcf-2d58e664fa9e",
            "command": "/bin/bash -lc 'ls /tmp'",
            "cwd": "/tmp",
            "status": "completed",
            "aggregatedOutput": "file1\nfile2",
            "exitCode": 0,
        },
    },
}

FILE_CHANGE_STARTED = {
    "method": "item/started",
    "params": {
        "item": {
            "type": "fileChange",
            "id": "fc-1",
            "changes": [
                {"kind": {"type": "add"}, "path": "/tmp/new.py"},
                {"kind": {"type": "update"}, "path": "/tmp/old.py"},
            ],
        },
    },
}

MCP_TOOL_CALL_STARTED_USER_SERVER = {
    "method": "item/started",
    "params": {
        "item": {
            "type": "mcpToolCall",
            "id": "mcp-1",
            "server": "filesystem",
            "tool": "read_file",
            "arguments": {"path": "/home/jake/notes.md"},
        },
    },
}

MCP_TOOL_CALL_STARTED_HERMES_TOOLS = {
    "method": "item/started",
    "params": {
        "item": {
            "type": "mcpToolCall",
            "id": "mcp-2",
            "server": _INTERNAL_MCP_SERVER,
            "tool": "web_search",
            "arguments": {"query": "rust borrow checker"},
        },
    },
}

DYNAMIC_TOOL_CALL_STARTED = {
    "method": "item/started",
    "params": {
        "item": {
            "type": "dynamicToolCall",
            "id": "dyn-1",
            "tool": "git_status",
            "arguments": {"cwd": "/home/jake/project"},
        },
    },
}

WEB_SEARCH_STARTED = {
    "method": "item/started",
    "params": {
        "item": {
            "type": "webSearch",
            "id": "ws-1",
            "query": "openai codex sdk release notes",
        },
    },
}

REASONING_COMPLETED = {
    "method": "item/completed",
    "params": {
        "item": {
            "type": "reasoning",
            "id": "r-1",
            "summary": ["thinking about it"],
            "content": [],
        },
    },
}

AGENT_MESSAGE_COMPLETED = {
    "method": "item/completed",
    "params": {
        "item": {
            "type": "agentMessage",
            "id": "am-1",
            "text": "All done.",
        },
    },
}

TURN_STARTED = {
    "method": "turn/started",
    "params": {"threadId": "t-1", "turnId": "u-1"},
}

OUTPUT_DELTA = {
    "method": "item/commandExecution/outputDelta",
    "params": {"itemId": "f8a75c66", "delta": "partial chunk"},
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _capture():
    """Return (bridge, mock) where mock records progress_callback calls.

    The bridge's signature takes a getter, not a callback directly, to
    support per-turn callback swaps. For simple tests we wrap the mock
    in a fixed lambda.
    """
    cb = MagicMock()
    return make_progress_bridge(lambda: cb), cb


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

class TestCommandExecution:
    def test_started_emits_tool_started(self) -> None:
        bridge, cb = _capture()
        bridge(COMMAND_EXEC_STARTED)
        assert cb.call_count == 1
        event_type, tool_name, preview, args = cb.call_args.args
        assert event_type == "tool.started"
        assert tool_name == "exec_command"
        assert "ls /tmp" in preview
        assert args["command"] == "/bin/bash -lc 'ls /tmp'"
        assert args["cwd"] == "/tmp"

    def test_completed_emits_tool_completed(self) -> None:
        bridge, cb = _capture()
        bridge(COMMAND_EXEC_COMPLETED)
        assert cb.call_count == 1
        event_type, tool_name, _preview, _args = cb.call_args.args
        assert event_type == "tool.completed"
        assert tool_name == "exec_command"

    def test_long_command_truncated_in_preview(self) -> None:
        bridge, cb = _capture()
        long_cmd = "echo " + ("x" * 500)
        note = {
            "method": "item/started",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "command": long_cmd,
                    "cwd": "/",
                }
            },
        }
        bridge(note)
        _, _, preview, _ = cb.call_args.args
        # Preview must be bounded — gateway has its own further trimming
        # but the bridge should never emit unbounded strings.
        assert len(preview) < 250


class TestFileChange:
    def test_started_emits_apply_patch_with_summary(self) -> None:
        bridge, cb = _capture()
        bridge(FILE_CHANGE_STARTED)
        assert cb.call_count == 1
        event_type, tool_name, preview, args = cb.call_args.args
        assert event_type == "tool.started"
        assert tool_name == "apply_patch"
        # Preview should mention what's changing
        assert "1 add" in preview
        assert "1 update" in preview
        # First path surfaces in preview
        assert "/tmp/new.py" in preview
        # Args carry the full kind/path summary
        assert len(args["changes"]) == 2
        assert args["changes"][0]["kind"] == "add"
        assert args["changes"][0]["path"] == "/tmp/new.py"

    def test_empty_changes_list_does_not_crash(self) -> None:
        bridge, cb = _capture()
        bridge({
            "method": "item/started",
            "params": {"item": {"type": "fileChange", "changes": []}},
        })
        # Still fires — just with a degraded preview
        assert cb.call_count == 1
        _, tool_name, _, _ = cb.call_args.args
        assert tool_name == "apply_patch"


class TestMcpToolCall:
    def test_user_mcp_server_emits_namespaced_display_name(self) -> None:
        bridge, cb = _capture()
        bridge(MCP_TOOL_CALL_STARTED_USER_SERVER)
        assert cb.call_count == 1
        _, tool_name, preview, args = cb.call_args.args
        assert tool_name == "mcp.filesystem.read_file"
        assert "/home/jake/notes.md" in preview
        assert args == {"path": "/home/jake/notes.md"}

    def test_hermes_tools_mcp_server_emits_bare_tool_name(self):
        """When codex calls back into the hermes-tools MCP server (a
        separate subprocess that doesn't have access to the parent
        agent's tool_progress_callback), the codex-level mcpToolCall
        event IS the display event — there is no inner native dispatch
        that fires its own progress event. We surface the bare tool name
        (web_search, browser_*, vision_analyze, ...) instead of the
        ugly mcp.hermes-tools.<tool> namespacing.
        """
        bridge, cb = _capture()
        bridge(MCP_TOOL_CALL_STARTED_HERMES_TOOLS)
        assert cb.call_count == 1, (
            "hermes-tools mcpToolCall events must surface as display "
            "events since the inner dispatch (separate subprocess) can't "
            "fire tool_progress_callback"
        )
        _, tool_name, preview, args = cb.call_args.args
        assert tool_name == "web_search", (
            f"expected bare tool name, got {tool_name!r} — namespacing "
            f"as mcp.hermes-tools.* would be ugly and lose user intent"
        )
        assert "rust borrow checker" in preview
        assert args == {"query": "rust borrow checker"}


class TestDynamicToolCall:
    def test_emits_bare_tool_name(self) -> None:
        bridge, cb = _capture()
        bridge(DYNAMIC_TOOL_CALL_STARTED)
        assert cb.call_count == 1
        _, tool_name, _preview, args = cb.call_args.args
        assert tool_name == "git_status"
        assert args == {"cwd": "/home/jake/project"}


class TestWebSearch:
    def test_emits_web_search_with_query(self) -> None:
        bridge, cb = _capture()
        bridge(WEB_SEARCH_STARTED)
        assert cb.call_count == 1
        _, tool_name, preview, args = cb.call_args.args
        assert tool_name == "web_search"
        assert "codex sdk" in preview
        assert args["query"].startswith("openai codex sdk")


class TestIgnoredEvents:
    """Items that aren't tool calls, or methods we don't surface."""

    def test_reasoning_is_skipped(self) -> None:
        bridge, cb = _capture()
        bridge(REASONING_COMPLETED)
        assert cb.call_count == 0

    def test_agent_message_is_skipped(self) -> None:
        bridge, cb = _capture()
        bridge(AGENT_MESSAGE_COMPLETED)
        assert cb.call_count == 0

    def test_turn_started_is_skipped(self) -> None:
        bridge, cb = _capture()
        bridge(TURN_STARTED)
        assert cb.call_count == 0

    def test_streaming_delta_is_skipped(self) -> None:
        """Per design note: only item/started + item/completed surface,
        not the per-chunk streaming deltas. Matches HA-native UX which
        also doesn't show streaming stdout."""
        bridge, cb = _capture()
        bridge(OUTPUT_DELTA)
        assert cb.call_count == 0


class TestDefensiveBehavior:
    def test_none_progress_callback_returns_safe_noop(self) -> None:
        """When no gateway/CLI has installed a progress callback, the
        bridge must be a no-op rather than crashing the codex transport.

        Also covers the case where the getter starts returning None
        mid-session (e.g. CLI thread tearing down its callback)."""
        bridge = make_progress_bridge(lambda: None)
        # Should not raise on any input
        bridge(COMMAND_EXEC_STARTED)
        bridge(FILE_CHANGE_STARTED)
        bridge({})
        bridge({"method": "garbage"})

    def test_misbehaving_callback_does_not_propagate(self) -> None:
        """The bridge wraps invocations in try/except so a buggy
        progress callback can never crash the codex transport read loop.
        """
        def broken_cb(*_args, **_kwargs):
            raise RuntimeError("display went sideways")

        bridge = make_progress_bridge(lambda: broken_cb)
        # Must not raise
        bridge(COMMAND_EXEC_STARTED)
        bridge(COMMAND_EXEC_COMPLETED)
        bridge(FILE_CHANGE_STARTED)

    def test_misbehaving_getter_does_not_propagate(self) -> None:
        """The getter itself is wrapped — if it raises (e.g. agent
        teardown made the attribute disappear), the bridge still
        shouldn't kill the codex transport."""
        def broken_getter():
            raise RuntimeError("agent attribute gone")

        bridge = make_progress_bridge(broken_getter)
        # Must not raise
        bridge(COMMAND_EXEC_STARTED)

    def test_malformed_note_does_not_crash(self) -> None:
        bridge, cb = _capture()
        # Missing method
        bridge({})
        # Method present but params missing
        bridge({"method": "item/started"})
        # item not a dict
        bridge({"method": "item/started", "params": {"item": "garbage"}})
        # Empty item
        bridge({"method": "item/started", "params": {"item": {}}})
        # Unknown item type
        bridge({
            "method": "item/started",
            "params": {"item": {"type": "weirdNewItem"}},
        })
        assert cb.call_count == 0


class TestStartCompletePairing:
    """Verify the same item produces matching start + complete events.

    Important for the gateway's dedup-and-edit logic: it needs to see
    one tool.started per call followed by one tool.completed with the
    same tool name."""

    def test_command_execution_round_trip(self) -> None:
        bridge, cb = _capture()
        bridge(COMMAND_EXEC_STARTED)
        bridge(COMMAND_EXEC_COMPLETED)
        assert cb.call_count == 2
        started_args = cb.call_args_list[0].args
        completed_args = cb.call_args_list[1].args
        assert started_args[0] == "tool.started"
        assert completed_args[0] == "tool.completed"
        # Same display name for both
        assert started_args[1] == completed_args[1] == "exec_command"


class TestLateBindingAcrossTurns:
    """Regression: CodexAppServerSession lives across turns but the
    gateway's progress_callback is per-turn (each turn has its own
    progress queue, dedup state, and cleanup tracking).

    If the bridge captures the callback at session-construction time,
    tool events on turn N+1 fire into turn N's dead queue — the user
    sees tool bubbles on the first turn that touches tools and nothing
    after. The bridge must late-bind via the getter on every event.

    Surfaced via live testing on Discord, May 15 2026 — first tool turn
    after gateway restart rendered, second turn was silent.
    """

    def test_callback_swap_between_events_is_observed(self) -> None:
        """Same bridge, different callbacks across events. Each event
        must route to whatever callback was current at event time."""
        current: list[Any] = [None]
        bridge = make_progress_bridge(lambda: current[0])

        # Turn 1: install a callback, fire an event
        turn1_calls: list[tuple] = []
        current[0] = lambda *args: turn1_calls.append(args)
        bridge(COMMAND_EXEC_STARTED)
        assert len(turn1_calls) == 1

        # Turn 2: gateway built a fresh closure (new queue), swap it in.
        # The bridge must observe the new callback, not the stale one.
        turn2_calls: list[tuple] = []
        current[0] = lambda *args: turn2_calls.append(args)
        bridge(COMMAND_EXEC_STARTED)
        assert len(turn1_calls) == 1, (
            "turn 1 callback received turn 2's event — bridge captured "
            "the callback instead of late-binding"
        )
        assert len(turn2_calls) == 1, (
            "turn 2 callback received no events — bridge didn't see the "
            "swap"
        )

    def test_getter_returning_none_then_callback_starts_routing(self) -> None:
        """The getter can return None at construction time (no gateway
        attached yet) and later return a real callback. Events arriving
        before the callback is wired are dropped; events after are
        routed correctly. Mirrors the case where a session was created
        in a context without a display attached and a display gets
        attached later."""
        current: list[Any] = [None]
        bridge = make_progress_bridge(lambda: current[0])

        # No callback wired yet
        bridge(COMMAND_EXEC_STARTED)  # silently dropped

        # Callback installed
        captured: list[tuple] = []
        current[0] = lambda *args: captured.append(args)
        bridge(COMMAND_EXEC_STARTED)
        assert len(captured) == 1
