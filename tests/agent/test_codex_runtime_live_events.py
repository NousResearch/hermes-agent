"""Regression tests for live Codex app-server events.

The history projector is completion-only. These tests protect the parallel
display bridge (make_codex_app_server_event_bridge) that makes deltas and
tool cards visible before resume: it fires both the tool_progress bubbles
AND the authoritative stable-ID tool_start/tool_complete callbacks the TUI
tool cards depend on.

Grafted from PR #65412 (@HaiderSultanArc) onto the merged bridge.
"""

from types import SimpleNamespace

from agent.codex_runtime import (
    _codex_item_completion_payload,
    make_codex_app_server_event_bridge,
    _persist_projected_messages,
)
from agent.transports.codex_event_projector import _deterministic_call_id


def _recording_agent():
    calls = {
        "stream": [],
        "reasoning": [],
        "tool_progress": [],
        "tool_start": [],
        "tool_complete": [],
    }
    agent = SimpleNamespace(
        _fire_stream_delta=lambda text: calls["stream"].append(text),
        _fire_reasoning_delta=lambda text: calls["reasoning"].append(text),
        tool_progress_callback=lambda *args, **kwargs: calls["tool_progress"].append((
            args,
            kwargs,
        )),
        tool_start_callback=lambda call_id, name, args: calls["tool_start"].append((
            call_id,
            name,
            args,
        )),
        tool_complete_callback=lambda call_id, name, args, result: calls[
            "tool_complete"
        ].append((call_id, name, args, result)),
        _emit_interim_assistant_message=None,
        show_commentary=True,
    )
    return agent, calls


def test_projected_message_flush_failure_is_reported_to_gateway():
    agent = SimpleNamespace(
        _session_db=object(),
        session_id="session-1",
        _flush_messages_to_session_db=lambda _messages: False,
    )
    turn = SimpleNamespace(projected_messages=[{"role": "assistant", "content": "answer"}])

    assert _persist_projected_messages(agent, turn, []) is False


def test_projected_message_flush_requires_a_session_database():
    agent = SimpleNamespace(_session_db=None)
    turn = SimpleNamespace(projected_messages=[{"role": "assistant", "content": "answer"}])

    assert _persist_projected_messages(agent, turn, []) is False






def test_stable_ids_match_history_projector():
    """The bridge's stable call ids mirror CodexEventProjector so a live
    TUI tool card correlates with the projected history entry after
    resume."""
    agent, calls = _recording_agent()
    bridge = make_codex_app_server_event_bridge(agent)

    mcp = {
        "type": "mcpToolCall",
        "id": "m1",
        "server": "filesystem",
        "tool": "read",
        "arguments": {"path": "a.py"},
    }
    bridge({"method": "item/started", "params": {"item": mcp}})
    call_id, name, args = calls["tool_start"][0]
    assert call_id == _deterministic_call_id("mcp__filesystem__read", "m1")
    assert name == "mcp.filesystem.read"
    assert args == {"path": "a.py"}

    calls["tool_start"].clear()
    patch = {
        "type": "fileChange",
        "id": "p1",
        "changes": [{"kind": {"type": "add"}, "path": "a.py"}],
    }
    bridge({"method": "item/started", "params": {"item": patch}})
    call_id, name, args = calls["tool_start"][0]
    assert call_id == _deterministic_call_id("apply_patch", "p1")
    assert name == "apply_patch"
    assert args == {"changes": [{"kind": "add", "path": "a.py"}]}


def test_failed_command_result_and_error_flag_are_preserved():
    agent, calls = _recording_agent()
    bridge = make_codex_app_server_event_bridge(agent)
    item = {
        "type": "commandExecution",
        "id": "failed",
        "command": "false",
        "aggregatedOutput": "boom",
        "exitCode": 2,
    }

    bridge({"method": "item/completed", "params": {"item": item}})

    result, is_error = _codex_item_completion_payload(item)
    assert result == "[exit 2]\nboom"
    assert is_error is True
    assert calls["tool_progress"][0][1]["is_error"] is True
    assert calls["tool_complete"][0][3] == "[exit 2]\nboom"




