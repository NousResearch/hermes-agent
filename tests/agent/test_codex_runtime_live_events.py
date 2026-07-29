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


def test_agent_message_and_reasoning_deltas_are_forwarded_live():
    agent, calls = _recording_agent()
    bridge = make_codex_app_server_event_bridge(agent)

    bridge({"method": "item/agentMessage/delta", "params": {"delta": "Working"}})
    bridge({"method": "item/reasoning/delta", "params": {"delta": "Thinking"}})
    bridge({"method": "item/reasoning/summaryDelta", "params": {"delta": "Summary"}})

    assert calls["stream"] == ["Working"]
    assert calls["reasoning"] == ["Thinking", "Summary"]


def test_command_start_and_complete_fire_both_callback_contracts():
    agent, calls = _recording_agent()
    bridge = make_codex_app_server_event_bridge(agent)
    started = {
        "type": "commandExecution",
        "id": "abc123",
        "command": "echo hi",
        "cwd": "/tmp",
    }
    completed = dict(
        started,
        aggregatedOutput="hi\n",
        exitCode=0,
        durationMs=250,
    )

    bridge({"method": "item/started", "params": {"item": started}})
    bridge({"method": "item/completed", "params": {"item": completed}})

    expected_args = {"command": "echo hi", "cwd": "/tmp"}
    expected_id = "codex_exec_abc123"
    assert calls["tool_start"] == [(expected_id, "exec_command", expected_args)]
    assert calls["tool_complete"] == [
        (expected_id, "exec_command", expected_args, "hi\n")
    ]
    assert calls["tool_progress"][0] == (
        ("tool.started", "exec_command", "echo hi", expected_args),
        {},
    )
    assert calls["tool_progress"][1] == (
        ("tool.completed", "exec_command", None, None),
        {"duration": 0.25, "is_error": False, "result": "hi\n"},
    )


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


def test_non_tool_events_and_malformed_payloads_are_ignored():
    agent, calls = _recording_agent()
    bridge = make_codex_app_server_event_bridge(agent)
    for note in (
        {"method": "item/started", "params": {"item": {"type": "reasoning"}}},
        {"method": "turn/completed", "params": {}},
        {"method": "item/started", "params": []},
        {},
        None,
    ):
        bridge(note)

    assert all(not entries for entries in calls.values())


def test_one_broken_callback_does_not_hide_other_live_events():
    starts = []

    def broken_progress(*_args, **_kwargs):
        raise RuntimeError("display consumer failed")

    agent = SimpleNamespace(
        tool_progress_callback=broken_progress,
        tool_start_callback=lambda call_id, name, args: starts.append(
            (call_id, name, args)
        ),
    )
    bridge = make_codex_app_server_event_bridge(agent)
    item = {"type": "dynamicToolCall", "id": "d1", "tool": "search"}

    bridge({"method": "item/started", "params": {"item": item}})

    assert starts == [("codex_dyn_search_d1", "search", {})]
