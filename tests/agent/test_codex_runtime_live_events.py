"""Regression tests for live Codex app-server events.

The history projector is completion-only. These tests protect the parallel
display bridge that makes deltas and tool cards visible before resume.
"""

from types import SimpleNamespace

from agent.codex_runtime import (
    _codex_live_event,
    _codex_tool_descriptor,
    _codex_tool_result,
)
from agent.transports.codex_event_projector import _deterministic_call_id


def _recording_agent():
    calls = {
        "stream": [],
        "interim": [],
        "reasoning": [],
        "tool_progress": [],
        "tool_start": [],
        "tool_complete": [],
    }
    agent = SimpleNamespace(
        _fire_stream_delta=lambda text: calls["stream"].append(text),
        stream_delta_callback=calls["stream"].append,
        _stream_callback=None,
        interim_assistant_callback=lambda text, *, already_streamed=False: calls[
            "interim"
        ].append((text, already_streamed)),
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
    )
    return agent, calls


def test_agent_message_and_reasoning_deltas_are_forwarded_live():
    agent, calls = _recording_agent()

    _codex_live_event(
        agent,
        {"method": "item/agentMessage/delta", "params": {"delta": "Working"}},
    )
    _codex_live_event(
        agent,
        {"method": "item/reasoning/delta", "params": {"delta": "Thinking"}},
    )
    _codex_live_event(
        agent,
        {"method": "item/reasoning/summaryDelta", "params": {"delta": "Summary"}},
    )

    assert calls["stream"] == ["Working"]
    assert calls["interim"] == []
    assert calls["reasoning"] == ["Thinking", "Summary"]


def test_agent_message_delta_uses_interim_callback_without_streaming():
    agent, calls = _recording_agent()
    agent.stream_delta_callback = None
    agent._stream_callback = None

    _codex_live_event(
        agent,
        {"method": "item/agentMessage/delta", "params": {"delta": "Working"}},
    )

    assert calls["stream"] == []
    assert calls["interim"] == [("Working", False)]


def test_command_start_and_complete_fire_both_callback_contracts():
    agent, calls = _recording_agent()
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

    _codex_live_event(agent, {"method": "item/started", "params": {"item": started}})
    _codex_live_event(
        agent, {"method": "item/completed", "params": {"item": completed}}
    )

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


def test_descriptor_ids_and_args_match_history_projector():
    mcp = {
        "type": "mcpToolCall",
        "id": "m1",
        "server": "filesystem",
        "tool": "read",
        "arguments": {"path": "a.py"},
    }
    call_id, name, args = _codex_tool_descriptor(mcp)
    assert call_id == _deterministic_call_id("mcp__filesystem__read", "m1")
    assert name == "mcp.filesystem.read"
    assert args == {"path": "a.py"}

    patch = {
        "type": "fileChange",
        "id": "p1",
        "changes": [{"kind": {"type": "add"}, "path": "a.py"}],
    }
    call_id, name, args = _codex_tool_descriptor(patch)
    assert call_id == _deterministic_call_id("apply_patch", "p1")
    assert name == "apply_patch"
    assert args == {"changes": [{"kind": "add", "path": "a.py"}]}


def test_failed_command_result_and_error_flag_are_preserved():
    agent, calls = _recording_agent()
    item = {
        "type": "commandExecution",
        "id": "failed",
        "command": "false",
        "aggregatedOutput": "boom",
        "exitCode": 2,
    }

    _codex_live_event(agent, {"method": "item/completed", "params": {"item": item}})

    assert _codex_tool_result(item) == "[exit 2]\nboom"
    assert calls["tool_progress"][0][1]["is_error"] is True
    assert calls["tool_complete"][0][3] == "[exit 2]\nboom"


def test_non_tool_events_and_malformed_payloads_are_ignored():
    agent, calls = _recording_agent()
    for note in (
        {"method": "item/started", "params": {"item": {"type": "reasoning"}}},
        {"method": "item/completed", "params": {"item": {"type": "agentMessage"}}},
        {"method": "turn/completed", "params": {}},
        {"method": "item/started", "params": []},
        {},
        None,
    ):
        _codex_live_event(agent, note)

    assert all(not entries for entries in calls.values())


def test_one_broken_callback_does_not_hide_other_live_events():
    starts = []

    def broken_progress(*_args, **_kwargs):
        raise RuntimeError("display consumer failed")

    agent = SimpleNamespace(
        tool_progress_callback=broken_progress,
        tool_start_callback=lambda *args: starts.append(args),
    )
    item = {"type": "dynamicToolCall", "id": "d1", "tool": "search"}

    _codex_live_event(agent, {"method": "item/started", "params": {"item": item}})

    assert starts == [("codex_dyn_search_d1", "search", {})]
