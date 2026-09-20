"""Observed Antigravity stream-json → Hermes projection contracts."""

from __future__ import annotations

import json

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.transports.antigravity_event_projector import (
    AntigravityEventProjector,
    make_antigravity_event_bridge,
)


def test_assistant_delta_is_projected_with_wire_correlation() -> None:
    projector = AntigravityEventProjector()

    projected = projector.project({
        "type": "step_update",
        "sequence": 4,
        "step_index": 1,
        "delta": {"type": "assistant", "text": "hello"},
    })

    assert projected.messages == []  # deltas are display-only; transcript stays alternation-safe
    assert projected.assistant_delta == "hello"
    assert projected.sequence == 4
    assert projected.step_index == 1
    assert projected.final_text is None


def test_bridge_forwards_assistant_delta_to_canonical_stream_callback() -> None:
    agent = SimpleNamespace(
        _fire_stream_delta=MagicMock(),
        _fire_reasoning_delta=MagicMock(),
        tool_progress_callback=MagicMock(),
        tool_start_callback=MagicMock(),
        tool_complete_callback=MagicMock(),
        _emit_interim_assistant_message=MagicMock(),
    )

    make_antigravity_event_bridge(agent)({
        "type": "step_update", "sequence": 4, "step_index": 1,
        "delta": {"type": "assistant", "text": "hello"},
    })

    agent._fire_stream_delta.assert_called_once_with("hello")


def test_tool_active_then_done_preserves_observed_command_metadata() -> None:
    projector = AntigravityEventProjector()
    active = projector.project({
        "type": "step_update", "sequence": 7, "step_index": 2,
        "step": {"type": "tool", "status": "ACTIVE", "id": "shell-1", "name": "terminal",
                 "command": "pwd", "cwd": "/repo"},
    })
    done = projector.project({
        "type": "step_update", "sequence": 8, "step_index": 2,
        "step": {"type": "tool", "status": "DONE", "id": "shell-1", "name": "terminal", "output": "/repo\\n"},
    })

    assert active.messages == []
    assert done.is_tool_iteration is True
    assistant, tool = done.messages
    assert assistant["sequence"] == 8 and assistant["step_index"] == 2
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"command": "pwd", "cwd": "/repo"}
    assert tool == {"role": "tool", "tool_call_id": "antigravity_terminal_shell-1", "content": "/repo\\n",
                    "sequence": 8, "step_index": 2}


def test_result_is_the_only_assistant_history_message_after_deltas() -> None:
    projector = AntigravityEventProjector()
    projector.project({"type": "step_update", "sequence": 1, "step_index": 0,
                       "delta": {"type": "assistant", "text": "he"}})
    result = projector.project({"type": "result", "sequence": 2, "step_index": 0,
                                "result": {"text": "hello"}})

    assert result.final_text == "hello"
    assert result.messages == [{"role": "assistant", "content": "hello", "sequence": 2, "step_index": 0}]


def test_tool_pair_without_explicit_id_correlates_by_observed_step_index() -> None:
    projector = AntigravityEventProjector()
    started = projector.feed({"event": "step_update", "step_update": {
        "step_index": 7, "state": "ACTIVE", "step_type": "tool", "tool_name": "view_file",
        "tool_info": {"parameters": {"AbsolutePath": "/tmp/a"}},
    }})
    completed = projector.feed({"event": "step_update", "step_update": {
        "step_index": 7, "state": "DONE", "step_type": "tool", "tool_name": "view_file",
        "tool_info": {"result": "ok"},
    }})

    assert started.tool_event is not None
    assert completed.tool_event is not None
    assert started.tool_event["id"] == completed.tool_event["id"]
    assert completed.messages[0]["tool_calls"][0]["function"]["arguments"] != "{}"


def test_untrusted_text_containing_permission_does_not_abort_turn() -> None:
    projected = AntigravityEventProjector().project({
        "event": "future_event",
        "message": "You do not have permission; approval is mentioned in model text",
    })

    assert projected.error is None
    assert projected.messages == []


def test_init_conversation_id_is_persisted_on_terminal_message_fallback() -> None:
    projector = AntigravityEventProjector()
    projector.feed({"event": "init", "conversation_id": "conv-init"})
    projected = projector.feed({"event": "result", "result": {"status": "SUCCESS", "response": "done"}})

    assert projected.messages[0]["_antigravity"] == {"conversation_id": "conv-init"}


def test_unknown_permission_is_fail_closed_without_assistant_text() -> None:
    projected = AntigravityEventProjector().project({
        "type": "step_update", "sequence": 9, "step_index": 3,
        "step": {"type": "permission", "action": "approval_required"},
    })

    assert projected.messages == []
    assert projected.error == "unsupported Antigravity approval/permission event; refusing by default"
