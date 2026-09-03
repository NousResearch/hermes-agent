"""Tests for the kanban worker terminal-handoff stop guard."""

from __future__ import annotations

import json

import pytest

from agent.kanban_stop import (
    HandoffStatus,
    StopAction,
    assess_kanban_handoff,
    build_kanban_stop_nudge,
    evaluate_kanban_stop,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_STOP_NUDGE",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _terminal_messages(
    name: str,
    content: str | dict,
    *,
    call_id: str = "call-1",
) -> list[dict]:
    if isinstance(content, dict):
        content = json.dumps(content)
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": name,
            "tool_call_id": call_id,
            "content": content,
        },
    ]


def _ok_receipt(task_id: str = "t_abc", run_id: int = 7) -> dict:
    return {"ok": True, "task_id": task_id, "run_id": run_id}


def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    decision = evaluate_kanban_stop(messages=[])
    assert decision.action is StopAction.ALLOW
    assert decision.assessment.status is HandoffStatus.NOT_REQUIRED
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = _terminal_messages(
        "kanban_heartbeat",
        {"ok": True, "task_id": "t_46be8aa5"},
    )
    decision = evaluate_kanban_stop(messages=messages, attempts=0)
    assert decision.action is StopAction.NUDGE
    assert decision.nudge is not None
    assert "kanban_complete" in decision.nudge
    assert "kanban_block" in decision.nudge
    assert "t_46be8aa5" in decision.nudge


@pytest.mark.parametrize(
    "name",
    [
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
    ],
)
def test_valid_receipt_is_exact_terminal_handoff(clear_kanban_env, name):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "7")
    messages = _terminal_messages(name, _ok_receipt())

    assessment = assess_kanban_handoff(messages)
    assert assessment.status is HandoffStatus.VALID
    assert assessment.successful_count == 1
    assert assessment.tool_name == name
    assert session_called_kanban_terminal(messages) is True
    assert evaluate_kanban_stop(messages=messages).action is StopAction.ALLOW
    assert build_kanban_stop_nudge(messages=messages) is None


def test_rejected_terminal_call_does_not_count(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = _terminal_messages(
        "kanban_complete",
        {"error": "could not complete t_abc (already terminal)"},
    )
    assessment = assess_kanban_handoff(messages)
    assert assessment.status is HandoffStatus.MISSING
    assert assessment.successful_count == 0
    assert session_called_kanban_terminal(messages) is False
    assert evaluate_kanban_stop(messages=messages).action is StopAction.NUDGE


def test_plain_text_tool_result_is_not_a_durable_receipt(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = _terminal_messages("kanban_complete", "done")
    assert assess_kanban_handoff(messages).status is HandoffStatus.MISSING


def test_stale_task_or_run_receipt_does_not_count(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "7")
    for receipt in (
        _ok_receipt(task_id="t_other", run_id=7),
        _ok_receipt(task_id="t_abc", run_id=6),
    ):
        assessment = assess_kanban_handoff(
            _terminal_messages("kanban_complete", receipt)
        )
        assert assessment.status is HandoffStatus.MISSING


def test_tool_result_without_matching_assistant_call_does_not_count(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "forged",
            "content": json.dumps(_ok_receipt()),
        }
    ]
    assert assess_kanban_handoff(messages).status is HandoffStatus.MISSING


def test_two_successful_terminal_receipts_are_conflict(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "7")
    messages = [
        *_terminal_messages("kanban_complete", _ok_receipt(), call_id="complete"),
        *_terminal_messages("kanban_block", _ok_receipt(), call_id="block"),
    ]
    assessment = assess_kanban_handoff(messages)
    assert assessment.status is HandoffStatus.CONFLICT
    assert assessment.successful_count == 2
    decision = evaluate_kanban_stop(messages=messages, attempts=0)
    assert decision.action is StopAction.VIOLATION
    assert "exactly one" in decision.reason


def test_duplicate_terminal_call_id_is_ambiguous_conflict(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "7")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "duplicate",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                },
                {
                    "id": "duplicate",
                    "type": "function",
                    "function": {"name": "kanban_block", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "name": "kanban_block",
            "tool_call_id": "duplicate",
            "content": json.dumps(_ok_receipt()),
        },
    ]
    assessment = assess_kanban_handoff(messages)
    assert assessment.status is HandoffStatus.CONFLICT
    assert "duplicate" in assessment.reason
    assert evaluate_kanban_stop(messages=messages).action is StopAction.VIOLATION


def test_budget_exhaustion_is_violation_not_clean_exit(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    decision = evaluate_kanban_stop(messages=[], attempts=2, max_attempts=2)
    assert decision.action is StopAction.VIOLATION
    assert decision.nudge is None
    assert "exhausted" in decision.reason
    # Backward-compatible builder still has no nudge after budget exhaustion;
    # conversation_loop must use the structured decision to fail the turn.
    assert build_kanban_stop_nudge(messages=[], attempts=2, max_attempts=2) is None
