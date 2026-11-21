"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_disabled_inside_delegated_child(clear_kanban_env):
    from agent.delegation_context import delegated_child_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with delegated_child_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_disabled_inside_non_dispatcher_context(clear_kanban_env):
    from agent.delegation_context import non_dispatcher_owned_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.


@pytest.mark.parametrize(
    "tool_name,who",
    [
        ("kanban_request_review", "build worker handing off for same-card review"),
        ("kanban_request_changes", "review agent sending the card back"),
        ("kanban_schedule", "worker parking the card on a timed wait"),
    ],
)
def test_no_nudge_after_handoff_tool(clear_kanban_env, tool_name, who):
    """Handoff tools end the worker's turn just like complete/block.

    Both move the card out of ``running``, and the worker is told to call
    them — goals.py's continuation/finalize prompts name
    ``kanban_request_review``; the force-loaded sdlc-review skill names
    ``kanban_request_changes``. Nudging afterwards asks a worker that did
    the right thing to close a card it must not close.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_handoff")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is True, who
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_still_fires_for_non_terminal_kanban_tool(clear_kanban_env):
    """Widening the set must not swallow the case the guard exists for."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "Let me open the review next.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_comment", "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is False
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    # The nudge offers every worker exit, not just close-out; a card that must go
    # through review must never be steered to ``kanban_complete`` alone.
    assert "kanban_request_review" in nudge and "kanban_block" in nudge


def test_nudge_after_rejected_terminal_handoff(clear_kanban_env):
    """Regression (#136843): a rejected handoff must not end worker ownership.

    The handler returned a structured tool error and the board state is
    unchanged, but the transcript contains the tool call and result row, so the
    name-only scan read the run as already handed off and the guard went silent
    for the rest of the session — a narrated stop then exits rc=0 as a
    protocol_violation with no nudge at the moment it is most needed.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_rejected")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "1",
            "content": '{"error": "kanban_complete blocked: the following created_cards '
                       'do not exist or were not created by this worker: [card_x]. '
                       'Your task is still in-flight (no state change)."}',
        },
        {"role": "assistant", "content": "Let me gather more evidence instead.", "tool_calls": []},
    ]
    assert session_called_kanban_terminal(messages) is False
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_block" in nudge


def test_nudge_after_timeout_failure_string(clear_kanban_env):
    """A failed terminal call can also surface as the executor's plain failure
    string (no JSON body), e.g. a handler stalling to the tool timeout."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_timeout")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_request_changes", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_request_changes",
            "tool_call_id": "1",
            "content": "Error executing tool 'kanban_request_changes': timed out after 420.0s",
        },
    ]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


def test_no_nudge_after_successful_handoff_result(clear_kanban_env):
    """The success shape the fix must keep treating as terminal: a structured
    ok payload on the result row (no blank-body assumptions)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_ok_payload")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_request_review", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_request_review",
            "tool_call_id": "1",
            "content": '{"ok": true, "task_id": "t_ok_payload", "run_id": 3507, "status": "review"}',
        },
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_rejected_then_retried_successfully_stays_terminal(clear_kanban_env):
    """A call rejected early and retried successfully later in the session must
    still read as terminal: a rejection suppresses only its own call id."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_retry_ok")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "1",
            "content": '{"error": "kanban_complete: summary is required"}',
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "2",
            "content": '{"ok": true, "task_id": "t_retry_ok", "status": "done"}',
        },
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_rejection_after_success_keeps_guard_silent(clear_kanban_env):
    """Order independence: success earlier in the session ends ownership even
    if a later terminal call is rejected (same session, new lifecycle)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_success_first")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "1",
            "content": '{"ok": true, "task_id": "t_success_first", "status": "done"}',
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": "kanban_request_review", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_request_review",
            "tool_call_id": "2",
            "content": '{"error": "kanban_request_review: task not in running/ready"}',
        },
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_rejection_with_appended_harness_notice(clear_kanban_env):
    """The persisted rejection can carry a harness notice AFTER the JSON body
    (``[Tool loop warning: ...]``), so whole-string ``json.loads`` fails; the
    leading-JSON fallback must still read the underlying ``{"error": ...}``."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_loop_suffix")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "1",
            "content": '{"error": "kanban_complete: summary is required"}\n\n'
                       '[Tool loop warning: repeated_exact_failure_warning; count=2; '
                       'kanban_complete has failed 2 times with identical arguments.]',
        },
    ]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None
