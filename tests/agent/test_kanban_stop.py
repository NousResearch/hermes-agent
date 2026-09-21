"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    reconcile_kanban_stop_to_review,
    session_called_kanban_terminal,
    successful_kanban_terminal_transition,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_KANBAN_AUTO_REVIEW_ON_STOP",
    ):
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
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


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


def test_rejected_kanban_request_review_is_nudged(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_review")
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
            "content": "moved to review",
        },
    ]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


def test_successful_kanban_request_changes_stops_nudge(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_review")
    messages = [
        {
            "role": "tool",
            "name": "kanban_request_changes",
            "tool_call_id": "1",
            "content": '{"ok": true, "status": "changes_requested"}',
        }
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_rejected_kanban_request_changes_is_nudged(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_review")
    messages = [{
        "role": "tool", "name": "kanban_request_changes",
        "tool_call_id": "1", "content": '{"ok": false, "error": "not reviewer"}',
    }]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


def test_successful_terminal_transition_matches_current_durable_result(
    clear_kanban_env,
):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    tool_calls = [
        {
            "id": "complete-current",
            "function": {"name": "kanban_complete", "arguments": "{}"},
        }
    ]
    messages = [
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "complete-old",
            "content": '{"ok": true}',
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "complete-current",
            "content": '{"ok": true, "run_id": 715}',
        },
    ]

    assert successful_kanban_terminal_transition(
        messages=messages, tool_calls=tool_calls
    ) is True


def test_failed_or_non_worker_terminal_call_does_not_stop_execution(
    clear_kanban_env,
):
    tool_calls = [
        {
            "id": "complete-current",
            "function": {"name": "kanban_complete", "arguments": "{}"},
        }
    ]
    messages = [
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "complete-current",
            "content": '{"error": "completion rejected"}',
        }
    ]

    assert successful_kanban_terminal_transition(
        messages=messages, tool_calls=tool_calls
    ) is False
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert successful_kanban_terminal_transition(
        messages=messages, tool_calls=tool_calls
    ) is False


def test_exhausted_nudges_handoff_useful_output_to_review(
    clear_kanban_env, monkeypatch
):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_review_fallback")
    clear_kanban_env.setenv("HERMES_KANBAN_AUTO_REVIEW_ON_STOP", "1")
    monkeypatch.setattr(
        "agent.kanban_stop._configured_review_profile",
        lambda: "review-verification-steward",
    )
    calls = []

    def fake_request_review(args, **_kwargs):
        calls.append(args)
        return '{"ok": true, "status": "review"}'

    monkeypatch.setattr(
        "tools.kanban_tools._handle_request_review", fake_request_review
    )

    handed_off = reconcile_kanban_stop_to_review(
        messages=[{"role": "user", "content": "work the task"}],
        final_response="Implemented the bounded repair and ran 12 focused tests.",
        attempts=2,
    )

    assert handed_off is True
    assert calls == [
        {
            "summary": (
                "Automatic terminal handoff after 2 unanswered Kanban stop "
                "nudges. Worker final output:\n\nImplemented the bounded repair "
                "and ran 12 focused tests."
            ),
            "reviewer": "review-verification-steward",
            "metadata": {
                "source": "kanban_stop_guard",
                "terminal_nudges": 2,
                "completion_inferred": False,
            },
        }
    ]


def test_exhausted_nudges_retry_original_worker_by_default(
    clear_kanban_env, monkeypatch
):
    """Unverified prose must not consume a reviewer slot without opt-in."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_retry_default")
    monkeypatch.setattr(
        "agent.kanban_stop._configured_review_profile",
        lambda: "review-verification-steward",
    )
    calls = []
    monkeypatch.setattr(
        "tools.kanban_tools._handle_request_review",
        lambda args: calls.append(args) or '{"ok": true}',
    )

    assert reconcile_kanban_stop_to_review(
        messages=[],
        final_response="I inspected the task but did not record a terminal receipt.",
        attempts=2,
    ) is False
    assert calls == []


def test_review_handoff_is_bounded_and_never_infers_completion(
    clear_kanban_env, monkeypatch
):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_review_fallback")
    monkeypatch.setattr(
        "agent.kanban_stop._configured_review_profile",
        lambda: "review-verification-steward",
    )
    calls = []
    monkeypatch.setattr(
        "tools.kanban_tools._handle_request_review",
        lambda args: calls.append(args) or '{"ok": true}',
    )

    assert reconcile_kanban_stop_to_review(
        messages=[], final_response="useful evidence", attempts=1
    ) is False
    assert reconcile_kanban_stop_to_review(
        messages=[], final_response="   ", attempts=2
    ) is False
    assert calls == []


def test_review_handoff_failure_leaves_dispatcher_ownership(
    clear_kanban_env, monkeypatch
):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_review_fallback")
    monkeypatch.setattr(
        "agent.kanban_stop._configured_review_profile",
        lambda: "review-verification-steward",
    )
    monkeypatch.setattr(
        "tools.kanban_tools._handle_request_review",
        lambda _args: '{"ok": false, "error": "goal evidence incomplete"}',
    )

    assert reconcile_kanban_stop_to_review(
        messages=[], final_response="partial report", attempts=2
    ) is False






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
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": '{"ok": true}'},
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
