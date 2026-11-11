"""Aggregate review-budget exhaustion must produce an honest, bounded status.

The fork's last skill_view result has already executed when begin_iteration
checks the aggregate budget. No extra provider request or mutation is needed
to tell the user the review did not reach final synthesis.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent import background_review as bg


def _run_with_result(reason, actions, *, notification_mode="on"):
    messages, notices, results, failures = [], [], [], []
    parent = SimpleNamespace(
        provider="custom", client=None, memory_notifications=notification_mode,
        background_review_callback=notices.append,
        _safe_print=messages.append,
        _emit_auxiliary_failure=failures.append,
    )

    def fake_review(_agent, _snapshot, _prompt, _cfg, _review_run, state, *_rest):
        state.review_result = {"turn_exit_reason": reason, "failed": reason != "text_response"}
        state.review_input_budget = 250_000
        state.review_usage = {"api_calls": 7, "input_tokens": 305_700, "output_tokens": 444}
        state.review_messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "read-1", "function": {"name": "skill_view", "arguments": '{"name":"example"}'}}]},
            {"role": "tool", "tool_call_id": "read-1", "content": '{"success":true,"skill":"example"}'},
        ]

    with (
        patch.object(bg, "_run_review_fork", side_effect=fake_review),
        patch.object(bg, "_parent_can_emit_tool_calls", return_value=True),
        patch.object(bg, "_set_thread_approval_callback"),
        patch.object(bg, "summarize_background_review_actions", return_value=actions) as summarize,
        patch.object(bg, "_log_review_completion", side_effect=lambda _usage, outcome: results.append(outcome)),
    ):
        bg._run_review_in_thread(
            parent, [{"role": "user", "content": "prior snapshot"}],
            "Review existing skills", review_memory=False, explicit=True,
        )
    assert failures == []
    summarize.assert_called_once()
    return messages, notices, results


def test_read_only_budget_stop_notifies_without_claiming_skill_updates():
    displayed, notices, outcomes = _run_with_result("review_input_budget_exhausted", [])
    assert outcomes == ["input_budget_exhausted"]
    assert len(notices) == len(displayed) == 1
    text = notices[0]
    assert "Review incomplete" in text
    assert "305,700/250,000" in text
    assert "No skill or memory updates were confirmed" in text
    assert "no final findings" in text


def test_budget_stop_reports_only_confirmed_changes_then_incomplete_status():
    _displayed, notices, outcomes = _run_with_result(
        "review_input_budget_exhausted", ["Skill 'example' patched"]
    )
    assert outcomes == ["input_budget_exhausted"]
    assert len(notices) == 1
    text = notices[0]
    assert "Skill 'example' patched" in text
    assert "Only the confirmed updates listed above were applied" in text
    assert "Review incomplete" in text


def test_normal_completed_review_keeps_existing_notification_behavior():
    _displayed, notices, outcomes = _run_with_result("text_response", [])
    assert outcomes == ["none"]
    assert notices == []
    _displayed, notices, outcomes = _run_with_result("text_response", ["Skill 'example' patched"])
    assert outcomes == ["skill"]
    assert len(notices) == 1
    assert "Review incomplete" not in notices[0]


def test_budget_stop_notice_even_when_memory_change_notifications_are_off():
    _displayed, notices, outcomes = _run_with_result(
        "review_input_budget_exhausted", [], notification_mode="off"
    )
    assert outcomes == ["input_budget_exhausted"]
    assert len(notices) == 1
    assert "Review incomplete" in notices[0]
