"""Regression tests for adaptive gateway workstream progress cards."""

from typing import Any

from gateway.run import (
    _activity_percent,
    _format_duration,
    _format_gateway_workstream_progress,
    _progress_bar,
)


def test_progress_bar_clamps_percentages():
    assert _progress_bar(-20, width=5) == "░░░░░"
    assert _progress_bar(250, width=5) == "█████"
    invalid_percent: Any = "bad"
    assert _progress_bar(invalid_percent, width=4) == "░░░░"


def test_activity_percent_uses_budget_and_caps_running_work():
    assert _activity_percent({"budget_used": 0, "budget_max": 50}) == 5
    assert _activity_percent({"budget_used": 25, "budget_max": 50}) == 50
    assert _activity_percent({"budget_used": 99, "budget_max": 50}) == 95
    assert _activity_percent({}) == 10


def test_format_duration_uses_compact_human_units():
    assert _format_duration(7) == "7s"
    assert _format_duration(67) == "1m 7s"
    assert _format_duration(3667) == "1h 1m 7s"
    assert _format_duration(None) == "0s"


def test_workstream_progress_card_uses_clean_user_facing_layout_without_raw_ids():
    card = _format_gateway_workstream_progress(
        activity={
            "budget_used": 3,
            "budget_max": 10,
            "current_tool": "delegate_task",
            "last_activity_desc": "coordinating production readiness",
            "task_id": "parent-task",
            "active_children": [
                {
                    "subagent_id": "20260708_043811_d27c98:30",
                    "model": "qwen/qwen3-coder:free",
                    "budget_used": 1,
                    "budget_max": 5,
                    "current_tool": "terminal",
                    "task_id": "child-task",
                    "last_activity_desc": "running targeted tests",
                    "seconds_since_activity": 12,
                }
            ],
        },
        elapsed_seconds=125,
        session_id="telegram-session",
        run_generation=4,
    )

    assert "## 🔄 Hermes is working" in card
    assert "[███░░░░░░░] 30% · running · 2m 5s elapsed" in card
    assert "Work summary" in card
    assert "• Current focus: delegate_task" in card
    assert "• Latest update: coordinating production readiness" in card
    assert "• Team: 1 orchestrator + 1 worker · 0 failures" in card
    assert "Workers" in card
    assert "1. 🔄 Orchestrator" in card
    assert "Role: Main agent" in card
    assert "2. 🔄 Worker 1" in card
    assert "Role: Subagent · model: qwen/qwen3-coder:free" in card
    assert "Updated: 12s ago" in card
    assert "Doing: running targeted tests" in card
    assert "Reference: telegram-session · run 4" in card
    assert "TaskFlow:" not in card
    assert "IDs:" not in card
    assert "taskId=" not in card
    assert "parent-task" not in card
    assert "child-task" not in card
    assert "20260708_043811_d27c98:30" not in card


def test_workstream_progress_card_handles_missing_activity_defensively():
    card = _format_gateway_workstream_progress(
        activity=None,
        elapsed_seconds=0,
        session_id=None,
        run_generation=None,
    )

    assert "## 🔄 Hermes is working" in card
    assert "[█░░░░░░░░░] 10% · running · 0s elapsed" in card
    assert "• Current focus: working" in card
    assert "Reference:" not in card
    assert "TaskFlow:" not in card
    assert "IDs:" not in card
