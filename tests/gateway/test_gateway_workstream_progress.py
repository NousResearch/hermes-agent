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


def test_workstream_progress_card_includes_main_agent_and_child_subagents():
    card = _format_gateway_workstream_progress(
        activity={
            "budget_used": 3,
            "budget_max": 10,
            "current_tool": "delegate_task",
            "last_activity_desc": "coordinating production readiness",
            "task_id": "parent-task",
            "active_children": [
                {
                    "subagent_id": "sa-reviewer",
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

    assert "## Workstream 1: 🔄 Hermes" in card
    assert "Progress: [███░░░░░░░] 30% | status: running" in card
    assert "TaskFlow: telegram-session:4" in card
    assert "Updated:" in card
    assert "tasks: 2 | active: 2 | failures: 0" in card
    assert "Subagents / workers:" in card
    assert "Job title: Main agent / orchestrator" in card
    assert "- delegate_task" in card
    assert "- coordinating production readiness" in card
    assert "sa-reviewer" in card
    assert "Job title: Subagent worker (qwen/qwen3-coder:free)" in card
    assert "Time spent: 12s since update | runtime: subagent" in card
    assert "IDs: agent=sa-reviewer | taskId=child-task" in card


def test_workstream_progress_card_handles_missing_activity_defensively():
    card = _format_gateway_workstream_progress(
        activity=None,
        elapsed_seconds=0,
        session_id=None,
        run_generation=None,
    )

    assert "TaskFlow: gateway-run" in card
    assert "Progress: [█░░░░░░░░░] 10% | status: running" in card
    assert "Latest update: working" in card
