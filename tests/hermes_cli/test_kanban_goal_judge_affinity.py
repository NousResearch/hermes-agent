"""Goal-judge handoff gates: per-task relay-affinity scope + fail-open on transport failure.

Both terminal-handoff gates (``hermes_cli.kanban`` for the CLI, ``tools.kanban_tools`` for
worker tool calls) run outside any agent turn, so without a bound scope the relay rejects the
judge call (400 MissingSessionID, #113669). ``judge_goal`` then fails open to ``continue`` with
``transport_failed=True``; a gate that reads that as a human "not done" rejects the handoff
(#83610). These tests pin the scope binding and that only a genuine verdict rejects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
from hermes_cli import kanban as kanban_cli
from tools import kanban_tools

_TRANSPORT_FAILED = ("continue", "judge error: BadRequestError", False, None, True)
_GENUINE_CONTINUE = ("continue", "goal not met yet", False, None, False)


def _task(tid="task-1"):
    return SimpleNamespace(id=tid, title="goal title", body="goal body", goal_mode=True)


def _aux_client():
    return patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(object(), "test-model"),
    )


def _recording_judge(seen):
    def fake_judge(**kwargs):
        seen.append(get_affinity_scope())
        return ("done", "ok", False, None, False)
    return fake_judge


def test_cli_gate_transport_failure_fails_open_but_genuine_verdict_rejects():
    with _aux_client(), patch("hermes_cli.goals.judge_goal", return_value=_TRANSPORT_FAILED):
        assert kanban_cli._goal_mode_handoff_rejection(_task(), "evidence") == ("done", None)
    with _aux_client(), patch("hermes_cli.goals.judge_goal", return_value=_GENUINE_CONTINUE):
        assert kanban_cli._goal_mode_handoff_rejection(_task(), "evidence") == (
            "continue", "goal not met yet")


def test_tool_gate_transport_failure_fails_open_but_genuine_verdict_rejects():
    with patch.object(kanban_tools, "_goal_judge_available", return_value=True):
        with patch.object(kanban_tools, "judge_goal", return_value=_TRANSPORT_FAILED):
            kanban_tools._goal_gate("kanban_complete", _task(), "task-1", "ev")  # no raise
        with patch.object(kanban_tools, "judge_goal", return_value=_GENUINE_CONTINUE):
            with pytest.raises(kanban_tools._Reject):
                kanban_tools._goal_gate("kanban_complete", _task(), "task-1", "ev")


def test_cli_gate_binds_per_task_affinity_scope():
    """Headless judge call runs under kanban:<task_id>; a bound scope is kept."""
    seen = []
    with _aux_client(), patch("hermes_cli.goals.judge_goal", side_effect=_recording_judge(seen)):
        assert kanban_cli._goal_mode_handoff_rejection(_task("task-9"), "ev") == ("done", None)
    assert seen == ["kanban:task-9"]
    assert get_affinity_scope() is None

    token = set_affinity_scope("outer-conversation")
    try:
        with _aux_client(), patch("hermes_cli.goals.judge_goal", side_effect=_recording_judge(seen)):
            kanban_cli._goal_mode_handoff_rejection(_task("task-9"), "ev")
    finally:
        reset_affinity_scope(token)
    assert seen[-1] == "outer-conversation"


def test_tool_gate_binds_per_task_affinity_scope():
    seen = []
    with patch.object(kanban_tools, "_goal_judge_available", return_value=True), patch.object(
            kanban_tools, "judge_goal", side_effect=_recording_judge(seen)):
        kanban_tools._goal_gate("kanban_complete", _task("task-9"), "task-9", "ev")
    assert seen == ["kanban:task-9"]
    assert get_affinity_scope() is None


def test_goal_loop_judge_binds_per_task_affinity_scope():
    """The between-turns judge in run_kanban_goal_loop runs under kanban:<task_id> too."""
    from hermes_cli import goals

    seen = []

    def fake_judge(goal, last_response):
        seen.append(get_affinity_scope())
        return ("continue", "not yet", False, None, False)

    with patch.object(goals, "judge_goal", side_effect=fake_judge):
        result = goals.run_kanban_goal_loop(
            task_id="task-7",
            goal_text="goal",
            run_turn=lambda prompt: "still working",
            task_status_fn=lambda: "running",
            block_fn=lambda msg: None,
            max_turns=2,
            first_response="first",
        )
    assert result["outcome"] == "blocked_budget"
    assert seen == ["kanban:task-7", "kanban:task-7"]
    assert get_affinity_scope() is None
