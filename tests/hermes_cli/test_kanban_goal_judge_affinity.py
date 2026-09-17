"""Goal-judge handoff gates bind the per-task relay-affinity scope.

Both terminal-handoff gates (``hermes_cli.kanban`` for the CLI dispatcher,
``tools.kanban_tools`` for worker tool calls) run outside any agent turn, so
without a bound scope the relay rejects the judge call (400 MissingSessionID)
(#113669). Fail-open on transport failure belongs to the shared gate fix
(#73119); these tests pin only the scope binding plus unchanged genuine
verdicts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.portal_tags import get_affinity_scope, set_affinity_scope
from hermes_cli import kanban as kanban_cli
from tools import kanban_tools


def _task(tid="task-1"):
    return SimpleNamespace(id=tid, title="goal title", body="goal body", goal_mode=True)


def _aux_client():
    return patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(object(), "test-model"),
    )


def test_cli_gate_genuine_continue_still_rejects():
    """A real not-done verdict keeps rejecting; only transport failure fails open."""
    with _aux_client(), patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "goal not met yet", False, None, False),
    ):
        assert kanban_cli._goal_mode_handoff_rejection(_task(), "evidence") == (
            "continue", "goal not met yet")


def test_cli_gate_binds_per_task_affinity_scope():
    """Headless judge call runs under kanban:<task_id>; a bound scope is kept."""
    seen = []

    def fake_judge(**kwargs):
        seen.append(get_affinity_scope())
        return ("done", "ok", False, None, False)

    with _aux_client(), patch("hermes_cli.goals.judge_goal", side_effect=fake_judge):
        assert kanban_cli._goal_mode_handoff_rejection(_task("task-9"), "ev") == ("done", None)
    assert seen == ["kanban:task-9"]
    assert get_affinity_scope() is None

    token = set_affinity_scope("outer-conversation")
    try:
        with _aux_client(), patch("hermes_cli.goals.judge_goal", side_effect=fake_judge):
            kanban_cli._goal_mode_handoff_rejection(_task("task-9"), "ev")
    finally:
        from agent.portal_tags import reset_affinity_scope
        reset_affinity_scope(token)
    assert seen[-1] == "outer-conversation"


def test_tool_gate_genuine_blocked_still_rejects():
    """A real unachievable verdict still raises; only transport failure fails open."""
    with patch.object(kanban_tools, "_goal_judge_available", return_value=True), patch.object(
        kanban_tools, "judge_goal",
        return_value=("blocked", "unachievable", False, None, False),
    ):
        with pytest.raises(kanban_tools._Reject):
            kanban_tools._goal_gate("kanban_complete", _task(), "task-1", "ev")
