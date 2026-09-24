"""Tests for /goal queue: goals lined up behind the active one.

- GoalState roundtrips queued_goals through JSON; old rows without the field load clean.
- GoalManager queue management (add / remove / clear / render) and promotion on terminal verdicts.
- dispatch_goal_command: ``/goal queue <text>`` queues when a goal is active, sets when none is,
  and control subcommands (list / remove / clear) never replace the active goal.
"""

import json
import unittest.mock
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import goals as goals_mod
from hermes_cli.goal_command import dispatch_goal_command
from hermes_cli.goals import GoalManager, GoalState


@pytest.fixture(autouse=True)
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes stay hermetic."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals_mod._DB_CACHE.clear()
    yield home
    goals_mod._DB_CACHE.clear()


def _mgr_with_goal(session_id="queue-test-sid", goal_text="first goal"):
    mgr = GoalManager(session_id=session_id)
    mgr.set(goal_text)
    return mgr


# ──────────────────────────────────────────────────────────────────────
# Serialization / backwards compatibility
# ──────────────────────────────────────────────────────────────────────


def test_queued_goals_roundtrip_through_goalstate_json():
    state = GoalState(goal="ship it", status="active")
    state.queued_goals = ["second goal", "third goal"]
    loaded = GoalState.from_json(state.to_json())
    assert loaded.queued_goals == ["second goal", "third goal"]


def test_old_state_rows_without_queue_load_clean():
    old = {"goal": "legacy", "status": "active"}
    state = GoalState.from_json(json.dumps(old))
    assert state.queued_goals == []


def test_queue_persists_and_reloads():
    mgr = _mgr_with_goal("queue-persist-sid")
    mgr.queue_goal("later goal")
    reloaded = GoalManager(session_id="queue-persist-sid")
    assert reloaded.state.queued_goals == ["later goal"]


# ──────────────────────────────────────────────────────────────────────
# GoalManager queue management
# ──────────────────────────────────────────────────────────────────────


def test_queue_add_remove_clear():
    mgr = _mgr_with_goal("queue-mgmt-sid")
    assert mgr.queue_goal("goal two") == 1
    assert mgr.queue_goal("goal three") == 2
    assert "1. goal two" in mgr.render_queue()

    assert mgr.remove_queued_goal(1) == "goal two"
    assert mgr.state.queued_goals == ["goal three"]

    assert mgr.clear_queued_goals() == 1
    assert mgr.state.queued_goals == []


def test_queue_requires_active_goal():
    mgr = GoalManager(session_id="queue-nogoal-sid")
    with pytest.raises(RuntimeError):
        mgr.queue_goal("nope")


def test_queue_pop_returns_fifo():
    mgr = _mgr_with_goal("queue-pop-sid")
    mgr.queue_goal("first queued")
    mgr.queue_goal("second queued")
    assert mgr.pop_queued_goal() == "first queued"
    assert mgr.pop_queued_goal() == "second queued"
    assert mgr.pop_queued_goal() is None


# ──────────────────────────────────────────────────────────────────────
# Promotion on terminal verdicts
# ──────────────────────────────────────────────────────────────────────


def test_done_verdict_promotes_queued_goal_and_continues():
    mgr = _mgr_with_goal("queue-done-sid")
    mgr.queue_goal("next objective")
    with patch("hermes_cli.goals.judge_goal", return_value=("done", "all verified", False, None, False)):
        decision = mgr.evaluate_after_turn("everything works")
    assert decision["verdict"] == "done"
    assert decision["should_continue"] is True
    assert "next objective" in decision["continuation_prompt"]
    assert mgr.is_active()
    assert mgr.state.goal == "next objective"
    assert mgr.state.turns_used == 0


def test_blocked_verdict_keeps_queue_intact():
    """A blocked goal pauses for user review — the queue must NOT silently replace it."""
    mgr = _mgr_with_goal("queue-blocked-sid")
    mgr.queue_goal("salvage objective")
    with patch("hermes_cli.goals.judge_goal", return_value=("blocked", "impossible as stated", False, None, False)):
        decision = mgr.evaluate_after_turn("gave up")
    assert decision["status"] == "paused"
    assert mgr.state.goal == "first goal"
    assert mgr.state.queued_goals == ["salvage objective"]


def test_no_promotion_when_queue_empty():
    mgr = _mgr_with_goal("queue-empty-sid")
    with patch("hermes_cli.goals.judge_goal", return_value=("done", "done", False, None, False)):
        decision = mgr.evaluate_after_turn("done")
    assert decision["verdict"] == "done"
    assert decision["should_continue"] is False
    assert mgr.state.status == "done"


def test_continue_verdict_never_promotes():
    mgr = _mgr_with_goal("queue-continue-sid")
    mgr.queue_goal("not yet")
    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False, None, False)):
        decision = mgr.evaluate_after_turn("partial progress")
    assert decision["verdict"] == "continue"
    assert mgr.state.goal == "first goal"
    assert mgr.state.queued_goals == ["not yet"]


# ──────────────────────────────────────────────────────────────────────
# dispatch_goal_command surface
# ──────────────────────────────────────────────────────────────────────


def test_dispatch_queue_lists_without_replacing():
    mgr = _mgr_with_goal("dispatch-list-sid")
    mgr.queue_goal("listed goal")
    result = dispatch_goal_command(mgr, "queue", authorize_gate=lambda: None)
    assert "listed goal" in result.output
    assert mgr.state.goal == "first goal"


def test_dispatch_queue_with_active_goal_queues_it():
    mgr = _mgr_with_goal("dispatch-set-sid")
    result = dispatch_goal_command(mgr, "queue the follow-up plan", authorize_gate=lambda: None)
    assert "Queued" in result.output
    assert mgr.state.goal == "first goal"
    assert mgr.state.queued_goals == ["the follow-up plan"]
    # Queuing must NOT kick a new turn.
    assert result.kickoff is False
    assert result.prompt is None


def test_dispatch_queue_without_goal_sets_directly():
    mgr = GoalManager(session_id="dispatch-nogoal-sid")
    result = dispatch_goal_command(mgr, "queue brand new objective", authorize_gate=lambda: None)
    assert result.kickoff is True
    assert mgr.state.goal == "brand new objective"


def test_dispatch_queue_clear():
    mgr = _mgr_with_goal("dispatch-clear-sid")
    mgr.queue_goal("doomed goal")
    result = dispatch_goal_command(mgr, "queue clear", authorize_gate=lambda: None)
    assert "Cleared 1" in result.output
    assert mgr.state.queued_goals == []


def test_dispatch_queue_remove_requires_index():
    mgr = _mgr_with_goal("dispatch-rm-sid")
    mgr.queue_goal("kept goal")
    result = dispatch_goal_command(mgr, "queue remove", authorize_gate=lambda: None)
    assert result.error is True
    assert mgr.state.queued_goals == ["kept goal"]
    result = dispatch_goal_command(mgr, "queue remove 1", authorize_gate=lambda: None)
    assert "kept goal" in result.output
    assert mgr.state.queued_goals == []
