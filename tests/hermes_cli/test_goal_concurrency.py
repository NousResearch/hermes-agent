"""Durable GoalManager concurrency contracts using temporary SQLite state."""
from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from hermes_cli import goals
from hermes_cli.goals import GoalManager


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.mark.parametrize("control", ["pause", "clear", "set"])
def test_slow_judge_cannot_overwrite_pause_clear_or_new_goal(hermes_home, control):
    sid = f"judge-control-{control}"
    manager = GoalManager(sid)
    manager.set("old goal")

    def control_update(*_args, **_kwargs):
        current = GoalManager(sid)
        if control == "pause":
            current.pause("user paused")
        elif control == "clear":
            current.clear()
        else:
            current.set("new goal")
        return "continue", "old result", False, None, False

    with patch.object(goals, "judge_goal", side_effect=control_update):
        decision = manager.evaluate_after_turn("old response")

    latest = GoalManager(sid).state
    assert decision == {
        "status": "inactive",
        "should_continue": False,
        "continuation_prompt": None,
        "verdict": "inactive",
        "reason": "goal changed during evaluation",
        "message": "",
    }
    assert latest is not None
    if control == "pause":
        assert latest.status == "paused"
    elif control == "clear":
        assert latest.status == "cleared"
    else:
        assert latest.goal == "new goal"
        assert latest.status == "active"
        assert latest.turns_used == 0


def test_done_and_continue_decisions_match_durable_state(hermes_home):
    for verdict, expected_status, should_continue in (
        ("done", "done", False),
        ("continue", "active", True),
    ):
        sid = f"judge-result-{verdict}"
        manager = GoalManager(sid)
        manager.set("ship it")
        with patch.object(
            goals,
            "judge_goal",
            return_value=(verdict, "judge reason", False, None, False),
        ):
            decision = manager.evaluate_after_turn("response")
        latest = GoalManager(sid).state
        assert latest is not None and latest.status == expected_status
        assert decision["status"] == expected_status
        assert decision["should_continue"] is should_continue
        assert decision["message"]


def test_wait_does_not_burn_another_turn(hermes_home):
    manager = GoalManager("judge-wait-budget")
    manager.set("wait for CI", max_turns=5)
    with patch.object(
        goals,
        "judge_goal",
        return_value=("wait", "CI is running", False, {"seconds": 60}, False),
    ) as judge:
        first = manager.evaluate_after_turn("started CI")
        second = manager.evaluate_after_turn("still waiting")

    assert first["verdict"] == "wait"
    assert first["should_continue"] is False
    assert second["verdict"] == "waiting"
    assert second["should_continue"] is False
    assert manager.state.turns_used == 1
    assert GoalManager("judge-wait-budget").state.turns_used == 1
    judge.assert_called_once()


@pytest.mark.parametrize("control", ["pause", "clear", "set"])
def test_slow_gate_cannot_overwrite_control_update(hermes_home, control):
    sid = f"gate-control-race-{control}"
    manager = GoalManager(sid)
    manager.set("run gate")
    manager.add_gate("check")
    gate_started = threading.Event()
    release_gate = threading.Event()

    def slow_gate(*_args, **_kwargs):
        gate_started.set()
        assert release_gate.wait(5), "gate test barrier was not released"
        return False, 7, "still failing"

    result = {}

    def evaluate():
        with patch.object(goals, "run_gate", side_effect=slow_gate):
            result["decision"] = manager.evaluate_after_turn("response")

    worker = threading.Thread(target=evaluate)
    worker.start()
    assert gate_started.wait(5), "gate did not start"
    current = GoalManager(sid)
    if control == "pause":
        current.pause("paused during gate")
    elif control == "clear":
        current.clear()
    else:
        current.set("new gate goal")
    release_gate.set()
    worker.join(5)
    assert not worker.is_alive()

    latest = GoalManager(sid).state
    assert latest is not None
    if control == "pause":
        assert latest.status == "paused"
    elif control == "clear":
        assert latest.status == "cleared"
    else:
        assert latest.goal == "new gate goal"
        assert latest.turns_used == 0
    assert result["decision"]["status"] == "inactive"
    assert result["decision"]["should_continue"] is False
    assert result["decision"]["continuation_prompt"] is None
    assert result["decision"]["message"] == ""


def test_judge_exception_leaves_durable_state_unchanged(hermes_home):
    manager = GoalManager("judge-exception")
    manager.set("keep state")
    with patch.object(goals, "judge_goal", side_effect=RuntimeError("judge failed")):
        with pytest.raises(RuntimeError, match="judge failed"):
            manager.evaluate_after_turn("response")

    latest = GoalManager("judge-exception").state
    assert latest is not None
    assert latest.status == "active"
    assert latest.turns_used == 0
    assert latest.last_verdict is None
