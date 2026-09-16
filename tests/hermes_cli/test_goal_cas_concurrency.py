"""Durable goal-state concurrency and short critical-section regressions."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import goals
from hermes_cli.loops import goal_blocks_loop_tick


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    goals._DB_BOOTSTRAP_INFLIGHT.clear()
    yield home
    goals._DB_CACHE.clear()
    goals._DB_BOOTSTRAP_INFLIGHT.clear()


def test_manager_constructed_before_clear_cannot_resume_or_evaluate(isolated_home):
    session_id = "stale-manager-clear"
    creator = goals.GoalManager(session_id)
    creator.set("remain stopped")
    stale = goals.GoalManager(session_id)

    clearer = goals.GoalManager(session_id)
    clearer.clear()

    assert stale.resume() is None
    with patch("hermes_cli.goals.judge_goal") as judge:
        decision = stale.evaluate_after_turn("stale response")

    judge.assert_not_called()
    assert decision["should_continue"] is False
    assert goals.load_goal_authoritative(session_id).status == "cleared"


def test_independent_sessiondb_revision_change_rejects_stale_publication(isolated_home):
    from hermes_state import SessionDB

    session_id = "independent-writer"
    manager = goals.GoalManager(session_id)
    manager.set("do not overwrite the pause")
    independent = SessionDB(Path(isolated_home) / "state.db")

    def judge_then_competing_write(*_args, **_kwargs):
        key = goals._meta_key(session_id)
        before_raw = independent.get_meta(key)
        competing = goals.GoalState.from_json(before_raw)
        competing.status = "paused"
        competing.paused_reason = "independent writer"
        competing.revision += 1
        assert independent.compare_and_set_meta(
            key, before_raw, competing.to_json()
        ) is True
        return "continue", "stale verdict", False, None, False

    try:
        with patch("hermes_cli.goals.judge_goal", side_effect=judge_then_competing_write):
            decision = manager.evaluate_after_turn("old response")
    finally:
        independent.close()

    assert decision["verdict"] == "stale"
    assert decision["should_continue"] is False
    persisted = goals.load_goal_authoritative(session_id)
    assert persisted.status == "paused"
    assert persisted.paused_reason == "independent writer"
    assert persisted.revision == 2


def _assert_control_crosses_cas_boundary_while_slow_io_blocks(
    monkeypatch, *, slow_kind: str
):
    primary_id = f"slow-{slow_kind}-primary"
    unrelated_id = f"slow-{slow_kind}-unrelated"
    primary = goals.GoalManager(primary_id)
    primary.set("evaluate slowly")
    if slow_kind == "gate":
        primary.add_gate("fake slow gate")

    slow_entered = threading.Event()
    release_slow = threading.Event()
    control_cas_entered = threading.Event()
    control_done = threading.Event()
    unrelated_done = threading.Event()

    db = goals._get_session_db()
    real_cas = db.compare_and_set_meta

    def observed_cas(key, expected, replacement):
        if threading.current_thread().name.startswith("goal-control-"):
            control_cas_entered.set()
        return real_cas(key, expected, replacement)

    monkeypatch.setattr(db, "compare_and_set_meta", observed_cas)

    if slow_kind == "judge":
        def slow_judge(*_args, **_kwargs):
            slow_entered.set()
            assert release_slow.wait(5)
            return "continue", "judge finished", False, None, False

        monkeypatch.setattr(goals, "judge_goal", slow_judge)
    else:
        def slow_gate(*_args, **_kwargs):
            slow_entered.set()
            assert release_slow.wait(5)
            return False, 1, "gate failed"

        monkeypatch.setattr(goals, "run_gate", slow_gate)

    decision_box = {}

    def evaluate():
        decision_box["decision"] = primary.evaluate_after_turn("turn complete")

    evaluator = threading.Thread(target=evaluate, name=f"goal-evaluate-{slow_kind}")
    evaluator.start()
    assert slow_entered.wait(2), f"{slow_kind} did not start"

    def pause_same_session():
        primary.pause("urgent pause")
        control_done.set()

    def set_unrelated_session():
        goals.GoalManager(unrelated_id).set("unrelated control")
        unrelated_done.set()

    control = threading.Thread(target=pause_same_session, name="goal-control-pause")
    unrelated = threading.Thread(target=set_unrelated_session, name="goal-control-unrelated")
    control.start()
    unrelated.start()
    try:
        assert control_cas_entered.wait(1), "control never reached the actual CAS boundary"
        assert control_done.wait(1), "same-session pause convoyed behind slow I/O"
        assert unrelated_done.wait(1), "unrelated-session control convoyed behind slow I/O"
    finally:
        release_slow.set()
        evaluator.join(5)
        control.join(5)
        unrelated.join(5)

    assert not evaluator.is_alive()
    assert decision_box["decision"]["should_continue"] is False
    assert goals.load_goal_authoritative(primary_id).status == "paused"
    assert goals.load_goal_authoritative(unrelated_id).status == "active"


@pytest.mark.parametrize("slow_kind", ["judge", "gate"])
def test_slow_io_does_not_block_same_or_unrelated_session_controls(
    isolated_home, monkeypatch, slow_kind
):
    _assert_control_crosses_cas_boundary_while_slow_io_blocks(
        monkeypatch, slow_kind=slow_kind
    )


def test_responsiveness_probe_falsifies_old_broad_lock_shape():
    broad_lock = threading.Lock()
    slow_entered = threading.Event()
    release_slow = threading.Event()
    control_boundary = threading.Event()
    control_done = threading.Event()

    def old_evaluate_shape():
        with broad_lock:
            slow_entered.set()
            assert release_slow.wait(5)

    def old_control_shape():
        control_boundary.set()
        with broad_lock:
            control_done.set()

    evaluator = threading.Thread(target=old_evaluate_shape)
    evaluator.start()
    assert slow_entered.wait(1)
    control = threading.Thread(target=old_control_shape)
    control.start()
    try:
        assert control_boundary.wait(1)
        with pytest.raises(AssertionError, match="old broad lock blocks control"):
            assert control_done.wait(0.05), "old broad lock blocks control"
    finally:
        release_slow.set()
        evaluator.join(2)
        control.join(2)


@pytest.mark.parametrize("wait_kind", ["time", "pid", "session"])
def test_fresh_manager_expired_wait_cleanup_is_authoritative_and_blocks_loop(
    isolated_home, monkeypatch, wait_kind
):
    session_id = f"expired-{wait_kind}"
    manager = goals.GoalManager(session_id)
    manager.set("goal owns the idle boundary")
    if wait_kind == "time":
        manager.wait_for_seconds(1)
        monkeypatch.setattr(goals.time, "time", lambda: 10**12)
    elif wait_kind == "pid":
        monkeypatch.setattr(goals, "_pid_alive", lambda _pid: True)
        manager.wait_on(999999)
        monkeypatch.setattr(goals, "_pid_alive", lambda _pid: False)
    else:
        manager.wait_on_session("finished-worker")
        monkeypatch.setattr(goals, "_session_waiting", lambda _sid: False)

    before = goals.load_goal_authoritative(session_id)
    assert goal_blocks_loop_tick(session_id) is True

    persisted = goals.load_goal_authoritative(session_id)
    assert persisted.status == "active"
    assert persisted.revision == before.revision + 1
    assert persisted.waiting_on_pid is None
    assert persisted.waiting_on_session is None
    assert persisted.waiting_until == 0.0


@pytest.mark.parametrize("failure", ["unavailable", "read", "invalid", "drop", "conflict"])
def test_wait_arbitration_fails_closed_when_authoritative_cleanup_is_unavailable(
    isolated_home, monkeypatch, failure
):
    session_id = f"wait-failure-{failure}"
    expired = goals.GoalState(
        goal="possibly still active",
        status="active",
        waiting_until=1.0,
        revision=4,
    ).to_json()

    class FailingDB:
        def get_meta(self, _key):
            if failure == "read":
                raise OSError("read unavailable")
            if failure == "invalid":
                return "{invalid goal json"
            return expired

        def compare_and_set_meta(self, _key, _expected, _replacement):
            if failure == "drop":
                raise OSError("write dropped")
            return False

    monkeypatch.setattr(
        goals,
        "_get_session_db",
        (lambda: None) if failure == "unavailable" else (lambda: FailingDB()),
    )
    monkeypatch.setattr(goals.time, "time", lambda: 2.0)

    assert goal_blocks_loop_tick(session_id) is True


def test_competing_wait_writer_is_not_overwritten_or_admitted_as_loop(
    isolated_home, monkeypatch
):
    session_id = "competing-wait-writer"
    manager = goals.GoalManager(session_id)
    manager.set("original goal")
    manager.wait_for_seconds(1)
    monkeypatch.setattr(goals.time, "time", lambda: 10**12)

    db = goals._get_session_db()
    real_cas = db.compare_and_set_meta
    competed = False

    def competing_cas(key, expected, replacement):
        nonlocal competed
        if not competed:
            competed = True
            current_raw = db.get_meta(key)
            current = goals.GoalState.from_json(current_raw)
            current.goal = "independent writer wins"
            current.waiting_until = 0.0
            current.revision += 1
            assert real_cas(key, current_raw, current.to_json()) is True
        return real_cas(key, expected, replacement)

    monkeypatch.setattr(db, "compare_and_set_meta", competing_cas)

    assert goal_blocks_loop_tick(session_id) is True
    persisted = goals.load_goal_authoritative(session_id)
    assert persisted.goal == "independent writer wins"
    assert persisted.status == "active"
    assert persisted.waiting_until == 0.0
