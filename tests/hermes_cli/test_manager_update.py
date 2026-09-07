"""Behavioral contract tests for manager update() methods and race safety."""

from __future__ import annotations

import json
import threading
import time

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class TestGoalManagerUpdate:
    def test_update_requires_active_or_paused_goal(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState, save_goal
        save_goal("s1", GoalState(goal="done goal", status="done"))
        mgr = GoalManager(session_id="s1")
        with pytest.raises(RuntimeError, match="done"):
            mgr.update("new prompt")

    def test_update_refuses_cleared_or_missing(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="s-none")
        with pytest.raises(RuntimeError):
            mgr.update("new prompt")

    def test_update_changes_prompt_preserves_runtime(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState, save_goal, load_goal
        save_goal("s2", GoalState(
            goal="original", status="active", turns_used=7, max_turns=20,
            created_at=100.0, last_turn_at=200.0,
            last_verdict="continue", last_reason="still working",
            subgoals=["criterion A"],
        ))
        mgr = GoalManager(session_id="s2")
        result = mgr.update("revised", max_turns=30)
        assert result.goal == "revised"
        assert result.max_turns == 30
        state = load_goal("s2")
        assert state.turns_used == 7
        assert state.created_at == 100.0
        assert state.last_turn_at == 200.0
        assert state.last_verdict == "continue"
        assert state.last_reason == "still working"
        assert state.status == "active"
        assert state.subgoals == ["criterion A"]

    def test_update_criteria_absent_preserves(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState, save_goal, load_goal
        save_goal("s3", GoalState(goal="original", status="active", subgoals=["A", "B"]))
        mgr = GoalManager(session_id="s3")
        mgr.update("revised")
        state = load_goal("s3")
        assert state.subgoals == ["A", "B"]

    def test_update_criteria_empty_list_clears(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState, save_goal, load_goal
        save_goal("s4", GoalState(goal="original", status="active", subgoals=["A", "B"]))
        mgr = GoalManager(session_id="s4")
        mgr.update("revised", criteria=[])
        state = load_goal("s4")
        assert state.subgoals == []

    def test_update_criteria_replaces(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState, save_goal, load_goal
        save_goal("s5", GoalState(goal="original", status="active", subgoals=["old"]))
        mgr = GoalManager(session_id="s5")
        mgr.update("revised", criteria=["new1", "new2"])
        state = load_goal("s5")
        assert state.subgoals == ["new1", "new2"]

    def test_update_preserves_contract_gates_wait_barrier(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState, GoalContract, GoalGate, save_goal, load_goal
        contract = GoalContract(outcome="must work")
        gate = GoalGate(
            command="echo ok", timeout_seconds=30, max_retries=3,
            attempts=2, last_exit_code=1, last_output_tail="still red",
            last_failed_fingerprint="abc123",
        )
        save_goal("s6", GoalState(
            goal="original", status="paused", turns_used=9, max_turns=20,
            created_at=100.0, last_turn_at=200.0,
            last_verdict="blocked", last_reason="dependency missing",
            paused_reason="budget exhausted",
            consecutive_parse_failures=2, consecutive_transport_failures=3,
            contract=contract, gates=[gate],
            waiting_on_pid=999, waiting_on_session="child-session",
            waiting_until=500.0, waiting_on_delegations=4,
            waiting_reason="test", waiting_since=300.0,
        ))
        mgr = GoalManager(session_id="s6")
        mgr.update("revised")
        state = load_goal("s6")
        assert state.contract.outcome == "must work"
        assert len(state.gates) == 1
        assert state.gates[0].command == "echo ok"
        assert state.gates[0].attempts == 2
        assert state.gates[0].last_exit_code == 1
        assert state.gates[0].last_output_tail == "still red"
        assert state.gates[0].last_failed_fingerprint == "abc123"
        assert state.status == "paused"
        assert state.turns_used == 9
        assert state.created_at == 100.0
        assert state.last_turn_at == 200.0
        assert state.last_verdict == "blocked"
        assert state.last_reason == "dependency missing"
        assert state.paused_reason == "budget exhausted"
        assert state.consecutive_parse_failures == 2
        assert state.consecutive_transport_failures == 3
        assert state.waiting_on_pid == 999
        assert state.waiting_on_session == "child-session"
        assert state.waiting_until == 500.0
        assert state.waiting_on_delegations == 4
        assert state.waiting_reason == "test"
        assert state.waiting_since == 300.0

    def test_stale_manager_cannot_resurrect_cleared_goal(self, hermes_home):
        from hermes_cli.goals import GoalManager

        owner = GoalManager(session_id="stale-goal")
        owner.set("original")
        stale = GoalManager(session_id="stale-goal")
        owner.clear()

        with pytest.raises(RuntimeError, match="editable"):
            stale.update("resurrected")

    def test_stale_goal_edit_cannot_lower_cap_below_fresh_turn_count(self, hermes_home):
        from hermes_cli.goals import GoalManager, load_goal, save_goal

        stale = GoalManager(session_id="stale-budget")
        stale.set("objective", max_turns=10)
        fresh = load_goal("stale-budget")
        assert fresh is not None
        fresh.turns_used = 6
        save_goal("stale-budget", fresh)

        with pytest.raises(RuntimeError, match="below turns already used"):
            stale.update("edited", max_turns=5)

        persisted = load_goal("stale-budget")
        assert persisted is not None
        assert persisted.goal == "objective"
        assert persisted.max_turns == 10


class TestLoopManagerUpdate:
    def test_update_requires_active_or_paused(self, hermes_home):
        from hermes_cli.loops import LoopManager, LoopState, save_loop
        save_loop("s1", LoopState(prompt="done", status="done"))
        mgr = LoopManager(session_id="s1")
        with pytest.raises(RuntimeError, match="done"):
            mgr.update("new prompt", interval_seconds=120)

    def test_update_refuses_awaiting_response(self, hermes_home):
        from hermes_cli.loops import LoopManager, LoopState, save_loop
        save_loop("s2", LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60, awaiting_response=True,
        ))
        mgr = LoopManager(session_id="s2")
        with pytest.raises(RuntimeError, match="awaiting"):
            mgr.update("new prompt", interval_seconds=120)

    def test_update_refuses_cap_below_ticks(self, hermes_home):
        from hermes_cli.loops import LoopManager, LoopState, save_loop
        save_loop("s3", LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60, ticks_fired=10,
        ))
        mgr = LoopManager(session_id="s3")
        with pytest.raises(RuntimeError, match="below"):
            mgr.update("new prompt", interval_seconds=120, times=5)

    def test_update_preserves_route_and_history(self, hermes_home):
        from hermes_cli.loops import LoopManager, LoopState, save_loop, load_loop
        save_loop("s4", LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60,
            ticks_fired=5, created_at=100.0, last_fired_at=500.0,
            route={"platform": "desktop", "chat_id": "abc"},
            last_response_digest="some digest",
        ))
        mgr = LoopManager(session_id="s4")
        result = mgr.update("revised", interval_seconds=120)
        assert result.prompt == "revised"
        assert result.interval_seconds >= 120
        state = load_loop("s4")
        assert state.ticks_fired == 5
        assert state.created_at == 100.0
        assert state.last_fired_at == 500.0
        assert state.route == {"platform": "desktop", "chat_id": "abc"}
        assert state.status == "active"
        assert state.last_response_digest == "some digest"

    def test_update_rearms_next_due_at(self, hermes_home):
        from hermes_cli.loops import LoopManager, LoopState, save_loop, load_loop
        save_loop("s5", LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60,
            next_due_at=100.0,
        ))
        mgr = LoopManager(session_id="s5")
        before = time.time()
        mgr.update("revised", interval_seconds=120)
        state = load_loop("s5")
        assert before + 120 <= state.next_due_at <= time.time() + 120
        assert state.mode == "interval"
        assert state.current_delay == state.interval_seconds

    def test_update_empty_until_clears(self, hermes_home):
        from hermes_cli.loops import LoopManager, LoopState, save_loop, load_loop
        save_loop("s6", LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60, until="task complete",
        ))
        mgr = LoopManager(session_id="s6")
        mgr.update("revised", interval_seconds=120, until="")
        state = load_loop("s6")
        assert state.until == ""

    def test_update_race_with_fire_tick_has_only_serialized_outcomes(self, hermes_home, monkeypatch):
        from hermes_cli import goals
        from hermes_cli.loops import LoopManager, LoopState, save_loop, load_loop
        save_loop("s7", LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60,
            next_due_at=0.0,
        ))
        db = goals._get_session_db()
        original_mutate = db.mutate_meta
        rendezvous = threading.Barrier(2)

        def synchronized_mutate(key, mutator):
            if key == "loop:s7":
                rendezvous.wait(timeout=5)
            return original_mutate(key, mutator)

        monkeypatch.setattr(db, "mutate_meta", synchronized_mutate)
        results = {"update_ok": False, "tick_message": None}
        errors = []

        def _editor():
            try:
                mgr = LoopManager(session_id="s7")
                mgr.update("edited prompt", interval_seconds=120)
                results["update_ok"] = True
            except Exception as exc:
                errors.append(("update", exc))

        def _ticker():
            try:
                results["tick_message"] = LoopManager(session_id="s7").fire_tick()
            except Exception as exc:
                errors.append(("tick", exc))

        t1 = threading.Thread(target=_editor)
        t2 = threading.Thread(target=_ticker)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert all(not thread.is_alive() for thread in (t1, t2))
        state = load_loop("s7")
        assert state is not None
        if results["update_ok"]:
            assert not errors
            assert results["tick_message"] is None
            assert state.prompt == "edited prompt"
            assert state.ticks_fired == 0
            assert state.awaiting_response is False
        else:
            assert len(errors) == 1
            assert errors[0][0] == "update"
            assert "awaiting" in str(errors[0][1]).lower()
            assert results["tick_message"] is not None
            assert state.prompt == "original"
            assert state.ticks_fired == 1
            assert state.awaiting_response is True

    def test_stale_manager_cannot_resurrect_cleared_loop(self, hermes_home):
        from hermes_cli.loops import LoopManager

        owner = LoopManager(session_id="stale-loop")
        owner.set("original", interval_seconds=60)
        stale = LoopManager(session_id="stale-loop")
        owner.clear()

        with pytest.raises(RuntimeError, match="editable"):
            stale.update("resurrected", interval_seconds=120)


class TestHeartbeatManagerUpdate:
    def test_update_requires_active_or_paused(self, hermes_home):
        from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat
        save_heartbeat("s1", HeartbeatState(prompt="done", interval_seconds=60, status="cleared"))
        mgr = HeartbeatManager(session_id="s1")
        with pytest.raises(RuntimeError):
            mgr.update("new prompt", interval_seconds=120)

    def test_update_preserves_counters_and_timestamps(self, hermes_home):
        from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat, load_heartbeat
        save_heartbeat("s2", HeartbeatState(
            prompt="original", interval_seconds=60, status="active",
            created_at=100.0, last_fired_at=500.0, fire_count=8,
        ))
        mgr = HeartbeatManager(session_id="s2")
        result = mgr.update("revised", interval_seconds=300)
        assert result.prompt == "revised"
        assert result.interval_seconds == 300
        state = load_heartbeat("s2")
        assert state.fire_count == 8
        assert state.created_at == 100.0
        assert state.last_fired_at == 500.0
        assert state.status == "active"

    def test_update_enforces_min_interval(self, hermes_home):
        from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat, MIN_INTERVAL_SECONDS
        save_heartbeat("s3", HeartbeatState(prompt="original", interval_seconds=120, status="active"))
        mgr = HeartbeatManager(session_id="s3")
        with pytest.raises(ValueError, match="at least"):
            mgr.update("revised", interval_seconds=10)

    def test_update_interval_shrink_makes_immediately_due(self, hermes_home):
        from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat, load_heartbeat, MIN_INTERVAL_SECONDS
        fired_at = time.time() - (MIN_INTERVAL_SECONDS + 5)
        save_heartbeat("s4", HeartbeatState(
            prompt="original", interval_seconds=3600, status="active",
            created_at=fired_at - 100, last_fired_at=fired_at,
        ))
        mgr = HeartbeatManager(session_id="s4")
        mgr.update("revised", interval_seconds=MIN_INTERVAL_SECONDS)
        state = load_heartbeat("s4")
        assert state.interval_seconds == MIN_INTERVAL_SECONDS
        assert state.last_fired_at == fired_at
        assert state.is_due(time.time()) is True

    def test_update_race_with_due_prompt_loses_neither_edit_nor_fire(self, hermes_home, monkeypatch):
        from hermes_cli import goals
        from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat, load_heartbeat, MIN_INTERVAL_SECONDS
        save_heartbeat("s5", HeartbeatState(
            prompt="original", interval_seconds=MIN_INTERVAL_SECONDS, status="active",
            created_at=time.time() - 200,
            last_fired_at=time.time() - 120,
        ))
        db = goals._get_session_db()
        original_mutate = db.mutate_meta
        rendezvous = threading.Barrier(2)

        def synchronized_mutate(key, mutator):
            if key == "heartbeat:s5":
                rendezvous.wait(timeout=5)
            return original_mutate(key, mutator)

        monkeypatch.setattr(db, "mutate_meta", synchronized_mutate)
        errors = []
        result = {"prompt": None}

        def _editor():
            try:
                mgr = HeartbeatManager(session_id="s5")
                mgr.update("edited", interval_seconds=MIN_INTERVAL_SECONDS + 10)
            except Exception as exc:
                errors.append(("update", exc))

        def _firer():
            try:
                result["prompt"] = HeartbeatManager(session_id="s5").due_prompt()
            except Exception as exc:
                errors.append(("fire", exc))

        t1 = threading.Thread(target=_editor)
        t2 = threading.Thread(target=_firer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert all(not thread.is_alive() for thread in (t1, t2))
        assert not errors, f"Errors during race: {errors}"
        state = load_heartbeat("s5")
        assert state is not None
        assert state.prompt == "edited"
        assert state.interval_seconds == MIN_INTERVAL_SECONDS + 10
        assert state.fire_count == 1
        assert result["prompt"] is not None

    def test_stale_manager_cannot_resurrect_cleared_heartbeat(self, hermes_home):
        from hermes_cli.heartbeat import HeartbeatManager

        owner = HeartbeatManager(session_id="stale-heartbeat")
        owner.set("original", interval_seconds=60)
        stale = HeartbeatManager(session_id="stale-heartbeat")
        owner.clear()

        with pytest.raises(RuntimeError, match="editable"):
            stale.update("resurrected", interval_seconds=120)
