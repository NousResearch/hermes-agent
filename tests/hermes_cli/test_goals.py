"""Tests for hermes_cli/goals.py — persistent cross-turn goals."""

from __future__ import annotations

import json
import threading
import time

import pytest


def _begin_goal_turn(mgr, turn_id: str = "model-turn") -> dict:
    """Return the exact mechanical authority required by goal writers."""

    generation_id = mgr.begin_model_turn(turn_id)
    assert generation_id
    return {
        "originating_turn_id": turn_id,
        "goal_generation_id": generation_id,
    }


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module's DB cache for each test so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# GoalManager lifecycle + persistence
# ──────────────────────────────────────────────────────────────────────


class TestGoalManager:
    def test_no_goal_initial(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-1")
        assert mgr.state is None
        assert not mgr.is_active()
        assert not mgr.has_goal()
        assert "No active goal" in mgr.status_line()

    def test_set_then_status(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-2", default_max_turns=5)
        state = mgr.set("port the thing")
        assert state.goal == "port the thing"
        assert state.status == "active"
        assert state.max_turns == 5
        assert state.turns_used == 0
        assert mgr.is_active()
        assert "active" in mgr.status_line().lower()
        assert "port the thing" in mgr.status_line()

    def test_set_rejects_empty(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-3")
        with pytest.raises(ValueError):
            mgr.set("")
        with pytest.raises(ValueError):
            mgr.set("   ")

    def test_pause_and_resume(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-4")
        mgr.set("goal text")
        mgr.pause(reason="user-paused")
        assert mgr.state.status == "paused"
        assert not mgr.is_active()
        assert mgr.has_goal()

        mgr.resume()
        assert mgr.state.status == "active"
        assert mgr.is_active()

    def test_clear(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-5")
        mgr.set("goal")
        mgr.clear()
        assert mgr.state is None
        assert not mgr.is_active()

    def test_persistence_across_managers(self, hermes_home):
        """Key invariant: a second manager on the same session sees the goal.

        This is what makes /resume work — each session rebinds its
        GoalManager and picks up the saved state.
        """
        from hermes_cli.goals import GoalManager

        mgr1 = GoalManager(session_id="persist-sid")
        mgr1.set("do the thing")

        mgr2 = GoalManager(session_id="persist-sid")
        assert mgr2.state is not None
        assert mgr2.state.goal == "do the thing"
        assert mgr2.is_active()

    def test_evaluate_after_turn_done(self, hermes_home):
        """Primary model says complete → status=done, no continuation."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-1")
        mgr.set("ship it")

        authority = _begin_goal_turn(mgr)
        mgr.record_model_outcome("complete", "shipped", **authority)
        decision = mgr.evaluate_after_turn("I shipped the feature.", **authority)

        assert decision["verdict"] == "done"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert mgr.state.status == "done"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_continue_under_budget(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-2", default_max_turns=5)
        mgr.set("a long goal")

        authority = _begin_goal_turn(mgr)
        mgr.record_model_outcome("continue", "more work", **authority)
        decision = mgr.evaluate_after_turn("made some progress", **authority)

        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert decision["continuation_prompt"] is not None
        assert "a long goal" in decision["continuation_prompt"]
        assert mgr.state.status == "active"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_budget_exhausted(self, hermes_home):
        """When turn budget hits ceiling, auto-pause instead of continuing."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-3", default_max_turns=2)
        mgr.set("hard goal")

        authority = _begin_goal_turn(mgr, "turn-1")
        mgr.record_model_outcome("continue", "first step remains", **authority)
        d1 = mgr.evaluate_after_turn("step 1", **authority)
        assert d1["should_continue"] is True
        assert mgr.state.turns_used == 1
        assert mgr.state.status == "active"

        authority = _begin_goal_turn(mgr, "turn-2")
        mgr.record_model_outcome("continue", "second step remains", **authority)
        d2 = mgr.evaluate_after_turn("step 2", **authority)
        # turns_used is now 2 which equals max_turns → paused
        assert d2["should_continue"] is False
        assert mgr.state.status == "paused"
        assert mgr.state.turns_used == 2
        assert "budget" in (mgr.state.paused_reason or "").lower()

    def test_zero_budget_continues_until_model_authored_outcome(self, hermes_home):
        """Zero removes only the arbitrary cross-turn pause."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-unbounded", default_max_turns=0)
        state = mgr.set("finish the approved plan")
        assert state.max_turns == 0

        for index in range(25):
            authority = _begin_goal_turn(mgr, f"turn-{index}")
            mgr.record_model_outcome(
                "continue",
                f"verified progress {index}",
                **authority,
            )
            decision = mgr.evaluate_after_turn(
                f"step {index}",
                **authority,
            )
            assert decision["should_continue"] is True
            assert decision["verdict"] == "continue"

        restored = GoalManager(
            session_id="eval-unbounded",
            default_max_turns=20,
        )
        assert restored.state is not None
        assert restored.state.max_turns == 0
        assert restored.state.status == "active"
        assert restored.state.turns_used == 25
        assert "no automatic turn cap" in restored.status_line()

        authority = _begin_goal_turn(restored, "turn-complete")
        restored.record_model_outcome(
            "complete",
            "all criteria verified",
            **authority,
        )
        completed = restored.evaluate_after_turn("done", **authority)
        assert completed["should_continue"] is False
        assert completed["verdict"] == "done"

    def test_negative_default_budget_is_rejected(self, hermes_home):
        from hermes_cli.goals import GoalManager

        with pytest.raises(ValueError, match="non-negative"):
            GoalManager(session_id="eval-invalid-budget", default_max_turns=-1)
        with pytest.raises(ValueError, match="non-negative integer"):
            GoalManager(session_id="eval-bool-budget", default_max_turns=False)

    def test_missing_model_outcome_fails_open_to_continue(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-missing", default_max_turns=3)
        mgr.set("keep working")

        authority = _begin_goal_turn(mgr)
        decision = mgr.evaluate_after_turn("response prose says done", **authority)

        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert "not recorded" in decision["reason"]
        assert mgr.state.status == "active"

    def test_model_blocked_outcome_pauses(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-blocked")
        mgr.set("perform approved work")
        authority = _begin_goal_turn(mgr)
        mgr.record_model_outcome(
            "blocked", "owner credential is genuinely required", **authority
        )

        decision = mgr.evaluate_after_turn(
            "I exhausted the safe approaches.", **authority
        )

        assert decision["verdict"] == "blocked"
        assert decision["should_continue"] is False
        assert mgr.state.status == "paused"
        assert mgr.state.paused_reason == "owner credential is genuinely required"

    def test_record_model_outcome_validates_schema(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-validation")
        mgr.set("goal")
        authority = _begin_goal_turn(mgr)
        with pytest.raises(ValueError):
            mgr.record_model_outcome("done-ish", "ambiguous", **authority)
        with pytest.raises(ValueError):
            mgr.record_model_outcome("continue", "", **authority)

    def test_evaluate_after_turn_inactive(self, hermes_home):
        """evaluate_after_turn is a no-op when goal isn't active."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-4")
        d = mgr.evaluate_after_turn(
            "anything", originating_turn_id="none", goal_generation_id="none"
        )
        assert d["verdict"] == "inactive"
        assert d["should_continue"] is False

        mgr.set("a goal")
        mgr.pause()
        d2 = mgr.evaluate_after_turn(
            "anything", originating_turn_id="none", goal_generation_id="none"
        )
        assert d2["verdict"] == "inactive"
        assert d2["should_continue"] is False

    def test_continuation_prompt_shape(self, hermes_home):
        """The continuation prompt must include the goal text verbatim —
        and must be safe to inject as a user-role message (prompt-cache
        invariants: no system-prompt mutation)."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="cont-sid")
        mgr.set("port goal command to hermes")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "port goal command to hermes" in prompt
        assert "goal_outcome" in prompt
        assert prompt.strip()  # non-empty

    def test_kickoff_prompt_requires_structured_model_outcome(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="kickoff-sid")
        mgr.set("finish the migration")
        prompt = mgr.next_kickoff_prompt()

        assert "Begin working toward your standing goal" in prompt
        assert "finish the migration" in prompt
        assert "goal_outcome" in prompt
        assert "every safe available approach is exhausted" in prompt


# ──────────────────────────────────────────────────────────────────────
# Smoke: CommandDef is wired
# ──────────────────────────────────────────────────────────────────────


def test_goal_command_in_registry():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("goal")
    assert cmd is not None
    assert cmd.name == "goal"


def test_goal_command_dispatches_in_cli_registry_helpers():
    """goal shows up in autocomplete / help categories alongside other Session cmds."""
    from hermes_cli.commands import COMMANDS, COMMANDS_BY_CATEGORY

    assert "/goal" in COMMANDS
    session_cmds = COMMANDS_BY_CATEGORY.get("Session", {})
    assert "/goal" in session_cmds


# ──────────────────────────────────────────────────────────────────────
# /subgoal — user-added criteria
# ──────────────────────────────────────────────────────────────────────


class TestGoalStateSubgoalsBackcompat:
    def test_old_state_meta_row_loads_without_subgoals(self):
        """A goal serialized BEFORE the subgoals field existed must
        round-trip with an empty list, not crash."""
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "do a thing",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
        })
        state = GoalState.from_json(legacy)
        assert state.goal == "do a thing"
        assert state.subgoals == []

    def test_subgoals_round_trip(self):
        from hermes_cli.goals import GoalState
        state = GoalState(goal="g", subgoals=["a", "b", "c"])
        rt = GoalState.from_json(state.to_json())
        assert rt.subgoals == ["a", "b", "c"]


class TestMigrateGoalToSession:
    """migrate_goal_to_session carries a /goal from a parent session to its
    compression continuation child (#33618). load_goal does a flat
    per-session lookup with no lineage walk, so without migration an active
    goal silently dies when compression rotates session_id."""

    def test_migrates_active_goal_to_child(self, hermes_home):
        from hermes_cli.goals import save_goal, load_goal, migrate_goal_to_session, GoalState
        save_goal("parent-sid", GoalState(goal="ship the feature"))
        assert migrate_goal_to_session("parent-sid", "child-sid", reason="compression") is True
        child = load_goal("child-sid")
        assert child is not None and child.goal == "ship the feature"
        # Parent row archived (cleared) so only the child is active.
        parent = load_goal("parent-sid")
        assert parent is not None and parent.status == "cleared"

    def test_no_goal_to_migrate_returns_false(self, hermes_home):
        from hermes_cli.goals import migrate_goal_to_session, load_goal
        assert migrate_goal_to_session("empty-parent", "child2") is False
        assert load_goal("child2") is None

    def test_does_not_clobber_existing_child_goal(self, hermes_home):
        from hermes_cli.goals import save_goal, load_goal, migrate_goal_to_session, GoalState
        save_goal("p3", GoalState(goal="parent goal"))
        save_goal("c3", GoalState(goal="child already has one"))
        assert migrate_goal_to_session("p3", "c3") is False
        assert load_goal("c3").goal == "child already has one"

    def test_same_id_is_noop(self, hermes_home):
        from hermes_cli.goals import save_goal, migrate_goal_to_session, GoalState
        save_goal("same", GoalState(goal="g"))
        assert migrate_goal_to_session("same", "same") is False

    def test_cleared_goal_not_migrated(self, hermes_home):
        from hermes_cli.goals import save_goal, clear_goal, migrate_goal_to_session, load_goal, GoalState
        save_goal("p4", GoalState(goal="done already"))
        clear_goal("p4")
        assert migrate_goal_to_session("p4", "c4") is False
        assert load_goal("c4") is None


class TestGoalManagerSubgoals:
    def test_add_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-add")
        mgr.set("main goal")
        text = mgr.add_subgoal("  use bullet points  ")
        assert text == "use bullet points"
        assert mgr.state.subgoals == ["use bullet points"]

    def test_add_subgoal_requires_active_goal(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-noactive")
        with pytest.raises(RuntimeError):
            mgr.add_subgoal("oops")

    def test_add_empty_subgoal_rejected(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-empty")
        mgr.set("g")
        with pytest.raises(ValueError):
            mgr.add_subgoal("   ")

    def test_remove_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-remove")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")
        mgr.add_subgoal("third")
        removed = mgr.remove_subgoal(2)
        assert removed == "second"
        assert mgr.state.subgoals == ["first", "third"]

    def test_remove_subgoal_out_of_range(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-oob")
        mgr.set("g")
        mgr.add_subgoal("only")
        with pytest.raises(IndexError):
            mgr.remove_subgoal(5)
        with pytest.raises(IndexError):
            mgr.remove_subgoal(0)

    def test_clear_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-clear")
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        prev = mgr.clear_subgoals()
        assert prev == 2
        assert mgr.state.subgoals == []

    def test_subgoals_persist_across_reloads(self, hermes_home):
        """Subgoals stored in SessionDB survive a fresh GoalManager."""
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-persist")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")

        mgr2 = GoalManager(session_id="sub-persist")
        assert mgr2.state.subgoals == ["first", "second"]


class TestContinuationPromptWithSubgoals:
    def test_empty_subgoals_uses_original_template(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="cp-empty")
        mgr.set("ship the feature")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "ship the feature" in prompt
        assert "Additional criteria" not in prompt

    def test_with_subgoals_includes_them(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="cp-with")
        mgr.set("ship the feature")
        mgr.add_subgoal("write tests")
        mgr.add_subgoal("update docs")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "ship the feature" in prompt
        assert "Additional criteria" in prompt
        assert "1. write tests" in prompt
        assert "2. update docs" in prompt


class TestStatusLineSubgoalCount:
    def test_status_line_no_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-empty")
        mgr.set("ship it")
        line = mgr.status_line()
        assert "ship it" in line
        assert "subgoal" not in line.lower()

    def test_status_line_with_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-with")
        mgr.set("ship it")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        line = mgr.status_line()
        assert "2 subgoals" in line


# ──────────────────────────────────────────────────────────────────────
# Wait barrier — parking the goal loop on a background process
# ──────────────────────────────────────────────────────────────────────


class TestWaitBarrier:
    """The /goal wait barrier parks the loop on a live PID and resumes when
    the process exits, without burning turns or interpreting response prose."""

    @staticmethod
    def _spawn_sleeper():
        """Start a short-lived child process; return its Popen handle."""
        import subprocess
        import sys
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    @staticmethod
    def _dead_pid():
        """A PID that is essentially guaranteed not to be running."""
        return 2_000_000_000

    def test_wait_on_requires_active_goal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="wb-noactive")
        with pytest.raises(RuntimeError):
            mgr.wait_on(12345)

    def test_wait_on_rejects_bad_pid(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="wb-badpid")
        mgr.set("g")
        with pytest.raises(ValueError):
            mgr.wait_on(0)

    def test_parked_on_live_pid_does_not_continue(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-live")
            mgr.set("ship it", max_turns=5)
            mgr.wait_on(proc.pid, reason="CI green")
            assert mgr.is_waiting() is True

            # No turn is burned and response prose is not interpreted.
            authority = _begin_goal_turn(mgr)
            decision = mgr.evaluate_after_turn("still waiting on CI", **authority)

            assert decision["verdict"] == "waiting"
            assert decision["should_continue"] is False
            assert decision["continuation_prompt"] is None
            assert mgr.state.turns_used == 0  # no turn consumed while parked
            assert "CI green" in decision["message"]
            assert mgr.state.status == "active"  # still active, just parked
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_barrier_auto_clears_when_process_exits_and_loop_resumes(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        mgr = GoalManager(session_id="wb-exit")
        mgr.set("ship it", max_turns=5)
        mgr.wait_on(proc.pid, reason="build")
        assert mgr.is_waiting() is True

        # Kill the process — barrier should auto-clear and outcome handling resumes.
        proc.terminate()
        proc.wait(timeout=10)

        assert mgr.is_waiting() is False  # lazy auto-clear
        assert mgr.state.waiting_on_pid is None

        authority = _begin_goal_turn(mgr)
        mgr.record_model_outcome("continue", "more", **authority)
        decision = mgr.evaluate_after_turn(
            "process finished, here are results", **authority
        )

        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert mgr.state.turns_used == 1  # now a turn IS consumed

    def test_dead_pid_never_parks(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="wb-dead")
        mgr.set("g", max_turns=5)
        mgr.wait_on(self._dead_pid(), reason="already-dead")
        # is_waiting clears the stale barrier immediately.
        assert mgr.is_waiting() is False

        authority = _begin_goal_turn(mgr)
        mgr.record_model_outcome("continue", "go", **authority)
        decision = mgr.evaluate_after_turn("response", **authority)
        assert decision["should_continue"] is True

    def test_stop_waiting_clears_barrier(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-stop")
            mgr.set("g")
            mgr.wait_on(proc.pid)
            assert mgr.is_waiting() is True
            assert mgr.stop_waiting() is True
            assert mgr.state.waiting_on_pid is None
            assert mgr.is_waiting() is False
            assert mgr.stop_waiting() is False  # idempotent
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_pause_and_resume_clear_barrier(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-pause")
            mgr.set("g")
            mgr.wait_on(proc.pid)
            mgr.pause()
            assert mgr.state.waiting_on_pid is None

            mgr.resume()
            assert mgr.state.waiting_on_pid is None
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_barrier_persists_and_reloads(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-persist")
            mgr.set("g")
            mgr.wait_on(proc.pid, reason="deploy")

            # Fresh manager loads the persisted barrier.
            mgr2 = GoalManager(session_id="wb-persist")
            assert mgr2.state.waiting_on_pid == proc.pid
            assert mgr2.state.waiting_reason == "deploy"
            assert mgr2.is_waiting() is True
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_old_state_row_loads_without_barrier_fields(self, hermes_home):
        """Backwards-compat: a state_meta row written before the barrier
        existed must load with no barrier."""
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "old goal",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
        })
        st = GoalState.from_json(legacy)
        assert st.goal == "old goal"
        assert st.waiting_on_pid is None
        assert st.waiting_reason is None
        assert st.waiting_since == 0.0
        assert st.waiting_until == 0.0


# ──────────────────────────────────────────────────────────────────────
# Explicit mechanical wait controls
# ──────────────────────────────────────────────────────────────────────


class TestMechanicalWait:
    """Explicit mechanical wait barriers park the model-owned goal loop."""

    @staticmethod
    def _spawn_sleeper():
        import subprocess, sys
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    def test_wait_pid_parks_loop(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="jw-pid", default_max_turns=10)
            mgr.set("ship the PR")
            mgr.wait_on(proc.pid, "CI watcher still running")
            authority = _begin_goal_turn(mgr, "waiting-turn-1")
            decision = mgr.evaluate_after_turn(
                "Pushed the PR, watching CI.", **authority
            )
            assert decision["verdict"] == "waiting"
            assert decision["should_continue"] is False
            assert decision["continuation_prompt"] is None
            assert mgr.state.waiting_on_pid == proc.pid
            assert mgr.is_waiting() is True

            # The next turn remains parked without consuming model outcome.
            authority = _begin_goal_turn(mgr, "waiting-turn-2")
            d2 = mgr.evaluate_after_turn("still going", **authority)
            assert d2["verdict"] == "waiting"
            assert d2["should_continue"] is False
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_wait_seconds_parks_loop(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-secs", default_max_turns=10)
        mgr.set("retry after backoff")
        mgr.wait_for_seconds(120, "rate limited")
        authority = _begin_goal_turn(mgr)
        decision = mgr.evaluate_after_turn("Hit a 429, backing off.", **authority)
        assert decision["verdict"] == "waiting"
        assert decision["should_continue"] is False
        assert mgr.state.waiting_until > 0
        assert mgr.state.waiting_on_pid is None
        assert mgr.is_waiting() is True

    def test_time_barrier_clears_after_deadline(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-deadline")
        mgr.set("g")
        mgr.wait_for_seconds(120, reason="backoff")
        assert mgr.is_waiting() is True
        # Force the deadline into the past → barrier auto-clears.
        mgr.state.waiting_until = time.time() - 1
        assert mgr.is_waiting() is False
        assert mgr.state.waiting_until == 0.0

    def test_background_metadata_cannot_override_model_outcome(self, hermes_home):
        """A running process cannot semantically park a model-owned goal."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-cont", default_max_turns=10)
        mgr.set("do work")
        authority = _begin_goal_turn(mgr)
        mgr.record_model_outcome("continue", "more to do", **authority)
        decision = mgr.evaluate_after_turn(
            "made progress",
            **authority,
            background_processes=[{"pid": 999999, "command": "x", "status": "running"}],
        )
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert mgr.state.waiting_on_pid is None


# ──────────────────────────────────────────────────────────────────────
# Session/trigger barrier — wait on a process's OWN trigger, not just exit
# ──────────────────────────────────────────────────────────────────────


class TestSessionTriggerBarrier:
    """The session barrier (wait_on_session) releases when a process's own
    trigger fires — a watch_patterns match mid-run (process may never exit)
    OR exit — not only on PID exit. CI-safe: uses synthetic registry session
    objects, no real child processes."""

    @staticmethod
    def _inject(sid, *, watch_patterns=None, exited=False):
        import time as _t
        from tools.process_registry import process_registry, ProcessSession
        s = ProcessSession(id=sid, command="watcher.sh", task_id="t",
                           session_key="", cwd="/tmp", started_at=_t.time())
        if watch_patterns:
            s.watch_patterns = list(watch_patterns)
        s.exited = exited
        if exited:
            process_registry._finished[sid] = s
        else:
            process_registry._running[sid] = s
        return s, process_registry

    def test_registry_is_session_waiting_running_unmatched(self, hermes_home):
        s, reg = self._inject("proc_t1", watch_patterns=["READY"])
        assert reg.is_session_waiting("proc_t1") is True

    def test_registry_releases_on_watch_match_while_alive(self, hermes_home):
        s, reg = self._inject("proc_t2", watch_patterns=["READY"])
        assert reg.is_session_waiting("proc_t2") is True
        s._watch_hits = 1  # what _check_watch_patterns sets on a match
        # Released even though the process is STILL running (never exited).
        assert s.exited is False
        assert reg.is_session_waiting("proc_t2") is False

    def test_registry_releases_on_exit_plain_session(self, hermes_home):
        s, reg = self._inject("proc_t3")  # no watch pattern
        assert reg.is_session_waiting("proc_t3") is True
        s.exited = True
        assert reg.is_session_waiting("proc_t3") is False

    def test_registry_unknown_session_never_waits(self, hermes_home):
        from tools.process_registry import process_registry
        assert process_registry.is_session_waiting("proc_does_not_exist") is False

    def test_goal_parks_on_session_and_releases_on_trigger(self, hermes_home):
        from hermes_cli.goals import GoalManager

        s, _reg = self._inject("proc_t4", watch_patterns=["BUILD SUCCESSFUL"])
        mgr = GoalManager(session_id="st-goal", default_max_turns=10)
        mgr.set("wait for the build to succeed")
        mgr.wait_on_session("proc_t4", "blocked on build")
        authority = _begin_goal_turn(mgr, "session-wait-1")
        decision = mgr.evaluate_after_turn("Started the build watcher.", **authority)
        assert decision["verdict"] == "waiting"
        assert mgr.state.waiting_on_session == "proc_t4"
        assert mgr.is_waiting() is True

        # Further response prose cannot release or complete the goal.
        authority = _begin_goal_turn(mgr, "session-wait-2")
        d2 = mgr.evaluate_after_turn("still building", **authority)
        assert d2["should_continue"] is False

        # Trigger fires mid-run (process still alive) → barrier releases.
        s._watch_hits = 1
        assert mgr.is_waiting() is False
        assert mgr.state.waiting_on_session is None

        # Loop resumes with the primary model's structured outcome.
        authority = _begin_goal_turn(mgr, "session-wait-3")
        mgr.record_model_outcome("continue", "build done", **authority)
        d3 = mgr.evaluate_after_turn("build succeeded", **authority)
        assert d3["should_continue"] is True

    def test_wait_on_session_validation(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="st-val")
        # No active goal → RuntimeError
        with pytest.raises(RuntimeError):
            mgr.wait_on_session("proc_x")
        mgr.set("g")
        with pytest.raises(ValueError):
            mgr.wait_on_session("")

    def test_old_state_loads_without_session_field(self, hermes_home):
        from hermes_cli.goals import GoalState
        st = GoalState.from_json(json.dumps({
            "goal": "g", "status": "active", "turns_used": 0, "max_turns": 20,
        }))
        assert st.waiting_on_session is None


# ──────────────────────────────────────────────────────────────────────
# Completion contract (Codex-inspired structured goals)
# ──────────────────────────────────────────────────────────────────────


class TestParseContract:
    def test_plain_goal_no_contract(self):
        from hermes_cli.goals import parse_contract

        headline, contract = parse_contract("Migrate auth to JWT")
        assert headline == "Migrate auth to JWT"
        assert contract.is_empty()

    def test_incidental_colon_not_treated_as_field(self):
        from hermes_cli.goals import parse_contract

        # "Fix bug:" — "fix bug" is not a known alias, so the whole line
        # stays the headline and no contract field is populated.
        headline, contract = parse_contract("Fix bug: the parser drops trailing commas")
        assert headline == "Fix bug: the parser drops trailing commas"
        assert contract.is_empty()

    def test_inline_fields_parsed(self):
        from hermes_cli.goals import parse_contract

        text = (
            "Migrate auth to JWT\n"
            "verify: the auth test suite passes\n"
            "constraints: keep the /login response shape unchanged\n"
            "boundaries: only touch services/auth and its tests\n"
            "stop when: a schema change needs product sign-off"
        )
        headline, contract = parse_contract(text)
        assert headline == "Migrate auth to JWT"
        assert contract.verification == "the auth test suite passes"
        assert contract.constraints == "keep the /login response shape unchanged"
        assert contract.boundaries == "only touch services/auth and its tests"
        assert contract.stop_when == "a schema change needs product sign-off"
        assert not contract.is_empty()

    def test_alias_variants(self):
        from hermes_cli.goals import parse_contract

        _, c = parse_contract("Goal\nverified by: tests green\npreserve: public API")
        assert c.verification == "tests green"
        assert c.constraints == "public API"

    def test_multiple_lines_same_field_joined(self):
        from hermes_cli.goals import parse_contract

        _, c = parse_contract("G\nconstraints: a\nconstraints: b")
        assert c.constraints == "a b"


class TestGoalContractSerialization:
    def test_roundtrip_with_contract(self):
        from hermes_cli.goals import GoalState, GoalContract

        state = GoalState(
            goal="ship it",
            contract=GoalContract(
                verification="pytest passes",
                constraints="don't break the API",
            ),
        )
        restored = GoalState.from_json(state.to_json())
        assert restored.goal == "ship it"
        assert restored.contract.verification == "pytest passes"
        assert restored.contract.constraints == "don't break the API"
        assert restored.has_contract()

    def test_old_row_without_contract_loads_clean(self):
        # A state_meta row written before this feature has no "contract" key.
        from hermes_cli.goals import GoalState

        legacy = '{"goal": "old goal", "status": "active", "turns_used": 2}'
        state = GoalState.from_json(legacy)
        assert state.goal == "old goal"
        assert state.turns_used == 2
        assert state.contract.is_empty()
        assert not state.has_contract()

    def test_render_block_omits_empty_fields(self):
        from hermes_cli.goals import GoalContract

        block = GoalContract(outcome="X", verification="Y").render_block()
        assert "Outcome: X" in block
        assert "Verification: Y" in block
        assert "Constraints" not in block


class TestGoalManagerContract:
    def test_set_with_contract(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        mgr = GoalManager(session_id="c-set")
        mgr.set("ship it", contract=GoalContract(verification="tests pass"))
        assert mgr.has_contract()
        assert "contract" in mgr.status_line()

    def test_set_without_contract_no_marker(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="c-none")
        mgr.set("ship it")
        assert not mgr.has_contract()
        assert "contract" not in mgr.status_line()

    def test_continuation_prompt_includes_contract(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        mgr = GoalManager(session_id="c-cont")
        mgr.set("ship it", contract=GoalContract(verification="run pytest"))
        prompt = mgr.next_continuation_prompt()
        assert "Completion contract" in prompt
        assert "run pytest" in prompt
        assert "concrete evidence" in prompt

        kickoff = mgr.next_kickoff_prompt()
        assert "Completion contract" in kickoff
        assert "run pytest" in kickoff
        assert "goal_outcome" in kickoff

    def test_set_contract_after_the_fact(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        mgr = GoalManager(session_id="c-after")
        mgr.set("ship it")
        assert not mgr.has_contract()
        mgr.set_contract(GoalContract(verification="x"))
        assert mgr.has_contract()
        # Survives reload.
        from hermes_cli.goals import GoalManager as GM2
        assert GM2(session_id="c-after").has_contract()

    def test_persistence_roundtrip(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        GoalManager(session_id="c-persist").set(
            "ship it", contract=GoalContract(outcome="O", verification="V")
        )
        reloaded = GoalManager(session_id="c-persist")
        assert reloaded.state.contract.outcome == "O"
        assert reloaded.state.contract.verification == "V"


class TestPrimaryModelContract:
    def test_record_model_contract_persists(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="model-contract")
        mgr.set("ship it")
        authority = _begin_goal_turn(mgr)

        assert mgr.record_model_contract(
            {
                "outcome": "feature shipped",
                "verification": "focused tests pass",
                "constraints": "public API unchanged",
                "boundaries": "goal subsystem",
                "stop_when": "owner-only credential is required",
            },
            **authority,
        )

        reloaded = GoalManager(session_id="model-contract")
        assert reloaded.state.contract.outcome == "feature shipped"
        assert reloaded.state.contract.verification == "focused tests pass"
        assert "focused tests pass" in reloaded.next_continuation_prompt()

    def test_record_model_contract_is_schema_only(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="model-contract-validation")
        mgr.set("ship it")
        authority = _begin_goal_turn(mgr)

        with pytest.raises(ValueError, match="unknown fields"):
            mgr.record_model_contract(
                {"verification": "tests pass", "priority": "high"},
                **authority,
            )
        with pytest.raises(ValueError, match="at least one field"):
            mgr.record_model_contract(
                {
                    "outcome": "",
                    "verification": "",
                    "constraints": "",
                    "boundaries": "",
                    "stop_when": "",
                },
                **authority,
            )

    def test_retired_auxiliary_symbols_are_not_exported(self):
        from hermes_cli import goals

        for name in (
            "judge_goal",
            "draft_contract",
            "JUDGE_SYSTEM_PROMPT",
            "JUDGE_USER_PROMPT_TEMPLATE",
            "DRAFT_CONTRACT_SYSTEM_PROMPT",
        ):
            assert not hasattr(goals, name)
            assert name not in goals.__all__

        assert "PRIMARY_MODEL_DRAFT_PROMPT_TEMPLATE" in goals.__all__


class TestGoalTurnAuthority:
    def test_cached_manager_consumes_fresh_writer_outcome(self, hermes_home):
        from hermes_cli.goals import GoalManager

        cached = GoalManager("authority-cached")
        cached.set("ship it")
        authority = _begin_goal_turn(cached)

        writer = GoalManager("authority-cached")
        assert writer.record_model_outcome(
            "complete", "verified live", **authority
        )

        decision = cached.evaluate_after_turn("done", **authority)
        assert decision["verdict"] == "done"
        assert cached.state.status == "done"

    def test_old_turn_and_generation_writes_are_rejected(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager("authority-late")
        mgr.set("do the work")
        old_authority = _begin_goal_turn(mgr, "old-turn")
        new_authority = _begin_goal_turn(mgr, "new-turn")

        late = GoalManager("authority-late")
        assert not late.record_model_outcome(
            "complete", "late completion", **old_authority
        )
        assert not late.record_model_contract(
            {"verification": "late proof"}, **old_authority
        )
        assert late.record_model_outcome(
            "continue", "current turn", **new_authority
        )

    def test_abandon_revokes_pending_and_late_writes(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager("authority-abandon")
        mgr.set("do the work")
        authority = _begin_goal_turn(mgr)
        assert mgr.record_model_outcome("complete", "premature", **authority)

        assert mgr.abandon_model_turn(**authority)
        assert mgr.state.active_model_turn_id is None
        assert mgr.state.pending_model_outcome is None
        assert not GoalManager("authority-abandon").record_model_outcome(
            "complete", "late", **authority
        )
        stale = mgr.evaluate_after_turn("ignored", **authority)
        assert stale["verdict"] == "stale"
        assert stale["should_continue"] is False

    def test_resume_and_new_goal_rotate_generation(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager("authority-generation")
        first = mgr.set("first")
        first_authority = _begin_goal_turn(mgr, "first-turn")
        mgr.pause()
        resumed = mgr.resume()
        assert resumed.generation_id != first.generation_id
        assert not mgr.record_model_outcome(
            "complete", "late first", **first_authority
        )

        resumed_generation = resumed.generation_id
        replacement = mgr.set("replacement")
        assert replacement.generation_id != resumed_generation

    def test_migration_revokes_parent_turn_on_both_rows(self, hermes_home):
        from hermes_cli.goals import GoalManager, migrate_goal_to_session

        parent = GoalManager("authority-parent")
        parent.set("survive compression")
        authority = _begin_goal_turn(parent, "parent-turn")

        assert migrate_goal_to_session("authority-parent", "authority-child")
        child = GoalManager("authority-child")
        assert child.state.generation_id != authority["goal_generation_id"]
        assert child.state.active_model_turn_id is None
        assert not parent.record_model_outcome("complete", "late", **authority)
        assert not child.record_model_contract(
            {"verification": "late"}, **authority
        )

    def test_concurrent_contract_and_outcome_do_not_lose_fields(self, hermes_home):
        from hermes_cli.goals import GoalManager

        owner = GoalManager("authority-race")
        owner.set("race safely")
        authority = _begin_goal_turn(owner)
        barrier = threading.Barrier(3)
        results = []

        def _write_outcome():
            barrier.wait()
            results.append(
                GoalManager("authority-race").record_model_outcome(
                    "continue", "more work", **authority
                )
            )

        def _write_contract():
            barrier.wait()
            results.append(
                GoalManager("authority-race").record_model_contract(
                    {"verification": "focused tests pass"}, **authority
                )
            )

        threads = [
            threading.Thread(target=_write_outcome),
            threading.Thread(target=_write_contract),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)

        assert len(results) == 2 and all(results)
        durable = GoalManager("authority-race").state
        assert durable.pending_model_outcome == "continue"
        assert durable.contract.verification == "focused tests pass"
