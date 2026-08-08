"""Tests for orphan goal recovery — resuming active goals whose owning
process died (issue #81109).

A standing /goal only runs inside a live process; when the owner dies
(CLI terminal closed, desktop chat PTY reaped, gateway stopped) the goal
stays "active" in state_meta but nothing feeds continuation turns. The
recovery sweep (hermes_cli/goal_recovery.py) detects these goals and
hands them back to a live surface (gateway / hermes serve) to resume.

Covered here:
- orphan detection (dead owner pid → claimable; live owner pid → skipped)
- sweep claims goals (resume-on-boot data path)
- no double-fire for goals whose owning process is still alive
- paused/cleared goals are excluded
- /goal status distinguishes active-running from active-orphaned
- claim lock prevents a sibling sweep from re-claiming within the window
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _set_goal(home, session_id: str, *, status: str = "active", owner_pid=None):
    """Write a goal row directly (bypassing owner stamping) for sweep tests."""
    from hermes_state import SessionDB
    from hermes_cli.goals import GoalState, save_goal

    db = SessionDB(db_path=Path(home) / "state.db")
    state = GoalState(
        goal=f"goal for {session_id}",
        status=status,
        turns_used=0,
        max_turns=5,
        created_at=time.time() - 600,
        last_turn_at=time.time() - 600,
    )
    if owner_pid is not None:
        state.owner_pid = owner_pid
        state.last_owner_seen_at = time.time() - 60
    else:
        # No owner pid + stale owner stamp → past the silence window.
        state.last_owner_seen_at = time.time() - 3600
    save_goal(session_id, state, db=db)
    db.close()
    return state


class TestOrphanDetection:

    def test_dead_owner_pid_is_orphaned(self, hermes_home):
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        # A pid that can never be alive (max sys.pid_max on Linux is ~4M).
        _set_goal(hermes_home, "orphan-s1", owner_pid=4_000_000)
        sweeper = GoalRecoverySweeper([str(hermes_home)])
        claims = sweeper.sweep()
        assert len(claims) == 1
        assert claims[0][1] == "orphan-s1"

    def test_live_owner_pid_is_not_claimed(self, hermes_home):
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        # Our own pid is definitely alive — never claim.
        _set_goal(hermes_home, "live-s1", owner_pid=os.getpid())
        sweeper = GoalRecoverySweeper([str(hermes_home)])
        assert sweeper.sweep() == []

    def test_no_owner_pid_past_silence_window_is_orphaned(self, hermes_home):
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        _set_goal(hermes_home, "noowner-s1")  # no owner_pid, stale stamp
        sweeper = GoalRecoverySweeper([str(hermes_home)])
        assert len(sweeper.sweep()) == 1

    def test_surface_owns_goal_skips_it(self, hermes_home):
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        _set_goal(hermes_home, "owned-s1", owner_pid=4_000_000)
        sweeper = GoalRecoverySweeper(
            [str(hermes_home)],
            owns_goal=lambda sid: sid == "owned-s1",
        )
        assert sweeper.sweep() == []


class TestPausedAndClearedExcluded:

    def test_paused_goal_not_claimed(self, hermes_home):
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        _set_goal(hermes_home, "paused-s1", status="paused", owner_pid=4_000_000)
        _set_goal(hermes_home, "cleared-s1", status="cleared", owner_pid=4_000_000)
        _set_goal(hermes_home, "done-s1", status="done", owner_pid=4_000_000)
        sweeper = GoalRecoverySweeper([str(hermes_home)])
        assert sweeper.sweep() == []

    def test_mixed_goals_only_orphaned_active_claimed(self, hermes_home):
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        _set_goal(hermes_home, "mix-paused", status="paused", owner_pid=4_000_000)
        _set_goal(hermes_home, "mix-live", owner_pid=os.getpid())
        _set_goal(hermes_home, "mix-orphan", owner_pid=4_000_000)
        sweeper = GoalRecoverySweeper([str(hermes_home)])
        claims = sweeper.sweep()
        assert [c[1] for c in claims] == ["mix-orphan"]


class TestSweepClaims:

    def test_claim_carries_goal_state(self, hermes_home):
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        _set_goal(hermes_home, "claim-s1", owner_pid=4_000_000)
        sweeper = GoalRecoverySweeper([str(hermes_home)])
        claims = sweeper.sweep()
        assert len(claims) == 1
        home, session_id, state = claims[0]
        assert session_id == "claim-s1"
        assert getattr(state, "goal", "") == "goal for claim-s1"
        assert getattr(state, "orphaned", False) is True

    def test_clear_claim_flips_back_to_running(self, hermes_home):
        from hermes_cli.goal_recovery import GoalRecoverySweeper
        from hermes_cli.goals import load_goal

        _set_goal(hermes_home, "clear-s1", owner_pid=4_000_000)
        sweeper = GoalRecoverySweeper([str(hermes_home)])
        claims = sweeper.sweep()
        assert len(claims) == 1
        sweeper.clear_claim(claims[0][0], claims[0][1])
        assert load_goal("clear-s1").orphaned is False

    def test_uncleared_claim_not_reclaimed_within_window(self, hermes_home):
        """A claimed goal whose flag was never cleared (drive crashed) is
        not re-claimed within the recent-claim window — the next sweep after
        the window heals it."""
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        _set_goal(hermes_home, "uncleared-s1", owner_pid=4_000_000)
        sweeper_a = GoalRecoverySweeper([str(hermes_home)])
        sweeper_b = GoalRecoverySweeper([str(hermes_home)])
        assert len(sweeper_a.sweep()) == 1
        # Fresh claim → sibling skips it.
        assert sweeper_b.sweep() == []


class TestStatusLineDistinguishes:

    def test_orphaned_flag_reflected_in_status(self, hermes_home):
        from hermes_cli.goals import GoalManager, mark_goal_orphaned

        mgr = GoalManager(session_id="status-s1")
        mgr.set("ship it")
        assert "active-running" in mgr.status_line()

        assert mark_goal_orphaned("status-s1") is True
        # A fresh manager reloads from the DB and sees the flag.
        fresh = GoalManager(session_id="status-s1")
        line = fresh.status_line()
        assert "active-orphaned" in line

        # A live turn clears the flag back to running.
        from hermes_cli.goals import clear_goal_orphaned

        assert clear_goal_orphaned("status-s1") is True
        fresh2 = GoalManager(session_id="status-s1")
        assert "active-running" in fresh2.status_line()

    def test_orphaned_flag_cleared_by_evaluate_after_turn(self, hermes_home):
        """The post-turn judge path (the loop actually running) restamps
        ownership, so /goal status flips back to active-running."""
        from hermes_cli.goals import GoalManager, mark_goal_orphaned
        from hermes_cli import goals as goals_mod

        with patch("agent.auxiliary_client.call_llm") as call_llm:
            call_llm.return_value = type(
                "R", (), {"choices": [type("C", (), {"message": type(
                    "M", (), {"content": '{"done": false, "reason": "keep going"}'}
                )()})()]}
            )()
            mgr = GoalManager(session_id="eval-s1")
            mgr.set("keep working")
            mark_goal_orphaned("eval-s1")
            decision = mgr.evaluate_after_turn("some response")
        assert decision["should_continue"] is True
        assert "active-running" in mgr.status_line()
        assert goals_mod.load_goal("eval-s1").orphaned is False


class TestEnumeration:

    def test_enumerate_active_goals_scans_meta(self, hermes_home):
        from hermes_cli.goals import enumerate_active_goals

        _set_goal(hermes_home, "enum-1", owner_pid=os.getpid())
        _set_goal(hermes_home, "enum-2", owner_pid=os.getpid())
        rows = enumerate_active_goals()
        assert {sid for sid, _state in rows} == {"enum-1", "enum-2"}

    def test_enumerate_skips_cleared(self, hermes_home):
        from hermes_cli.goals import enumerate_active_goals

        _set_goal(hermes_home, "enum-c", status="cleared")
        rows = enumerate_active_goals()
        assert rows == []


class TestClaimLock:

    def test_cross_process_lock_serializes_claims(self, hermes_home):
        """Two sweeper instances share the same lock file; the second sees
        the flag already set and skips."""
        from hermes_cli.goal_recovery import GoalRecoverySweeper

        _set_goal(hermes_home, "lock-s1", owner_pid=4_000_000)
        sweeper = GoalRecoverySweeper([str(hermes_home)])
        assert len(sweeper.sweep()) == 1
        # A sibling instance (same home) respects the persisted claim.
        sibling = GoalRecoverySweeper([str(hermes_home)])
        assert sibling.sweep() == []
