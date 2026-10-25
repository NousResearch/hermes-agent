"""Tests for hermes_cli/session_standing_state.py — retiring standing state at a session boundary.

A ``/goal`` and a ``/loop`` are keyed by the session that owns them, so a row
outlives its session only as a frozen artifact that still reports "active".
``/new`` / ``/reset`` end the conversation, so its standing state must go with it.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    # loops.py shares the goals module's SessionDB cache, so clearing it re-resolves HERMES_HOME.
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class TestRetireStandingState:
    def test_active_goal_is_cleared(self, hermes_home):
        from hermes_cli.goals import GoalManager, load_goal
        from hermes_cli.session_standing_state import retire_standing_state

        GoalManager(session_id="s1").set("ship the thing")
        assert load_goal("s1").status == "active"

        goal_retired, loop_retired = retire_standing_state("s1", reason="new_session")

        assert goal_retired is True
        assert loop_retired is False
        # The row is kept for audit, but it no longer reports live work.
        assert load_goal("s1").status == "cleared"

    def test_paused_goal_is_cleared(self, hermes_home):
        from hermes_cli.goals import GoalManager, load_goal
        from hermes_cli.session_standing_state import retire_standing_state

        mgr = GoalManager(session_id="s2")
        mgr.set("ship the thing")
        mgr.pause(reason="user-paused")
        assert load_goal("s2").status == "paused"

        goal_retired, _ = retire_standing_state("s2", reason="new_session")

        assert goal_retired is True
        assert load_goal("s2").status == "cleared"

    def test_active_loop_is_cleared(self, hermes_home):
        from hermes_cli.loops import LoopManager, load_loop
        from hermes_cli.session_standing_state import retire_standing_state

        LoopManager(session_id="s3").set("check the build", interval_seconds=60)
        assert load_loop("s3").status == "active"

        goal_retired, loop_retired = retire_standing_state("s3", reason="new_session")

        assert loop_retired is True
        assert goal_retired is False
        assert load_loop("s3").status == "cleared"

    def test_both_goal_and_loop_are_cleared(self, hermes_home):
        from hermes_cli.goals import GoalManager, load_goal
        from hermes_cli.loops import LoopManager, load_loop
        from hermes_cli.session_standing_state import retire_standing_state

        GoalManager(session_id="s4").set("ship the thing")
        LoopManager(session_id="s4").set("check the build", interval_seconds=60)

        goal_retired, loop_retired = retire_standing_state("s4", reason="new_session")

        assert (goal_retired, loop_retired) == (True, True)
        assert load_goal("s4").status == "cleared"
        assert load_loop("s4").status == "cleared"

    def test_terminal_rows_are_left_alone(self, hermes_home):
        """A goal the judge already ruled done keeps its evidence, not a rewrite."""
        from hermes_cli.goals import GoalState, load_goal, save_goal
        from hermes_cli.session_standing_state import retire_standing_state

        save_goal("s5", GoalState(goal="already finished", status="done"))

        goal_retired, _ = retire_standing_state("s5", reason="new_session")

        assert goal_retired is False
        assert load_goal("s5").status == "done"

    def test_missing_rows_are_a_noop(self, hermes_home):
        from hermes_cli.session_standing_state import retire_standing_state

        assert retire_standing_state("never-existed", reason="new_session") == (False, False)

    def test_empty_session_id_is_a_noop(self, hermes_home):
        from hermes_cli.session_standing_state import retire_standing_state

        assert retire_standing_state("", reason="new_session") == (False, False)

    def test_retiring_one_session_leaves_another_alone(self, hermes_home):
        """Rows are keyed per session: a sibling conversation keeps its goal."""
        from hermes_cli.goals import GoalManager, load_goal
        from hermes_cli.session_standing_state import retire_standing_state

        GoalManager(session_id="s6").set("this one ends")
        GoalManager(session_id="s7").set("this one continues")

        retire_standing_state("s6", reason="new_session")

        assert load_goal("s6").status == "cleared"
        assert load_goal("s7").status == "active"

    def test_a_failure_does_not_raise(self, hermes_home, monkeypatch):
        """Retiring state must never keep the session boundary from completing."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager
        from hermes_cli.session_standing_state import retire_standing_state

        GoalManager(session_id="s8").set("ship the thing")

        def _boom(*_args, **_kwargs):
            raise RuntimeError("db is gone")

        monkeypatch.setattr(goals, "clear_goal", _boom)

        goal_retired, loop_retired = retire_standing_state("s8", reason="new_session")

        assert (goal_retired, loop_retired) == (False, False)

    def test_retired_summary_names_what_was_retired(self, hermes_home):
        from hermes_cli.session_standing_state import retired_summary

        assert retired_summary(True, False) == "standing /goal cleared for the previous session."
        assert retired_summary(False, True) == "standing /loop cleared for the previous session."
        assert retired_summary(True, True) == (
            "standing /goal and standing /loop cleared for the previous session."
        )
        assert retired_summary(False, False) is None
