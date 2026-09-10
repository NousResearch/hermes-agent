"""``/goal continue`` — pick a crash-interrupted goal back up.

``resume`` is the budget-exhausted verb: it un-pauses and resets ``turns_used``.
A goal whose turn died with the backend was never paused and never finished a
turn, so continuing it must spend nothing and change no status — otherwise a
desktop restart silently refunds the budget the goal already burned.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes stay in the temp dir."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _manager(session_id, goal="ship the crash-resume feature"):
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id=session_id)
    mgr.set(goal)
    return mgr


def test_continue_spends_no_budget_and_leaves_status_active(hermes_home):
    from hermes_cli.goals import load_goal

    mgr = _manager("continue-budget")
    mgr.state.turns_used = 7
    mgr._save()

    prompt = mgr.continue_after_interruption()

    assert prompt
    assert mgr.state.goal in prompt
    # The reload proves the durable row, not just the in-memory object.
    persisted = load_goal("continue-budget")
    assert persisted.turns_used == 7
    assert persisted.status == "active"


def test_continue_differs_from_resume_exactly_by_the_budget_reset(hermes_home):
    """Both verbs yield the same continuation text; only ``resume`` refunds turns."""
    continued = _manager("continue-vs-resume-a")
    continued.state.turns_used = 5
    continued._save()
    resumed = _manager("continue-vs-resume-b")
    resumed.state.turns_used = 5
    resumed._save()

    continue_prompt = continued.continue_after_interruption()
    resumed.resume()

    assert continue_prompt == resumed.next_continuation_prompt()
    assert continued.state.turns_used == 5
    assert resumed.state.turns_used == 0


def test_paused_and_cleared_goals_have_nothing_to_continue(hermes_home):
    from hermes_cli.goals import load_goal

    paused = _manager("continue-paused")
    paused.pause(reason="user-paused")

    assert paused.continue_after_interruption() is None
    # Refusing must not resurrect the goal.
    assert load_goal("continue-paused").status == "paused"

    cleared = _manager("continue-cleared")
    cleared.clear()
    assert cleared.continue_after_interruption() is None


def test_continue_retires_the_interruption_marker_it_answers(hermes_home):
    from hermes_cli.goals import load_goal

    mgr = _manager("continue-clears-flag")
    mgr.mark_interrupted(1234.0)
    assert load_goal("continue-clears-flag").interrupted_at == 1234.0

    assert mgr.continue_after_interruption()

    assert load_goal("continue-clears-flag").interrupted_at is None


@pytest.mark.parametrize("verb", ["resume", "pause", "clear"])
def test_every_lifecycle_verb_retires_the_interruption(hermes_home, verb):
    """An interruption describes one dead turn; any deliberate transition supersedes it."""
    from hermes_cli.goals import load_goal

    mgr = _manager(f"interrupt-{verb}")
    mgr.mark_interrupted(999.0)

    getattr(mgr, verb)()

    state = load_goal(f"interrupt-{verb}")
    assert state.interrupted_at is None


def test_interruption_survives_a_persistence_roundtrip_and_old_rows_load(hermes_home):
    from hermes_cli.goals import GoalState, load_goal

    mgr = _manager("interrupt-roundtrip")
    mgr.mark_interrupted(4242.5)

    assert load_goal("interrupt-roundtrip").interrupted_at == 4242.5
    # A row written before the field existed still loads.
    legacy = GoalState.from_json('{"goal": "old row", "status": "active"}')
    assert legacy.interrupted_at is None


def test_mark_interrupted_only_applies_to_an_active_goal(hermes_home):
    from hermes_cli.goals import load_goal

    mgr = _manager("interrupt-paused")
    mgr.pause()

    assert mgr.mark_interrupted(1.0) is None
    assert load_goal("interrupt-paused").interrupted_at is None


def test_goal_continue_command_returns_the_continuation_prompt(hermes_home):
    from hermes_cli.goal_command import dispatch_goal_command

    mgr = _manager("continue-command")
    mgr.state.turns_used = 4
    mgr._save()

    result = dispatch_goal_command(mgr, "continue", authorize_gate=lambda: None)

    assert not result.error
    assert result.prompt == mgr.next_continuation_prompt()
    assert mgr.state.goal in result.output
    assert mgr.state.turns_used == 4


def test_goal_continue_command_on_a_paused_goal_offers_no_prompt(hermes_home):
    from hermes_cli.goal_command import dispatch_goal_command

    mgr = _manager("continue-command-paused")
    mgr.pause()

    result = dispatch_goal_command(mgr, "continue", authorize_gate=lambda: None)

    assert result.prompt is None
    assert not result.error
    assert "continue" in result.output.lower()
