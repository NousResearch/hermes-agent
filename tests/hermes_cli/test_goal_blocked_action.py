"""`goals.blocked_action` controls what a judge BLOCKED verdict does.

Default ``pause`` keeps the #100954 contract (blocked pauses the goal). Opt-in ``continue`` keeps
the goal active for unattended runs and feeds the judge's reason into the continuation prompt,
still bounded by the turn budget. The setting is read from config.yaml at evaluation time.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

BLOCKED = ("blocked", "the agent asked which database to use", False, None, False)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _write_config(home, blocked_action=None):
    body = "goals:\n  max_turns: 20\n"
    if blocked_action is not None:
        body += f"  blocked_action: {blocked_action}\n"
    (home / "config.yaml").write_text(body, encoding="utf-8")


def _evaluate_blocked(mgr):
    with patch("hermes_cli.goals.judge_goal", return_value=BLOCKED):
        return mgr.evaluate_after_turn("Which database should I use?")


def test_default_blocked_verdict_pauses(hermes_home):
    from hermes_cli.goals import GoalManager

    _write_config(hermes_home)
    mgr = GoalManager(session_id="ba-default")
    mgr.set("build the service")
    decision = _evaluate_blocked(mgr)

    assert decision["status"] == "paused"
    assert decision["should_continue"] is False
    assert mgr.state.status == "paused"


def test_continue_keeps_goal_active_with_reason_in_prompt(hermes_home):
    from hermes_cli.goals import GoalManager, load_goal

    _write_config(hermes_home, "continue")
    mgr = GoalManager(session_id="ba-continue")
    mgr.set("build the service")
    decision = _evaluate_blocked(mgr)

    assert decision["status"] == "active"
    assert decision["should_continue"] is True
    assert decision["verdict"] == "blocked"
    assert "the agent asked which database to use" in decision["continuation_prompt"]
    persisted = load_goal("ba-continue")
    assert persisted.status == "active"
    assert persisted.last_verdict == "blocked"


def test_continue_still_pauses_when_turn_budget_exhausted(hermes_home):
    from hermes_cli.goals import GoalManager

    _write_config(hermes_home, "continue")
    mgr = GoalManager(session_id="ba-budget")
    mgr.set("build the service", max_turns=2)
    first = _evaluate_blocked(mgr)
    second = _evaluate_blocked(mgr)

    assert first["should_continue"] is True
    assert second["status"] == "paused"
    assert second["should_continue"] is False
    assert "turn budget" in mgr.state.paused_reason


def test_unknown_value_falls_back_to_pause(hermes_home):
    from hermes_cli.goals import GoalManager

    _write_config(hermes_home, "ignore")
    mgr = GoalManager(session_id="ba-garbage")
    mgr.set("build the service")
    decision = _evaluate_blocked(mgr)

    assert decision["status"] == "paused"
