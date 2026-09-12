"""Opt-in automatic Goal start for ordinary CLI and gateway messages."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hermes_cli import goals
from hermes_cli.goal_command import (
    auto_start_goal,
    goal_auto_start_enabled,
    goal_objective_text,
    is_goal_candidate,
)


def test_goal_helpers_normalize_text_and_gate_config():
    assert goal_objective_text("  build it  ") == "build it"
    assert goal_objective_text([{"type": "text", "text": "first"}, {"type": "image_url"}, {"text": "second"}]) == (
        "first\nsecond"
    )
    assert goal_auto_start_enabled({"goals": {"auto_start": True}})
    assert goal_auto_start_enabled({"goals": {"auto_start": "yes"}})
    assert not goal_auto_start_enabled({"goals": {"auto_start": False}})
    assert not goal_auto_start_enabled({})
    assert is_goal_candidate("Implement the config option and add tests")
    assert is_goal_candidate("Can you implement this option?")
    assert is_goal_candidate("この機能を実装してテストも追加して")
    assert not is_goal_candidate("What is a persistent Goal?")
    assert not is_goal_candidate("Can you tell me how to fix this?")
    assert not is_goal_candidate("Should I implement this?")
    assert not is_goal_candidate("/goal Implement the feature")
    assert not is_goal_candidate("!goal draft Implement the feature")
    assert not is_goal_candidate("どうやって実装する？")
    assert not is_goal_candidate("できる？")
    assert not is_goal_candidate("Thanks")


def test_auto_start_goal_drafts_and_persists_without_a_kickoff(monkeypatch):
    goals._DB_CACHE.clear()
    drafted = []

    def draft(objective):
        drafted.append(objective)
        return goals.GoalContract(verification="tests pass")

    monkeypatch.setattr(goals, "draft_contract", draft)
    manager = goals.GoalManager("auto-start")

    result = auto_start_goal(manager, "  make the tests pass  ")

    assert result is not None
    assert result.kickoff is True
    assert result.prompt == "make the tests pass"
    assert drafted == ["make the tests pass"]
    state = goals.load_goal("auto-start")
    assert state is not None
    assert state.goal == "make the tests pass"
    assert state.has_contract()


def test_auto_start_goal_does_not_replace_an_existing_goal(monkeypatch):
    goals._DB_CACHE.clear()
    monkeypatch.setattr(goals, "draft_contract", lambda objective: (_ for _ in ()).throw(AssertionError()))
    manager = goals.GoalManager("existing-goal")
    manager.set("keep this goal")

    assert auto_start_goal(manager, "new message") is None
    assert auto_start_goal(goals.GoalManager("existing-goal"), "another message") is None
    assert goals.load_goal("existing-goal").goal == "keep this goal"


def test_cli_auto_start_is_opt_in_and_fail_open(monkeypatch):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    goals._DB_CACHE.clear()
    manager = goals.GoalManager("cli-auto-start")
    cli = object.__new__(CLILoopsMixin)
    cli.session_id = "cli-auto-start"
    cli._get_goal_manager = lambda: manager

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"goals": {"auto_start": False}})
    assert not cli._maybe_auto_start_goal("Implement the first message")
    assert goals.load_goal("cli-auto-start") is None

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"goals": {"auto_start": True}})
    monkeypatch.setattr(goals, "draft_contract", lambda objective: None)
    assert cli._maybe_auto_start_goal("Implement the first message")
    assert goals.load_goal("cli-auto-start").goal == "Implement the first message"

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: (_ for _ in ()).throw(RuntimeError("config unavailable")))
    assert cli._maybe_auto_start_goal("second message") is False



def test_gateway_auto_start_uses_normal_events_and_skips_internal_events(monkeypatch):
    from gateway.run_goals import GatewayGoalsMixin

    goals._DB_CACHE.clear()
    monkeypatch.setattr(goals, "draft_contract", lambda objective: None)

    class Runner(GatewayGoalsMixin):
        config = {"goals": {"auto_start": True, "max_turns": 3}}

        async def _run_in_executor_with_context(self, fn):
            return fn()

    runner = object.__new__(Runner)
    event = SimpleNamespace(internal=False)
    assert asyncio.run(runner._maybe_auto_start_goal(event, "gateway-auto-start", "Implement the new feature"))
    state = goals.load_goal("gateway-auto-start")
    assert state.goal == "Implement the new feature"
    assert state.max_turns == 3

    internal_event = SimpleNamespace(internal=True)
    assert not asyncio.run(runner._maybe_auto_start_goal(internal_event, "internal", "Implement the internal change"))
    assert goals.load_goal("internal") is None


def test_gateway_auto_start_failure_does_not_block_turn():
    from gateway.run_goals import GatewayGoalsMixin

    class Runner(GatewayGoalsMixin):
        config = {"goals": {"auto_start": True}}

        async def _run_in_executor_with_context(self, fn):
            raise RuntimeError("executor unavailable")

    runner = object.__new__(Runner)
    event = SimpleNamespace(internal=False)
    assert not asyncio.run(runner._maybe_auto_start_goal(event, "gateway-failure", "Implement the change"))
    assert goals.load_goal("gateway-failure") is None
