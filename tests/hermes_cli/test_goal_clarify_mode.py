"""goals.clarify: skip answers clarify at once only while this session's goal is active."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.mark.parametrize("mode", ["skip", "block"])
def test_clarify_skips_only_for_an_active_goal_in_skip_mode(hermes_home, mode):
    from agent.inline_tool_executors import _clarify_callback
    from hermes_cli.goals import GoalManager
    from tools.clarify_tool import clarify_tool

    (hermes_home / "config.yaml").write_text(f"goals:\n  clarify: {mode}\n")
    asked = []

    def surface(questions):
        asked.append(questions)
        return {"answers": {"q0": "beta"}, "outcome": "submitted"}

    agent = SimpleNamespace(clarify_callback=surface, session_id="sess-goal")
    questions = [{"question": "alpha or beta?", "choices": ["alpha", "beta"]}]
    manager = GoalManager("sess-goal")
    manager.set("ship it")

    result = json.loads(clarify_tool(questions, callback=_clarify_callback(agent)))
    if mode == "skip":
        assert asked == [] and result["outcome"] == "undelivered"
        assert result["responses"][0]["user_response"] is None
    else:
        assert len(asked) == 1 and result["outcome"] == "submitted"

    # A paused goal (or no goal) always reaches the user, whatever the mode.
    manager.pause()
    asked.clear()
    result = json.loads(clarify_tool(questions, callback=_clarify_callback(agent)))
    assert len(asked) == 1 and result["responses"][0]["user_response"] == "beta"
