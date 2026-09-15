"""Deterministic Plugin completion-gate coverage for opted-in Goals."""

from __future__ import annotations

import time

import pytest

from hermes_cli import goals, plugins
from hermes_cli.plugins_manifest import PluginManifest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    plugins._reset_plugin_managers_for_tests()
    yield
    plugins._reset_plugin_managers_for_tests()


def _done_judge(*_args, **_kwargs):
    return "done", "model says complete", False, None, False


def _register_gate(name, callback):
    manager = plugins.get_plugin_manager()
    manager._discovered = True
    ctx = plugins.PluginContext(PluginManifest(name="test-gate"), manager)
    ctx.register_goal_completion_gate(name, callback)
    return manager


def test_unopted_goal_keeps_official_done_behavior(monkeypatch):
    monkeypatch.setattr(goals, "judge_goal", _done_judge)
    manager = goals.GoalManager("plain")
    manager.set("plain goal")

    decision = manager.evaluate_after_turn("final prose")

    assert decision["status"] == "done"
    assert manager.state.status == "done"


def test_opted_goal_allows_done_only_when_registered_gate_allows(monkeypatch):
    monkeypatch.setattr(goals, "judge_goal", _done_judge)
    seen = {}

    def gate(**payload):
        seen.update(payload)
        return {"action": "allow", "reason": "fresh evidence"}

    _register_gate("root-repair", gate)
    manager = goals.GoalManager("governed")
    manager.set("governed goal", completion_gate="root-repair")

    decision = manager.evaluate_after_turn("unsupported final prose", turn_id="turn-7")

    assert decision["status"] == "done"
    assert seen["session_id"] == "governed"
    assert seen["turn_id"] == "turn-7"
    assert seen["candidate_state"] == "done"
    assert seen["goal"]["objective"] == "governed goal"
    assert seen["generation"] == manager.state.created_at


def test_gate_continue_prevents_done_and_requests_more_work(monkeypatch):
    monkeypatch.setattr(goals, "judge_goal", _done_judge)
    _register_gate(
        "root-repair",
        lambda **_: {"action": "continue", "reason": "verification is stale"},
    )
    manager = goals.GoalManager("stale")
    manager.set("needs verification", completion_gate="root-repair")

    decision = manager.evaluate_after_turn("I am done")

    assert decision["status"] == "active"
    assert decision["should_continue"] is True
    assert decision["verdict"] == "continue"
    assert manager.state.status == "active"
    assert manager.state.last_reason == "verification is stale"


def test_gate_blocked_transitions_goal_to_blocked_pause(monkeypatch):
    monkeypatch.setattr(goals, "judge_goal", _done_judge)
    _register_gate(
        "root-repair",
        lambda **_: {"action": "blocked", "reason": "irreversible constraint violation"},
    )
    manager = goals.GoalManager("blocked")
    manager.set("unsafe goal", completion_gate="root-repair")

    decision = manager.evaluate_after_turn("done")

    assert decision["status"] == "paused"
    assert decision["verdict"] == "blocked"
    assert manager.state.status == "paused"
    assert "completion gate blocked" in manager.state.paused_reason


def test_missing_opted_gate_fails_closed(monkeypatch):
    monkeypatch.setattr(goals, "judge_goal", _done_judge)
    manager = plugins.get_plugin_manager()
    manager._discovered = True
    goal = goals.GoalManager("missing")
    goal.set("governed goal", completion_gate="root-repair")

    decision = goal.evaluate_after_turn("done")

    assert decision["status"] == "paused"
    assert decision["verdict"] == "blocked"
    assert "unavailable" in decision["reason"]


@pytest.mark.parametrize(
    "callback, fragment",
    [
        (lambda **_: {"action": "surprise"}, "invalid action"),
        (lambda **_: (_ for _ in ()).throw(RuntimeError("boom")), "failed"),
    ],
)
def test_invalid_or_crashed_gate_fails_closed(callback, fragment):
    _register_gate("root-repair", callback)

    result = plugins.evaluate_goal_completion_gate(
        "root-repair", session_id="s", generation=1.0, candidate_state="done",
    )

    assert result["action"] == "blocked"
    assert fragment in result["reason"]


def test_timed_out_gate_fails_closed(monkeypatch):
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.01)

    def slow_gate(**_):
        time.sleep(0.1)
        return {"action": "allow", "reason": "late"}

    _register_gate("root-repair", slow_gate)
    result = plugins.evaluate_goal_completion_gate(
        "root-repair", session_id="s", generation=1.0, candidate_state="done",
    )

    assert result["action"] == "blocked"
    assert "timed out" in result["reason"]


def test_duplicate_gate_registration_is_rejected():
    manager = plugins.get_plugin_manager()
    manager._discovered = True
    first = plugins.PluginContext(PluginManifest(name="one"), manager)
    second = plugins.PluginContext(PluginManifest(name="two"), manager)
    first.register_goal_completion_gate("root-repair", lambda **_: {"action": "allow"})

    with pytest.raises(ValueError, match="already registered"):
        second.register_goal_completion_gate("root-repair", lambda **_: {"action": "allow"})


def test_unload_removes_completion_gate_registration():
    manager = _register_gate("root-repair", lambda **_: {"action": "allow"})
    assert manager.has_goal_completion_gate("root-repair") is True

    manager.unload()

    assert manager.has_goal_completion_gate("root-repair") is False
