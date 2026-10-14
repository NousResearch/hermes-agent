"""Recovery must preserve authority, goal identity, completion checks and budget."""
import time
from copy import deepcopy

import pytest

from hermes_cli import goals
from hermes_cli.goal_command import dispatch_goal_command
from hermes_cli.plugins import PluginManager


def proposal(**changes):
    option = dict(id="inspect", route="gather_evidence", plan="Read the saved checks and inspect free disk space.",
                  verification="Record the measured free bytes and identify the failing check.",
                  effects=["read_only"], risk="low", authorization="authorized", reversibility="reversible",
                  progress="useful", evidence="sufficient")
    option.update(changes)
    return dict(selected="inspect", automatic=True, options=[option], classifications={"cause": "resource"})


@pytest.fixture
def manager(monkeypatch):
    pm = PluginManager()
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: pm)
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **kw: ("blocked", "disk verification blocked", False, None, False))
    mgr = goals.GoalManager("recovery-session", default_max_turns=8)
    mgr.set("Implement the product; verify in a browser", contract=goals.GoalContract(verification="browser acceptance"))
    return mgr, pm


def install(pm, data):
    pm._hooks["goal_blocked"] = [lambda **kw: deepcopy(data)]


def test_authorized_read_only_recovery_continues_and_survives_reload(manager):
    mgr, pm = manager
    install(pm, proposal())
    result = mgr.evaluate_after_turn("Browser needs more disk space.")
    assert result["should_continue"] is True
    assert "measured free bytes" in result["continuation_prompt"]
    assert mgr.state.turns_used == 1
    loaded = goals.GoalManager(mgr.session_id)
    assert loaded.state.recovery["selected"] == "inspect"
    assert loaded.state.contract.verification == "browser acceptance"


def test_no_plugin_preserves_blocked_verdict(manager):
    mgr, _ = manager
    result = mgr.evaluate_after_turn("Need input")
    assert result["status"] == "paused"
    assert result["verdict"] == "blocked"


@pytest.mark.parametrize("changes", [
    {"effects": ["delete"]}, {"effects": ["spend"]}, {"effects": ["credentials"]},
    {"effects": ["publish"]}, {"effects": ["security"]}, {"effects": ["unknown"]},
    {"effects": []}, {"risk": "high"}, {"authorization": "approval_required"},
    {"reversibility": "irreversible"}, {"progress": "bookkeeping"}, {"evidence": "insufficient"},
    {"route": "scope_change"},
])
def test_unsafe_or_unproven_paths_are_not_automatic(manager, changes):
    mgr, pm = manager
    install(pm, proposal(**changes))
    result = mgr.evaluate_after_turn("Need input")
    assert not result["should_continue"]
    assert result["status"] == "paused"
    assert "/goal choose" in result["message"]


def test_budget_is_not_reset_or_exceeded_by_recovery(manager):
    mgr, pm = manager
    install(pm, proposal())
    mgr.state.turns_used = mgr.state.max_turns - 1
    result = mgr.evaluate_after_turn("Need input")
    assert not result["should_continue"]
    assert mgr.state.turns_used == mgr.state.max_turns


def test_repeated_automatic_recovery_is_bounded(manager):
    mgr, pm = manager
    install(pm, proposal())
    results = [mgr.evaluate_after_turn("same unresolved blocker") for _ in range(3)]
    assert results[0]["should_continue"]
    assert not results[-1]["should_continue"]


@pytest.mark.parametrize("data", [{}, {"automatic": True, "selected": "missing", "options": []},
                                  {"options": "not a list"}, {"options": [None]}])
def test_malformed_plugin_results_pause_safely(manager, data):
    mgr, pm = manager
    install(pm, data)
    result = mgr.evaluate_after_turn("Need input")
    assert result["status"] == "paused"


def test_options_and_revision_bound_choice_resume_without_budget_reset(manager):
    mgr, pm = manager
    install(pm, {**proposal(authorization="approval_required"), "automatic": False})
    mgr.evaluate_after_turn("Need approval")
    show = dispatch_goal_command(mgr, "options", authorize_gate=lambda: None)
    assert "inspect" in show.output
    token = mgr.state.recovery["id"]
    choice = dispatch_goal_command(mgr, f"choose {token} inspect", authorize_gate=lambda: None)
    assert choice.prompt and "Tool approvals" in choice.prompt
    assert mgr.state.goal == "Implement the product; verify in a browser"
    assert mgr.state.turns_used == 1
    assert mgr.state.status == "active"
    replay = dispatch_goal_command(mgr, f"choose {token} inspect", authorize_gate=lambda: None)
    assert replay.error


@pytest.mark.parametrize("change", ["expired", "contract", "subgoal", "token"])
def test_stale_choices_cannot_resume(manager, change):
    mgr, pm = manager
    install(pm, {**proposal(), "automatic": False})
    mgr.evaluate_after_turn("Need input")
    token = mgr.state.recovery["id"]
    if change == "expired":
        mgr.state.recovery["expires_at"] = time.time() - 1
    elif change == "contract":
        mgr.set_contract(goals.GoalContract(verification="new acceptance"))
    elif change == "subgoal":
        mgr.add_subgoal("new requirement")
    else:
        token = "wrong"
    result = dispatch_goal_command(mgr, f"choose {token} inspect", authorize_gate=lambda: None)
    assert result.error
    assert not result.prompt
    assert mgr.state.status == "paused"


def test_choice_requires_authorized_surface(manager):
    mgr, pm = manager
    install(pm, {**proposal(), "automatic": False})
    mgr.evaluate_after_turn("Need input")
    token = mgr.state.recovery["id"]
    result = dispatch_goal_command(mgr, f"choose {token} inspect", authorize_gate=lambda: "admin required")
    assert result.error
    assert mgr.state.status == "paused"


def test_judge_rate_limit_parks_with_bounded_backoff(manager, monkeypatch):
    mgr, _ = manager
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **kw: ("continue", "judge error: RateLimitError", False, None, True))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"goal_recovery": {"judge_backoff": True}})
    result = mgr.evaluate_after_turn("Work is unfinished")
    assert result["verdict"] == "wait"
    assert not result["should_continue"]
    assert mgr.state.waiting_until > time.time()
    assert mgr.state.turns_used == 1


def test_recovery_config_corruption_cannot_crash_goal_loop(manager, monkeypatch):
    mgr, pm = manager
    install(pm, proposal())
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"goal_recovery": {"auto_effects": [{}]}})
    result = mgr.evaluate_after_turn("Need input")
    assert result["status"] == "paused"


@pytest.mark.parametrize("control", ["clear", "replace"])
def test_control_change_during_policy_does_not_get_overwritten(manager, control):
    mgr, pm = manager
    def callback(**kw):
        if control == "clear":
            mgr.clear()
        else:
            mgr.set("new user goal")
        return proposal()
    pm._hooks["goal_blocked"] = [callback]
    result = mgr.evaluate_after_turn("Need input")
    assert not result["should_continue"]
    if control == "clear":
        assert mgr.state is None
    else:
        assert mgr.state.goal == "new user goal"
        assert mgr.state.status == "active"


def test_new_turn_invalidates_previous_choice(manager):
    mgr, pm = manager
    install(pm, {**proposal(), "automatic": False})
    mgr.evaluate_after_turn("Need input")
    token = mgr.state.recovery["id"]
    mgr.state.turns_used += 1
    result = dispatch_goal_command(mgr, f"choose {token} inspect", authorize_gate=lambda: None)
    assert result.error
    assert not result.prompt


def test_wait_without_observed_deadline_never_runs(manager):
    mgr, pm = manager
    install(pm, proposal(route="wait"))
    result = mgr.evaluate_after_turn("Waiting but no concrete timer")
    assert not result["should_continue"]


def test_human_selected_wait_parks_instead_of_immediate_continuation(manager):
    mgr, pm = manager
    install(pm, {**proposal(route="wait", wait_seconds=120), "automatic": False})
    mgr.evaluate_after_turn("Await the observed dependency")
    token = mgr.state.recovery["id"]
    result = dispatch_goal_command(mgr, f"choose {token} inspect", authorize_gate=lambda: None)
    assert not result.error
    assert not result.prompt
    assert mgr.state.waiting_until > time.time()
    assert mgr.state.turns_used == 1


def test_explicit_block_selection_preserves_pause(manager):
    mgr, pm = manager
    install(pm, {**proposal(), "automatic": False})
    mgr.evaluate_after_turn("Need input")
    token = mgr.state.recovery["id"]
    result = dispatch_goal_command(mgr, f"choose {token} block", authorize_gate=lambda: None)
    assert not result.error
    assert not result.prompt
    assert mgr.state.status == "paused"
