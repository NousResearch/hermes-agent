"""Same-worker handoffs are opt-in, bounded, durable and never new authority."""
from copy import deepcopy

import pytest

from hermes_cli import goals
from hermes_cli.plugins import PluginManager


def approved():
    return {
        "selected": "diagnose", "automatic": True,
        "classifications": {"cause": {"label": "service", "probability": .99}},
        "options": [{
            "id": "diagnose", "route": "gather_evidence",
            "plan": "Inspect the saved failure and recommend one verified repair.",
            "verification": "Bind the diagnosis to the existing checkpoint.",
            "effects": ["read_only"], "risk": "low", "authorization": "authorized",
            "reversibility": "reversible", "progress": "useful", "evidence": "sufficient",
        }],
    }


@pytest.fixture
def setup(monkeypatch):
    config = {"goal_recovery": {"auto_effects": ["read_only"], "model_handoff": {
        "enabled": True, "session_ids": ["worker"], "max_attempts": 3,
        "routes": [{"provider": "test", "model": f"alternate-{n}"} for n in range(1, 5)],
    }}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: deepcopy(config))
    pm = PluginManager()
    data = approved()
    pm._hooks["goal_blocked"] = [lambda **kwargs: deepcopy(data)]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: pm)
    monkeypatch.setattr(goals, "judge_goal", lambda *args, **kwargs: ("blocked", "service failed", False, None, False))
    mgr = goals.GoalManager("worker", default_max_turns=20)
    mgr.set("Implement the existing product", contract=goals.GoalContract(verification="original gate"))
    mgr.worker_model_route = {"provider": "test", "model": "original"}
    return mgr, config, data


def test_safe_handoff_reserves_original_worker_and_budget(setup):
    mgr, _, _ = setup
    result = mgr.evaluate_after_turn("A small service failure needs another diagnosis.")
    assert result.get("model_handoff"), "recovery must reserve a model handoff"
    pending = result["model_handoff"]
    assert pending["session_id"] == "worker"
    assert pending["target"] == {"provider": "test", "model": "alternate-1"}
    assert result["should_continue"] is False, "do not continue before the switch commits"
    assert mgr.state.turns_used == 1
    assert mgr.state.max_turns == 20
    assert mgr.state.contract.verification == "original gate"
    loaded = goals.GoalManager("worker")
    assert loaded.state.model_recovery["pending"]["id"] == pending["id"]
    assert "recommended" in pending["prompt"].lower()


@pytest.mark.parametrize("change", ["disabled", "unscoped", "unsupported", "permission", "resource", "unsafe", "budget"])
def test_invalid_handoff_never_requests_a_model_change(setup, change):
    mgr, config, data = setup
    cfg = config["goal_recovery"]["model_handoff"]
    actions = {
        "disabled": lambda: cfg.update(enabled=False),
        "unscoped": lambda: cfg.update(session_ids=["someone-else"]),
        "unsupported": lambda: setattr(mgr, "worker_model_route", None),
        "permission": lambda: data["classifications"]["cause"].update(label="permissions"),
        "resource": lambda: data["classifications"]["cause"].update(label="resource"),
        "unsafe": lambda: data["options"][0].update(effects=["deploy"]),
        "budget": lambda: setattr(mgr.state, "turns_used", 19),
    }
    actions[change]()
    result = mgr.evaluate_after_turn("Still blocked")
    assert not result.get("model_handoff")


def test_three_attempts_remain_bounded_across_restart(setup):
    mgr, _, _ = setup
    for number in range(1, 4):
        result = mgr.evaluate_after_turn(f"Failure {number}")
        assert result.get("model_handoff")
        assert result["model_handoff"]["ordinal"] == number
        assert result["model_handoff"]["target"]["model"] == f"alternate-{number}"
        from hermes_cli.goal_model_recovery import settle
        assert settle(mgr, result["model_handoff"]["id"], "blocked") is not None
        mgr = goals.GoalManager("worker")
        mgr.worker_model_route = {"provider": "test", "model": f"alternate-{number}"}
    result = mgr.evaluate_after_turn("Failure 4")
    assert not result.get("model_handoff")
    assert result["status"] == "paused"
    assert len(mgr.state.model_recovery["attempts"]) == 3


def test_automation_resume_does_not_reset_handoff_attempts(setup):
    mgr, _, _ = setup
    result = mgr.evaluate_after_turn("Failure")
    assert result.get("model_handoff")
    mgr.resume(reset_budget=False)
    assert len(mgr.state.model_recovery["attempts"]) == 1


def test_newer_user_pause_wins_over_pending_handoff(setup):
    mgr, _, _ = setup
    result = mgr.evaluate_after_turn("Failure")
    assert result.get("model_handoff")
    another = goals.GoalManager("worker")
    another.pause("user pause")
    from hermes_cli.goal_model_recovery import current_pending
    assert current_pending(mgr.session_id, result["model_handoff"]["id"]) is None
    assert goals.load_goal("worker").status == "paused"


def test_config_cannot_inject_credentials_or_endpoint(setup):
    mgr, config, _ = setup
    config["goal_recovery"]["model_handoff"]["routes"] = [
        {"provider": "test", "model": "alternate-1", "api_key": "not-authorized"},
        {"provider": "test", "model": "alternate-2", "base_url": "https://other.invalid"},
    ]
    assert not mgr.evaluate_after_turn("Failure").get("model_handoff")


def test_explicit_pause_then_resume_cancels_pending_without_resetting_count(setup):
    mgr, _, _ = setup
    assert mgr.evaluate_after_turn("Failure").get("model_handoff")
    mgr.pause("user pause")
    mgr.resume(reset_budget=False)
    assert mgr.state.model_recovery.get("pending") is None
    assert mgr.state.turns_used == 1
    assert len(mgr.state.model_recovery["attempts"]) == 1


def test_contract_change_does_not_leave_a_stale_pending_loop(setup, monkeypatch):
    mgr, _, _ = setup
    result = mgr.evaluate_after_turn("Failure")
    assert result.get("model_handoff")
    mgr.set_contract(goals.GoalContract(verification="new user gate"))
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **kw: ("continue","safe progress",False,None,False))
    result = mgr.evaluate_after_turn("New user-requested progress")
    assert not result.get("model_handoff")
    assert result["should_continue"] is True
    assert mgr.state.model_recovery.get("pending") is None
    assert mgr.state.contract.verification == "new user gate"


def test_pending_wait_does_not_spend_another_goal_turn(setup):
    mgr, _, _ = setup
    first = mgr.evaluate_after_turn("Failure")
    second = mgr.evaluate_after_turn("Awaiting the same handoff")
    assert first["model_handoff"]["id"] == second["model_handoff"]["id"]
    assert mgr.state.turns_used == 1
