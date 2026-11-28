"""Regression for #137546: only applicable judge feedback reaches working turns."""

import json

import pytest

from hermes_cli import goals
from hermes_cli.goal_command import dispatch_goal_command
from hermes_cli.goals import GoalContract, GoalManager


FEEDBACK = "Show the exact test command and its passing result."
NEW_FEEDBACK = "Include the new verification evidence."


def judge(monkeypatch, mgr, reason=FEEDBACK, verdict="continue", parse=False, wait=None, transport=False):
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: (verdict, reason, parse, wait, transport))
    return mgr.evaluate_after_turn("The task is complete.")


def resumed_prompt(mgr):
    mgr.pause()
    reloaded = GoalManager(mgr.session_id)
    return dispatch_goal_command(reloaded, "resume", authorize_gate=lambda: None).prompt


@pytest.mark.parametrize("mutation", ["contract", "remove", "clear", "add", "goal", "gate"])
def test_criteria_mutation_drops_feedback_after_reload_resume(monkeypatch, mutation):
    mgr = GoalManager(f"feedback-mutation-{mutation}")
    mgr.set("ship the fix", contract=GoalContract(verification="old verification"))
    mgr.add_subgoal("old criterion")
    assert FEEDBACK in judge(monkeypatch, mgr)["continuation_prompt"]

    if mutation == "contract":
        mgr.set_contract(GoalContract(verification="replacement verification"))
    elif mutation == "remove":
        mgr.remove_subgoal(1)
    elif mutation == "clear":
        mgr.clear_subgoals()
    elif mutation == "add":
        mgr.add_subgoal("new criterion")
    elif mutation == "goal":
        mgr.set("ship a different fix")
    else:
        mgr.add_gate("verify")
    if mutation != "goal":
        assert mgr.state.last_reason == FEEDBACK

    prompt = resumed_prompt(mgr)
    assert FEEDBACK not in prompt
    assert mgr.state.goal in prompt
    if mutation == "contract":
        assert "replacement verification" in prompt
        assert "old verification" not in prompt
    if mutation in ("remove", "clear"):
        assert "old criterion" not in prompt
        # Restoring earlier criteria must not resurrect invalidated feedback.
        mgr.add_subgoal("old criterion")
        assert FEEDBACK not in resumed_prompt(mgr)

    if mutation == "gate":
        monkeypatch.setattr(goals, "_gate_workspace", lambda: (None, None))
        monkeypatch.setattr(goals, "run_gate", lambda *a, **k: (True, 0, ""))
    updated = GoalManager(mgr.session_id)
    prompt = judge(monkeypatch, updated, NEW_FEEDBACK)["continuation_prompt"]
    assert NEW_FEEDBACK in prompt
    assert FEEDBACK not in prompt


def test_unchanged_feedback_survives_reload_resume_and_session_rotation(monkeypatch):
    mgr = GoalManager("feedback-unchanged")
    mgr.set("ship the fix", contract=GoalContract(verification=" run the tests "))
    mgr.add_subgoal("keep the migration")
    judge(monkeypatch, mgr)
    prompt = resumed_prompt(mgr)
    assert FEEDBACK in prompt
    assert "run the tests" in prompt
    assert "keep the migration" in prompt
    assert goals.migrate_goal_to_session(mgr.session_id, "feedback-rotated")
    assert FEEDBACK in GoalManager("feedback-rotated").next_continuation_prompt()


def test_gate_failure_supersedes_feedback_without_erasing_diagnostics(monkeypatch):
    mgr = GoalManager("feedback-gate")
    mgr.set("ship the fix")
    mgr.add_gate("verify")
    monkeypatch.setattr(goals, "_gate_workspace", lambda: (None, None))
    monkeypatch.setattr(goals, "run_gate", lambda *a, **k: (True, 0, ""))
    judge(monkeypatch, mgr)
    monkeypatch.setattr(goals, "run_gate", lambda *a, **k: (False, 1, "gate evidence"))
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: pytest.fail("failed gate must skip judge"))
    decision = mgr.evaluate_after_turn("done")
    assert "gate evidence" in decision["continuation_prompt"]
    assert FEEDBACK not in decision["continuation_prompt"]
    assert mgr.state.last_reason == FEEDBACK
    assert FEEDBACK not in resumed_prompt(mgr)
    monkeypatch.setattr(goals, "run_gate", lambda *a, **k: (True, 0, ""))
    assert NEW_FEEDBACK in judge(monkeypatch, GoalManager(mgr.session_id), NEW_FEEDBACK)["continuation_prompt"]


@pytest.mark.parametrize("outcome", ["blocked", "done", "wait", "parse", "transport", "placeholder", "malformed_wait"])
def test_superseding_judgment_drops_previous_feedback(monkeypatch, outcome):
    mgr = GoalManager(f"feedback-outcome-{outcome}")
    mgr.set("ship the fix")
    judge(monkeypatch, mgr)
    kwargs = {
        "blocked": {"verdict": "blocked"},
        "done": {"verdict": "done"},
        "wait": {"verdict": "wait", "wait": {"seconds": 60}},
        "parse": {"parse": True},
        "transport": {"transport": True},
        "placeholder": {"reason": "no reason provided"},
        "malformed_wait": {"reason": "wait (wait verdict had no target — continuing)"},
    }[outcome]
    kwargs.setdefault("reason", "superseding diagnostic")
    judge(monkeypatch, mgr, **kwargs)
    prompt = resumed_prompt(mgr)
    assert FEEDBACK not in prompt
    assert kwargs["reason"] not in prompt
    assert NEW_FEEDBACK in judge(monkeypatch, GoalManager(mgr.session_id), NEW_FEEDBACK)["continuation_prompt"]


def test_legacy_reason_without_provenance_never_replays(monkeypatch):
    legacy = json.dumps({"goal": "ship the fix", "last_verdict": "continue", "last_reason": FEEDBACK})
    goals.save_goal("feedback-legacy", goals.GoalState.from_json(legacy))
    mgr = GoalManager("feedback-legacy")
    assert mgr.state.last_reason == FEEDBACK
    assert FEEDBACK not in resumed_prompt(mgr)
    assert NEW_FEEDBACK in judge(monkeypatch, GoalManager(mgr.session_id), NEW_FEEDBACK)["continuation_prompt"]


def test_prompt_checks_live_criteria_even_before_saving(monkeypatch):
    mgr = GoalManager("feedback-live-criteria")
    mgr.set("ship the fix")
    judge(monkeypatch, mgr)
    mgr.state.contract.verification = "replacement verification"
    prompt = mgr.next_continuation_prompt()
    assert "replacement verification" in prompt
    assert FEEDBACK not in prompt


def test_multiline_feedback_remains_bounded_after_reload(monkeypatch):
    mgr = GoalManager("feedback-bound")
    mgr.set("ship the fix")
    reason = "first request\nsecond request\n" + "x" * 2100 + "omitted tail"
    judge(monkeypatch, mgr, reason)
    prompt = resumed_prompt(mgr)
    assert "first request\nsecond request" in prompt
    assert "[truncated]" in prompt
    assert "omitted tail" not in prompt
