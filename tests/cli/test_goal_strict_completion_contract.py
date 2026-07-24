"""Regression tests for strict /goal completion semantics.

Goal Mode must not let an agent end a long task by apologizing, claiming the
scope is too large, or otherwise giving up before the stated goal is actually
complete. The judge may be model-backed and can be overly credulous, so the
GoalManager owns a small deterministic backstop for obvious premature stop
patterns.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes stay hermetic."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _manager(goal: str, *, max_turns: int = 5, contract=None):
    from hermes_cli.goals import GoalManager

    sid = f"sid-goal-{uuid.uuid4().hex}"
    mgr = GoalManager(session_id=sid, default_max_turns=max_turns)
    mgr.set(goal, contract=contract)
    return mgr


def _evaluate_with_done_judge(mgr, response: str, reason: str = "judge said done"):
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", reason, False, None),
    ):
        return mgr.evaluate_after_turn(response)


def assert_continues(decision, mgr):
    assert decision["status"] == "active"
    assert decision["verdict"] == "continue"
    assert decision["should_continue"] is True
    assert mgr.state is not None
    assert mgr.state.status == "active"


def assert_done(decision, mgr):
    assert decision["status"] == "done"
    assert decision["verdict"] == "done"
    assert decision["should_continue"] is False
    assert mgr.state is not None
    assert mgr.state.status == "done"


def test_premature_give_up_response_overrides_credulous_done_judge(hermes_home):
    mgr = _manager(
        "Create a complete migration plan and implement the safe local changes"
    )
    response = (
        "I'm sorry, but this task is too long for one response. "
        "I can't complete the whole thing here, but here's a summary of what I tried."
    )
    decision = _evaluate_with_done_judge(
        mgr, response, "assistant said it could not complete the task"
    )
    assert_continues(decision, mgr)
    assert decision["continuation_prompt"]
    assert "Continuing toward your standing goal" in decision["continuation_prompt"]
    assert mgr.state.last_verdict == "continue"
    assert "premature stop" in (mgr.state.last_reason or "").lower()


def test_clear_completion_evidence_still_allows_done_verdict(hermes_home):
    mgr = _manager("Write the artifact and verify it")
    response = "Done: wrote /tmp/artifact.txt and verified it with pytest tests/foo_test.py -q (3 passed)."
    decision = _evaluate_with_done_judge(
        mgr, response, "deliverable produced and verification passed"
    )
    assert_done(decision, mgr)


def test_specific_actionable_external_blocker_still_allows_done_verdict(hermes_home):
    mgr = _manager("Call the private API and write the response to disk")
    response = (
        "I'm sorry, I cannot complete this because the required API credentials "
        "are missing. Please provide API credentials or approve using a different source."
    )
    decision = _evaluate_with_done_judge(
        mgr, response, "specific external blocker requires user action"
    )
    assert_done(decision, mgr)


def test_specific_missing_data_blocker_still_allows_done_verdict(hermes_home):
    mgr = _manager("Analyze the provided CSV and write the report")
    response = (
        "I cannot complete this here because the required CSV data is missing. "
        "Please provide the data file."
    )
    decision = _evaluate_with_done_judge(
        mgr, response, "specific missing data file requires user action"
    )
    assert_done(decision, mgr)


def test_incidental_apology_after_verified_completion_still_allows_done(hermes_home):
    mgr = _manager("Write the artifact and verify it")
    response = (
        "I'm sorry this took too long. Done: wrote /tmp/artifact.txt and "
        "verified it with pytest tests/foo_test.py -q (3 passed)."
    )
    decision = _evaluate_with_done_judge(
        mgr, response, "deliverable produced and verification passed"
    )
    assert_done(decision, mgr)


def test_historical_surrender_wording_with_verified_completion_allows_done(hermes_home):
    mgr = _manager("Write the artifact and verify it")
    response = (
        "I initially thought I cannot complete the whole thing here, but I did: "
        "wrote /tmp/artifact.txt and pytest tests/foo_test.py passed."
    )
    decision = _evaluate_with_done_judge(mgr, response, "verified completion evidence")
    assert_done(decision, mgr)


def test_partial_progress_plus_give_up_still_continues(hermes_home):
    mgr = _manager("Fix all bugs and verify the full suite")
    response = (
        "I can't complete the whole thing here. I fixed the first bug, "
        "but the rest still needs more work."
    )
    decision = _evaluate_with_done_judge(
        mgr, response, "assistant mentioned fixing something"
    )
    assert_continues(decision, mgr)


def test_generic_need_more_time_is_not_actionable_blocker(hermes_home):
    mgr = _manager("Complete the migration and verify it")
    response = (
        "I cannot complete the whole thing because I need more time. "
        "Please confirm if you want me to continue later."
    )
    decision = _evaluate_with_done_judge(
        mgr, response, "assistant asked to continue later"
    )
    assert_continues(decision, mgr)


def test_cannot_complete_this_here_with_summary_still_continues(hermes_home):
    mgr = _manager("Complete the audit and write the final report")
    response = "I can't complete this here, but here's a summary of what I did so far."
    decision = _evaluate_with_done_judge(mgr, response, "assistant gave a summary")
    assert_continues(decision, mgr)


def test_vague_access_permission_blocker_still_continues(hermes_home):
    mgr = _manager("Complete all repository changes and verify them")
    response = "I cannot complete the whole thing because I need access and permission."
    decision = _evaluate_with_done_judge(
        mgr, response, "assistant said it needed access"
    )
    assert_continues(decision, mgr)


def test_vague_no_access_blocker_still_continues(hermes_home):
    mgr = _manager("Complete all repository changes and verify them")
    response = "I cannot complete this because I have no access."
    decision = _evaluate_with_done_judge(
        mgr, response, "assistant said it had no access"
    )
    assert_continues(decision, mgr)


def test_failed_tests_with_passed_word_still_continues(hermes_home):
    mgr = _manager("Complete all repository changes and verify them")
    response = "I can't complete the whole thing here because pytest failed (0 passed, 3 failed)."
    decision = _evaluate_with_done_judge(mgr, response, "assistant mentioned passed")
    assert_continues(decision, mgr)


def test_wait_verdict_retains_wait_directive_behavior(hermes_home):
    mgr = _manager("Wait for the running CI job, then finish the release packet")
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("wait", "CI is still running", False, {"seconds": 30}),
    ):
        decision = mgr.evaluate_after_turn("CI is still running. Waiting for it.")

    assert decision["status"] == "active"
    assert decision["verdict"] == "wait"
    assert decision["should_continue"] is False
    assert mgr.state is not None
    assert mgr.state.waiting_until > 0


def test_contract_aware_done_verdict_bypasses_free_form_backstop(hermes_home):
    from hermes_cli.goals import GoalContract

    mgr = _manager(
        "Produce the migration artifact",
        contract=GoalContract(
            verification="The contract-aware judge verifies the artifact"
        ),
    )
    response = (
        "I cannot complete the whole thing here, but the contract judge accepted it."
    )
    decision = _evaluate_with_done_judge(
        mgr,
        response,
        "completion contract was satisfied",
    )

    assert_done(decision, mgr)
