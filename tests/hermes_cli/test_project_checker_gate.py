"""Focused behavioral contract for the independent checker gate (HOF-015)."""

from __future__ import annotations

from dataclasses import replace

from hermes_cli.project_checker_gate import (
    BLOCKED,
    CHECKER_ALREADY_EXISTS,
    CHECKER_OF_CHECKER_DENIED,
    CHECKER_REQUIRED,
    FAIL_REPAIRABLE,
    MALFORMED_VERDICT,
    PASS_ACCEPTED,
    STALE_LOCK_VERSION,
    STALE_SNAPSHOT,
    STALE_VERDICT,
    UNAUTHORIZED_CHECKER,
    CandidateBinding,
    CheckerIdentity,
    CheckerRequest,
    MachineVerdict,
    ProjectIdentity,
    create_checker_gate,
    evaluate_checker_gate,
    reconcile_checker,
    submit_checker_verdict,
)


PROJECT = ProjectIdentity(project_id="project", generation=4)
CANDIDATE = CandidateBinding(snapshot_version="snapshot-a", candidate_id="candidate-a")
IMPLEMENTERS = ("implementer-a", "implementer-b")
CHECKER = CheckerRequest(task_id="checker-1", run_id="run-1", profile="checker-a")


def _gate():
    return create_checker_gate(
        project=PROJECT,
        candidate=CANDIDATE,
        implementation_profiles=IMPLEMENTERS,
    )


def _checked_gate():
    gate, created = reconcile_checker(_gate(), request=CHECKER, expected_version=0)
    assert created.outcome == CHECKER_REQUIRED
    return gate


def _verdict(status="PASS", **changes):
    fields = {
        "project": PROJECT,
        "candidate": CANDIDATE,
        "checker": CheckerIdentity(task_id="checker-1", run_id="run-1", profile="checker-a"),
        "status": status,
    }
    fields.update(changes)
    return MachineVerdict(**fields)


def test_current_authorized_pass_unlocks_complete_eligible():
    gate = _checked_gate()

    gate, accepted = submit_checker_verdict(gate, verdict=_verdict(), expected_version=1)
    decision = evaluate_checker_gate(gate, project=PROJECT, candidate=CANDIDATE)

    assert accepted.outcome == PASS_ACCEPTED
    assert decision.outcome == PASS_ACCEPTED
    assert decision.lifecycle == "COMPLETE_ELIGIBLE"


def test_missing_and_malformed_verdicts_do_not_unlock_completion():
    gate = _checked_gate()

    assert evaluate_checker_gate(gate, project=PROJECT, candidate=CANDIDATE).outcome == CHECKER_REQUIRED
    unchanged, malformed = submit_checker_verdict(gate, verdict={"status": "PASS"}, expected_version=1)

    assert malformed.outcome == MALFORMED_VERDICT
    assert unchanged == gate
    assert evaluate_checker_gate(unchanged, project=PROJECT, candidate=CANDIDATE).lifecycle != "COMPLETE_ELIGIBLE"


def test_non_independent_or_non_designated_checker_is_rejected():
    gate = _checked_gate()
    non_independent = _verdict(checker=CheckerIdentity("checker-1", "run-1", "implementer-a"))
    wrong_checker = _verdict(checker=CheckerIdentity("checker-2", "run-2", "checker-b"))

    _, independent_result = submit_checker_verdict(gate, verdict=non_independent, expected_version=1)
    _, designated_result = submit_checker_verdict(gate, verdict=wrong_checker, expected_version=1)

    assert independent_result.outcome == UNAUTHORIZED_CHECKER
    assert designated_result.outcome == UNAUTHORIZED_CHECKER


def test_failed_and_blocked_verdicts_are_historical_and_route_without_repair():
    failed_gate = _checked_gate()
    failed_gate, failed = submit_checker_verdict(failed_gate, verdict=_verdict("FAIL_REPAIRABLE"), expected_version=1)
    failed_decision = evaluate_checker_gate(failed_gate, project=PROJECT, candidate=CANDIDATE)

    blocked_gate = _checked_gate()
    blocked_gate, blocked = submit_checker_verdict(blocked_gate, verdict=_verdict("BLOCKED"), expected_version=1)
    blocked_decision = evaluate_checker_gate(blocked_gate, project=PROJECT, candidate=CANDIDATE)

    assert failed.outcome == FAIL_REPAIRABLE
    assert failed_decision.lifecycle == "REPAIR_POLICY"
    assert blocked.outcome == BLOCKED
    assert blocked_decision.lifecycle == "TERMINAL_MANUAL_BLOCKED"
    assert [entry.outcome for entry in failed_gate.history] == [FAIL_REPAIRABLE]
    assert [entry.outcome for entry in blocked_gate.history] == [BLOCKED]


def test_repaired_candidate_requires_a_fresh_checker_and_preserves_old_pass():
    gate = _checked_gate()
    gate, _ = submit_checker_verdict(gate, verdict=_verdict(), expected_version=1)
    repaired = CandidateBinding(snapshot_version="snapshot-b", candidate_id="candidate-b")

    gate, required = reconcile_checker(gate, request=CHECKER, candidate=repaired, expected_version=2)
    before_fresh = evaluate_checker_gate(gate, project=PROJECT, candidate=repaired)
    fresh = CheckerRequest(task_id="checker-2", run_id="run-2", profile="checker-a")
    gate, created = reconcile_checker(gate, request=fresh, candidate=repaired, expected_version=3)
    stale = evaluate_checker_gate(gate, project=PROJECT, candidate=CANDIDATE)

    assert required.outcome == CHECKER_REQUIRED
    assert before_fresh.outcome == CHECKER_REQUIRED
    assert created.outcome == CHECKER_REQUIRED
    assert stale.outcome == STALE_VERDICT
    assert [entry.outcome for entry in gate.history] == [PASS_ACCEPTED, STALE_VERDICT]


def test_one_active_checker_and_duplicate_reconciliation_are_idempotent():
    gate = _checked_gate()
    same_gate, repeated = reconcile_checker(gate, request=CHECKER, expected_version=1)
    other_gate, ambiguous = reconcile_checker(
        gate,
        request=CheckerRequest(task_id="checker-2", run_id="run-2", profile="checker-b"),
        expected_version=1,
    )

    assert same_gate == gate
    assert repeated.outcome == CHECKER_ALREADY_EXISTS
    assert other_gate == gate
    assert ambiguous.outcome == CHECKER_ALREADY_EXISTS
    assert gate.active_checker == CheckerIdentity("checker-1", "run-1", "checker-a")


def test_verdict_binding_rejects_stale_snapshot_and_preserves_evidence():
    gate = _checked_gate()
    stale_candidate = CandidateBinding(snapshot_version="snapshot-old", candidate_id="candidate-old")

    gate, result = submit_checker_verdict(
        gate,
        verdict=_verdict(candidate=stale_candidate),
        expected_version=1,
    )

    assert result.outcome == STALE_SNAPSHOT
    assert gate.active_verdict is None
    assert gate.history[-1].outcome == STALE_SNAPSHOT


def test_checker_of_checker_has_no_creation_path():
    gate = _gate()

    unchanged, result = reconcile_checker(
        gate,
        request=CheckerRequest("checker-1", "run-1", "checker-a", target_is_checker=True),
        expected_version=0,
    )

    assert result.outcome == CHECKER_OF_CHECKER_DENIED
    assert unchanged == gate


def test_stale_lock_version_refuses_mutation_deterministically():
    gate = _checked_gate()

    unchanged, result = reconcile_checker(gate, request=CHECKER, expected_version=0)

    assert result.outcome == STALE_LOCK_VERSION
    assert unchanged == gate


def test_machine_readable_mapping_requires_exact_identity_fields():
    gate = _checked_gate()
    verdict = {
        "project_id": "project",
        "generation": 4,
        "snapshot_version": "snapshot-a",
        "candidate_id": "candidate-a",
        "checker_task_id": "checker-1",
        "checker_run_id": "run-1",
        "checker_profile": "checker-a",
        "status": "PASS",
    }

    gate, accepted = submit_checker_verdict(gate, verdict=verdict, expected_version=1)
    _, malformed = submit_checker_verdict(gate, verdict={**verdict, "extra": "ambiguous"}, expected_version=2)

    assert accepted.outcome == PASS_ACCEPTED
    assert malformed.outcome == MALFORMED_VERDICT
    assert evaluate_checker_gate(gate, project=PROJECT, candidate=CANDIDATE).lifecycle == "COMPLETE_ELIGIBLE"
