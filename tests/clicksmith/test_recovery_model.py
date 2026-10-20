import pytest

from clicksmith.foundation.execution_identity import ExecutionIdentity
from clicksmith.foundation.execution_state import ExecutionState, RUNNING
from clicksmith.foundation.evidence_model import EvidenceProvenance, EvidenceRecord
from clicksmith.foundation.recovery_model import (
    RECOVERY_ACTIONS,
    RECOVERABILITY,
    AmbiguousRecoveryError,
    FailureOrDisruption,
    InvalidIdentityError,
    InvalidRecoveryContextError,
    InvalidRecoveryInputError,
    RecoveryDecision,
    RecoveryEvaluationError,
    RecoveryRequest,
    SchemaError,
    evaluate,
)
from clicksmith.foundation.task_contract import TaskContract


@pytest.fixture
def task():
    return TaskContract(task_id="task-1", contract_revision=1, objective="bounded action")


@pytest.fixture
def identity():
    return ExecutionIdentity("task-1", "run-1", "attempt-1", "session-1", 1)


def request(task, identity, *, recoverability="RECOVERABLE", retry_required=False, evidence=()):
    return RecoveryRequest(
        task=task,
        execution_identity=identity,
        execution_state=ExecutionState(identity, RUNNING),
        failure_or_disruption=FailureOrDisruption(
            "condition-1", recoverability, retry_required
        ),
        applicable_evidence=evidence,
    )


def test_canonical_actions_are_exact():
    assert RECOVERY_ACTIONS == frozenset({"RECOVERY_REQUIRED", "NO_RECOVERY", "RETRY_REQUIRED"})


def test_canonical_recoverability_is_exact():
    assert RECOVERABILITY == frozenset({"RECOVERABLE", "NON_RECOVERABLE"})


def test_recoverable_non_retry_condition_requires_recovery(task, identity):
    decision = evaluate(request(task, identity))
    assert decision.action == "RECOVERY_REQUIRED"
    assert decision.execution_identity == identity


def test_recoverable_retry_condition_declares_retry_required(task, identity):
    decision = evaluate(request(task, identity, retry_required=True))
    assert decision.action == "RETRY_REQUIRED"
    assert decision.action not in {RUNNING, "RECOVERING", "RETRYING", "RESTORING", "RESUMING"}


def test_non_recoverable_condition_requires_no_recovery(task, identity):
    decision = evaluate(request(task, identity, recoverability="NON_RECOVERABLE"))
    assert decision.action == "NO_RECOVERY"


def test_non_recoverable_retry_is_ambiguous(task, identity):
    with pytest.raises(AmbiguousRecoveryError):
        evaluate(request(task, identity, recoverability="NON_RECOVERABLE", retry_required=True))


def test_invalid_identity_is_rejected(task):
    bad = ExecutionIdentity("task-1", "bad id!", "attempt-1", None, 1)
    valid_state_identity = ExecutionIdentity("task-1", "run-1", "attempt-1", None, 1)
    req = RecoveryRequest(
        task, bad, ExecutionState(valid_state_identity, RUNNING),
        FailureOrDisruption("condition-1", "RECOVERABLE", True), ()
    )
    with pytest.raises(InvalidIdentityError):
        evaluate(req)


def test_task_identity_mismatch_is_rejected(identity):
    other_task = TaskContract(task_id="task-2", contract_revision=1, objective="bounded action")
    with pytest.raises(InvalidRecoveryContextError):
        evaluate(request(other_task, identity))


def test_state_identity_mismatch_is_rejected(task, identity):
    other = ExecutionIdentity("task-1", "run-1", "attempt-2", "session-1", 1)
    req = RecoveryRequest(
        task, identity, ExecutionState(other, RUNNING),
        FailureOrDisruption("condition-1", "RECOVERABLE", True), ()
    )
    with pytest.raises(InvalidRecoveryContextError):
        evaluate(req)


def test_execution_evidence_must_match_identity(task, identity):
    other = ExecutionIdentity("task-1", "run-1", "attempt-2", None, 1)
    evidence = EvidenceRecord(
        evidence_id="ev-1", evidence_type="FAILURE",
        provenance=EvidenceProvenance("FND-06", "recovery", "test"),
        identity_applicability="EXECUTION", identity=other,
    )
    with pytest.raises(InvalidRecoveryContextError):
        evaluate(request(task, identity, evidence=(evidence,)))


def test_non_execution_evidence_can_be_consumed_without_identity(task, identity):
    evidence = EvidenceRecord(
        evidence_id="ev-1", evidence_type="VERIFICATION_REFERENCE",
        provenance=EvidenceProvenance("FND-05", "verification", "test"),
        identity_applicability="NON_EXECUTION", reference="ref-1",
    )
    decision = evaluate(request(task, identity, evidence=(evidence,)))
    assert decision.action == "RECOVERY_REQUIRED"


def test_evidence_is_not_mutated(task, identity):
    evidence = EvidenceRecord(
        evidence_id="ev-1", evidence_type="FAILURE",
        provenance=EvidenceProvenance("FND-06", "recovery", "test"),
        identity_applicability="EXECUTION", identity=identity,
    )
    before = evidence
    evaluate(request(task, identity, evidence=(evidence,)))
    assert evidence == before


def test_request_requires_immutable_evidence_collection(task, identity):
    req = request(task, identity)
    object.__setattr__(req, "applicable_evidence", [])
    with pytest.raises(InvalidRecoveryInputError):
        evaluate(req)


def test_unknown_action_is_rejected(identity):
    with pytest.raises(Exception) as exc:
        RecoveryDecision(identity, "RETRY", "condition-1").validate()
    assert getattr(exc.value, "code", None) == "UNSUPPORTED_RECOVERY_ACTION"


def test_non_request_is_schema_error():
    with pytest.raises(SchemaError):
        evaluate(object())


def test_error_is_not_converted_to_retry(task, identity, monkeypatch):
    def explode(self):
        raise RuntimeError("injected")
    monkeypatch.setattr(FailureOrDisruption, "validate", explode)
    with pytest.raises(RecoveryEvaluationError):
        evaluate(request(task, identity))


def test_decision_is_bound_to_exact_attempt(task, identity):
    decision = evaluate(request(task, identity))
    other = ExecutionIdentity("task-1", "run-1", "attempt-2", "session-1", 1)
    assert decision.execution_identity.attempt_id == "attempt-1"
    assert decision.execution_identity != other


def test_identity_fields_are_not_rewritten(task, identity):
    decision = evaluate(request(task, identity, retry_required=True))
    assert decision.execution_identity.task_id == "task-1"
    assert decision.execution_identity.run_id == "run-1"
    assert decision.execution_identity.attempt_id == "attempt-1"
    assert decision.execution_identity.revision == 1


def test_decision_serialization_is_deterministic(task, identity):
    decision = evaluate(request(task, identity, retry_required=True))
    assert decision.serialize() == decision.serialize()
    assert '"action":"RETRY_REQUIRED"' in decision.serialize()


def test_decision_does_not_grant_authorization(task, identity):
    decision = evaluate(request(task, identity, retry_required=True))
    assert decision.action not in {"ALLOW", "DENY"}


def test_fnd03_state_is_not_changed(task, identity):
    state = ExecutionState(identity, RUNNING)
    req = RecoveryRequest(
        task, identity, state,
        FailureOrDisruption("condition-1", "RECOVERABLE", True), ()
    )
    evaluate(req)
    assert state.state == RUNNING
