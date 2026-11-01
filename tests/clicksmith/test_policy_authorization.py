import pytest

from clicksmith.foundation.execution_identity import ExecutionIdentity
from clicksmith.foundation.policy_authorization import (
    DECISIONS,
    AuthorizationDecision,
    AuthorizationRequest,
    AuthorityContext,
    InvalidAuthorityContextError,
    InvalidIdentityError,
    InvalidPolicyInputError,
    InvalidTaskContractError,
    Policy,
    PolicyEvaluationError,
    SchemaError,
    UnsupportedPolicyError,
    evaluate,
)
from clicksmith.foundation.task_contract import TaskContract


@pytest.fixture
def task():
    return TaskContract(task_id="task-1", contract_revision=1, objective="bounded action")


@pytest.fixture
def identity():
    return ExecutionIdentity("task-1", "run-1", "attempt-1", "session-1", 1)


def request(task, identity, allowed=("read",), granted=("read",)):
    return AuthorizationRequest(
        task=task, identity=identity,
        policy=Policy("p-1", "1", frozenset(allowed)),
        authority_context=AuthorityContext("authority-1", frozenset(granted)),
    )


def test_allow_when_scope_is_covered(task, identity):
    d = evaluate(request(task, identity, ("read",), ("read", "write")))
    assert d.decision == "ALLOW" and d.identity == identity


def test_deny_when_scope_is_not_covered(task, identity):
    d = evaluate(request(task, identity, ("write",), ("read",)))
    assert d.decision == "DENY" and d.identity == identity


def test_only_allow_and_deny_exist():
    assert DECISIONS == frozenset({"ALLOW", "DENY"})


def test_task_identity_mismatch_is_rejected(task):
    other = ExecutionIdentity("task-2", "run-1", "attempt-1", None, 1)
    with pytest.raises(InvalidTaskContractError):
        evaluate(request(task, other))


def test_invalid_identity_is_rejected(task):
    invalid = ExecutionIdentity("task-1", "bad id!", "attempt-1", None, 1)
    with pytest.raises(InvalidIdentityError):
        evaluate(request(task, invalid))


def test_missing_authority_is_error_not_decision(task, identity):
    req = AuthorizationRequest(
        task, identity, Policy("p-1", "1", frozenset({"read"})), None
    )
    with pytest.raises(InvalidAuthorityContextError):
        evaluate(req)


def test_invalid_policy_is_error_not_synthetic_deny(task, identity):
    req = AuthorizationRequest(
        task, identity, None, AuthorityContext("authority-1", frozenset({"read"}))
    )
    with pytest.raises(InvalidPolicyInputError):
        evaluate(req)


def test_empty_policy_is_unsupported(task, identity):
    with pytest.raises(UnsupportedPolicyError):
        evaluate(request(task, identity, (), ()))


def test_allow_is_bound_to_exact_attempt(task, identity):
    d = evaluate(request(task, identity))
    other = ExecutionIdentity("task-1", "run-1", "attempt-2", "session-1", 1)
    assert d.identity.attempt_id == "attempt-1"
    assert d.identity != other


def test_unknown_decision_is_rejected(identity):
    with pytest.raises(SchemaError):
        AuthorizationDecision(identity, "REQUIRE_APPROVAL", "p-1", "1").validate()


def test_decision_serialization_is_deterministic(task, identity):
    d = evaluate(request(task, identity))
    assert '"decision":"ALLOW"' in d.serialize()
    assert '"attempt_id":"attempt-1"' in d.serialize()


def test_policy_requires_immutable_scope():
    with pytest.raises(InvalidPolicyInputError):
        Policy("p-1", "1", {"read"}).validate()


def test_authority_context_only_consumes_supplied_facts():
    c = AuthorityContext("already-established", frozenset({"read"}))
    c.validate()
    assert c.authority_reference == "already-established"


def test_cancellation_vocabulary_is_not_introduced(task, identity):
    d = evaluate(request(task, identity))
    assert d.decision not in {"CANCEL", "CANCELLED"}


def test_broken_task_validation_is_fnd04_error(identity):
    class BrokenTask:
        task_id = "task-1"
        def validate(self):
            raise RuntimeError("broken")
    with pytest.raises(InvalidTaskContractError):
        evaluate(request(BrokenTask(), identity))


def test_non_request_is_schema_error():
    with pytest.raises(SchemaError):
        evaluate(object())


def test_extra_authority_does_not_change_policy_scope(task, identity):
    d = evaluate(request(task, identity, ("read",), ("read", "delete")))
    assert d.decision == "ALLOW"


def test_policy_reference_is_carried(task, identity):
    d = evaluate(request(task, identity))
    assert (d.policy_id, d.policy_version) == ("p-1", "1")


def test_revision_is_preserved(task):
    i = ExecutionIdentity("task-1", "run-1", "attempt-9", None, 7)
    assert evaluate(request(task, i)).identity.revision == 7


def test_unexpected_policy_exception_becomes_evaluation_error(task, identity, monkeypatch):
    def explode(self):
        raise RuntimeError("injected evaluator failure")
    monkeypatch.setattr(Policy, "validate", explode)
    with pytest.raises(PolicyEvaluationError):
        evaluate(request(task, identity))


def test_evaluation_error_does_not_return_decision(task, identity, monkeypatch):
    def explode(self):
        raise RuntimeError("injected")
    monkeypatch.setattr(Policy, "validate", explode)
    with pytest.raises(PolicyEvaluationError):
        evaluate(request(task, identity))
    assert not any(x in DECISIONS for x in {"ERROR", "UNKNOWN"})
