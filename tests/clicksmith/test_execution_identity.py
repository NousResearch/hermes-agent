import json

import pytest

from clicksmith.foundation.execution_identity import (
    SCHEMA_VERSION,
    ExecutionIdentity,
    ExecutionIdentityError,
    IdentityCollisionError,
    validate_retry_identity,
    validate_task_association,
)


class Resolver:
    def __init__(self, runs=None, attempts=None):
        self.runs = runs or {}
        self.attempts = attempts or {}

    def resolve_run_task(self, run_id):
        return self.runs.get(run_id)

    def resolve_attempt_run(self, attempt_id):
        return self.attempts.get(attempt_id)


def identity(
    task_id="task-001",
    run_id="run-001",
    attempt_id="attempt-001",
    session_id="session-001",
    revision=1,
):
    return ExecutionIdentity(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        session_id=session_id,
        revision=revision,
    )


# ============================================================
# FT-02-01 — valid construction and validation
# ============================================================

def test_valid_identity():
    item = identity()
    item.validate()
    assert item.task_id == "task-001"
    assert item.run_id == "run-001"
    assert item.attempt_id == "attempt-001"
    assert item.session_id == "session-001"
    assert item.revision == 1
    assert item.schema_version == SCHEMA_VERSION


# ============================================================
# FT-02-02 — canonical dictionary representation
# ============================================================

def test_to_dict_is_canonical():
    item = identity()

    assert item.to_dict() == {
        "attempt_id": "attempt-001",
        "revision": 1,
        "run_id": "run-001",
        "schema_version": "1.0",
        "session_id": "session-001",
        "task_id": "task-001",
    }


# ============================================================
# FT-02-03 — deterministic serialization
# ============================================================

def test_serialization_is_deterministic():
    item = identity()

    first = item.serialize()
    second = item.serialize()

    assert first == second
    assert first == (
        '{"attempt_id":"attempt-001",'
        '"revision":1,'
        '"run_id":"run-001",'
        '"schema_version":"1.0",'
        '"session_id":"session-001",'
        '"task_id":"task-001"}'
    )


# ============================================================
# FT-02-04 — round trip
# ============================================================

def test_deserialize_round_trip():
    item = identity()

    restored = ExecutionIdentity.deserialize(item.serialize())

    assert restored == item


# ============================================================
# FT-02-05 — TASK_ID association with FND-01 boundary
# ============================================================

def test_task_association_accepts_nonempty_fnd01_task_id():
    item = identity(task_id="FND-01/task 001")

    validate_task_association(item, "FND-01/task 001")


def test_task_association_rejects_mismatch():
    item = identity(task_id="task-001")

    with pytest.raises(ExecutionIdentityError):
        validate_task_association(item, "task-002")


# ============================================================
# FT-02-06 — retry semantics
# ============================================================

def test_retry_preserves_task_and_revision_and_changes_attempt():
    previous = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-001",
        revision=4,
    )

    retry = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-002",
        revision=4,
    )

    validate_retry_identity(previous, retry)


def test_retry_rejects_new_run_id():
    previous = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-001",
        revision=4,
    )

    retry = identity(
        task_id="task-001",
        run_id="run-002",
        attempt_id="attempt-002",
        revision=4,
    )

    with pytest.raises(ExecutionIdentityError):
        validate_retry_identity(previous, retry)


def test_retry_rejects_task_replacement():
    previous = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-001",
        revision=4,
    )

    retry = identity(
        task_id="task-002",
        run_id="run-001",
        attempt_id="attempt-002",
        revision=4,
    )

    with pytest.raises(ExecutionIdentityError):
        validate_retry_identity(previous, retry)


def test_retry_rejects_revision_replacement():
    previous = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-001",
        revision=4,
    )

    retry = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-002",
        revision=5,
    )

    with pytest.raises(ExecutionIdentityError):
        validate_retry_identity(previous, retry)


def test_retry_rejects_same_attempt_id():
    previous = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-001",
        revision=4,
    )

    retry = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-001",
        revision=4,
    )

    with pytest.raises(ExecutionIdentityError):
        validate_retry_identity(previous, retry)


# ============================================================
# FT-02-07 — collision semantics
# ============================================================

def test_run_id_collision_fails_closed():
    resolver = Resolver(
        runs={"run-001": "another-task"},
    )

    with pytest.raises(IdentityCollisionError):
        identity().validate(identity_resolver=resolver)


def test_attempt_id_collision_fails_closed():
    resolver = Resolver(
        attempts={"attempt-001": "another-run"},
    )

    with pytest.raises(IdentityCollisionError):
        identity().validate(identity_resolver=resolver)


def test_matching_existing_relationship_is_allowed():
    resolver = Resolver(
        runs={"run-001": "task-001"},
        attempts={"attempt-001": "run-001"},
    )

    identity().validate(identity_resolver=resolver)


# ============================================================
# FT-02-08 — session separation
# ============================================================

def test_session_id_is_distinct_from_other_identity_fields():
    first = identity(session_id="session-001")
    second = identity(session_id="session-002")

    assert first.task_id == second.task_id
    assert first.run_id == second.run_id
    assert first.attempt_id == second.attempt_id
    assert first.session_id != second.session_id


# ============================================================
# FT-02-09 — revision preservation
# ============================================================

def test_revision_is_preserved():
    item = identity(revision=17)

    restored = ExecutionIdentity.deserialize(item.serialize())

    assert restored.revision == 17


# ============================================================
# FT-02-10 — unknown identity is rejected
# ============================================================

def test_unknown_identity_field_is_rejected():
    payload = {
        "task_id": "task-001",
        "run_id": "run-001",
        "attempt_id": "attempt-001",
        "session_id": "session-001",
        "revision": 1,
        "schema_version": "1.0",
        "unknown_identity": "unexpected",
    }

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


# ============================================================
# FT-02-11 — security/prohibited fields are rejected
# ============================================================

@pytest.mark.parametrize(
    "field",
    [
        "password",
        "api_key",
        "apikey",
        "token",
        "secret",
        "private_key",
        "private_key_data",
        "recovery_code",
        "authorization",
        "authorization_context",
        "approval",
        "approval_context",
        "approval_decision",
        "execution_state",
        "state",
        "status",
    ],
)
def test_prohibited_field_is_rejected(field):
    payload = {
        "task_id": "task-001",
        "run_id": "run-001",
        "attempt_id": "attempt-001",
        "session_id": "session-001",
        "revision": 1,
        "schema_version": "1.0",
        field: "prohibited",
    }

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


# ============================================================
# FT-02-12 — schema and malformed input rejection
# ============================================================

@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not-a-mapping",
        123,
    ],
)
def test_non_mapping_payload_is_rejected(payload):
    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


def test_invalid_schema_version_is_rejected():
    payload = identity().to_dict()
    payload["schema_version"] = "999.0"

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


def test_invalid_revision_is_rejected():
    payload = identity().to_dict()
    payload["revision"] = -1

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    [
        "task_id",
        "run_id",
        "attempt_id",
        "session_id",
    ],
)
def test_empty_identity_id_is_rejected(field):
    payload = identity().to_dict()
    payload[field] = ""

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "attempt_id",
        "session_id",
    ],
)
def test_invalid_fnd02_identifier_domain_is_rejected(field):
    payload = identity().to_dict()
    payload[field] = "invalid identity"

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


# ============================================================
# FI-02-01 — missing required identity
# ============================================================

@pytest.mark.parametrize(
    "field",
    [
        "task_id",
        "run_id",
        "attempt_id",
        "revision",
    ],
)
def test_missing_required_identity_field_is_rejected(field):
    payload = identity().to_dict()
    del payload[field]

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


def test_session_id_is_optional_when_null():
    payload = identity().to_dict()
    payload["session_id"] = None

    restored = ExecutionIdentity.from_dict(payload)

    assert restored.session_id is None
    assert restored.to_dict()["session_id"] is None


def test_session_id_null_has_canonical_serialization():
    item = identity(session_id=None)

    assert item.serialize() == (
        '{"attempt_id":"attempt-001",'
        '"revision":1,'
        '"run_id":"run-001",'
        '"schema_version":"1.0",'
        '"session_id":null,'
        '"task_id":"task-001"}'
    )


# ============================================================
# FI-02-02 — malformed serialized identity
# ============================================================

def test_malformed_serialized_identity_is_rejected():
    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.deserialize("{not-valid-json")


# ============================================================
# FI-02-03 — authorization semantics are not accepted
# ============================================================

def test_authorization_field_is_not_part_of_identity():
    payload = identity().to_dict()
    payload["authorization"] = "approved"

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


# ============================================================
# FI-02-04 — execution-state semantics are not accepted
# ============================================================

def test_execution_state_field_is_not_part_of_identity():
    payload = identity().to_dict()
    payload["execution_state"] = "running"

    with pytest.raises(ExecutionIdentityError):
        ExecutionIdentity.from_dict(payload)


# ============================================================
# Historical preservation
# ============================================================

def test_historical_attempt_identity_is_preserved():
    historical = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-001",
        session_id="session-001",
        revision=2,
    )

    newer = identity(
        task_id="task-001",
        run_id="run-001",
        attempt_id="attempt-002",
        session_id="session-002",
        revision=2,
    )

    assert historical.attempt_id == "attempt-001"
    assert newer.attempt_id == "attempt-002"
    assert historical != newer


# ============================================================
# No authorization / execution-state semantics
# ============================================================

def test_identity_contains_no_authorization_or_state_fields():
    payload = identity().to_dict()

    assert "authorization" not in payload
    assert "approval" not in payload
    assert "state" not in payload
    assert "status" not in payload


# ============================================================
# JSON validity
# ============================================================

def test_serialized_identity_is_valid_json():
    item = identity()

    parsed = json.loads(item.serialize())

    assert parsed["task_id"] == "task-001"
    assert parsed["run_id"] == "run-001"
    assert parsed["attempt_id"] == "attempt-001"
