import hashlib
import json

import pytest

from clicksmith.foundation.evidence_model import (
    ERROR_CODES,
    EvidenceProvenance,
    EvidenceRecord,
    InvalidEvidenceInputError,
    ImmutableEvidenceMutationError,
    InvalidIdentityError,
    InvalidIntegrityError,
    UnsupportedEvidenceTypeError,
    AmbiguousEvidenceError,
)
from clicksmith.foundation.execution_identity import ExecutionIdentity


def identity(attempt="attempt-001"):
    return ExecutionIdentity(
        task_id="task-001", run_id="run-001", attempt_id=attempt,
        session_id="session-001", revision=1,
    )


def provenance():
    return EvidenceProvenance(
        source_boundary="clicksmith.execution",
        source_type="unit-test",
        creation_context="test-context",
    )


def record(**overrides):
    values = dict(
        evidence_id="EV-001",
        evidence_type="RESULT",
        provenance=provenance(),
        identity_applicability="EXECUTION",
        identity=identity(),
        metadata={"z": 2, "a": 1},
        reference="result://001",
    )
    values.update(overrides)
    return EvidenceRecord(**values)


def test_error_taxonomy_is_exact():
    assert ERROR_CODES == {
        "INVALID_IDENTITY", "INVALID_EVIDENCE_INPUT", "INVALID_PROVENANCE",
        "INVALID_INTEGRITY", "IMMUTABLE_EVIDENCE_MUTATION", "SCHEMA_ERROR",
        "AMBIGUOUS_EVIDENCE", "UNSUPPORTED_EVIDENCE_TYPE", "EVIDENCE_SERIALIZATION_ERROR",
    }


def test_execution_evidence_requires_identity():
    with pytest.raises(InvalidIdentityError):
        record(identity=None).validate()


def test_non_execution_identity_applicability_is_explicit():
    item = record(identity_applicability="NON_EXECUTION", identity=None)
    item.validate()
    assert "identity" not in item.to_dict()


def test_non_execution_with_identity_is_rejected():
    with pytest.raises(AmbiguousEvidenceError):
        record(identity_applicability="NON_EXECUTION").validate()


def test_evidence_id_is_explicit_and_strict():
    with pytest.raises(Exception):
        record(evidence_id="").validate()
    with pytest.raises(Exception):
        record(evidence_id="EV 001").validate()


def test_no_silent_id_generation():
    assert record().evidence_id == "EV-001"


def test_canonical_serialization_is_deterministic():
    item = record()
    first = item.serialize()
    second = item.serialize()
    assert first == second
    assert json.loads(first)["metadata"] == {"a": 1, "z": 2}


def test_sha256_excludes_integrity_field_and_is_reproducible():
    item = record()
    digest = item.compute_sha256()
    expected = hashlib.sha256(item.canonical_bytes()).hexdigest()
    assert digest == expected
    with_hash = item.with_integrity()
    assert with_hash.integrity_sha256 == digest
    assert with_hash.compute_sha256() == digest


def test_round_trip():
    item = record().with_integrity()
    assert EvidenceRecord.deserialize(item.serialize()) == item


def test_timestamp_is_optional_and_not_semantic_identity():
    a = record()
    b = record(timestamp="2026-09-09T07:00:00Z")
    assert a.evidence_id == b.evidence_id
    assert a.identity == b.identity
    assert a.compute_sha256() != b.compute_sha256()


def test_raw_payload_is_optional_and_bounded():
    record().validate()
    record(payload={"safe": [1, True, "x"]}).validate()
    with pytest.raises(Exception):
        record(payload={"x": "x" * 20000}).validate()


def test_unsupported_type_rejected():
    with pytest.raises(UnsupportedEvidenceTypeError):
        record(evidence_type="git").validate()


def test_supersedes_stays_same_execution_identity():
    previous = record(evidence_id="EV-001")
    corrected = record(evidence_id="EV-002", supersedes="EV-001")
    corrected.assert_can_supersede(previous)


def test_supersedes_cannot_cross_attempt():
    previous = record(evidence_id="EV-001", identity=identity("attempt-001"))
    corrected = record(
        evidence_id="EV-002", supersedes="EV-001", identity=identity("attempt-002")
    )
    with pytest.raises(ImmutableEvidenceMutationError):
        corrected.assert_can_supersede(previous)


def test_supersedes_cannot_cross_domain():
    previous = record(evidence_id="EV-001", evidence_type="RESULT")
    corrected = record(evidence_id="EV-002", evidence_type="FAILURE", supersedes="EV-001")
    with pytest.raises(ImmutableEvidenceMutationError):
        corrected.assert_can_supersede(previous)


def test_historical_record_is_frozen():
    item = record()
    with pytest.raises(Exception):
        item.evidence_id = "EV-002"


def test_evaluation_errors_are_not_evidence_records():
    with pytest.raises(InvalidEvidenceInputError):
        record(metadata={"bad": object()}).validate()


def test_integrity_tampering_rejected():
    item = record().with_integrity()
    with pytest.raises(InvalidIntegrityError):
        EvidenceRecord.deserialize(item.serialize().replace('result://001', 'result://002'))
