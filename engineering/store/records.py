"""Explicit, versioned persistence records for Engineering domain facts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TypedDict

from engineering.domain import (
    Evidence,
    EvidenceKind,
    EvidenceStatus,
    ReviewCategory,
    ReviewFinding,
    ReviewResult,
    ReviewSeverity,
    VerificationCheckKind,
    VerificationCheckResult,
    VerificationCheckStatus,
    VerificationResult,
    WorkflowRun,
    WorkflowState,
)


SCHEMA_VERSION = 1


class PersistenceRecordError(ValueError):
    """Base error for invalid Engineering persistence records."""


class InvalidPersistenceRecord(PersistenceRecordError):
    """Raised when a record cannot reconstruct a valid domain object."""


class UnsupportedSchemaVersion(PersistenceRecordError):
    """Raised when a record uses an unsupported schema version."""

    def __init__(self, version: object) -> None:
        self.version = version
        super().__init__(
            f"unsupported engineering record schema_version: {version!r}"
        )


class WorkflowRecord(TypedDict):
    schema_version: int
    workflow_run_id: str
    state: str
    created_at: str
    updated_at: str
    attempt: int
    max_attempts: int


class EvidenceRecord(TypedDict):
    schema_version: int
    evidence_id: str
    workflow_run_id: str
    attempt: int
    kind: str
    status: str
    producer: str
    summary: str
    started_at: str
    finished_at: str
    command: str | None
    cwd: str | None
    execution_backend: str | None
    exit_code: int | None
    stdout_summary: str | None
    stderr_summary: str | None
    artifact_references: list[str]


class VerificationCheckRecord(TypedDict):
    check_id: str
    kind: str
    status: str
    required: bool
    summary: str
    evidence_ids: list[str]
    name: str | None
    details: str | None


class VerificationRecord(TypedDict):
    schema_version: int
    verification_id: str
    workflow_run_id: str
    attempt: int
    checks: list[VerificationCheckRecord]
    started_at: str
    finished_at: str


class ReviewFindingRecord(TypedDict):
    finding_id: str
    category: str
    severity: str
    message: str
    file_path: str | None
    line: int | None
    evidence_ids: list[str]
    recommendation: str | None


class ReviewRecord(TypedDict):
    schema_version: int
    review_id: str
    workflow_run_id: str
    attempt: int
    findings: list[ReviewFindingRecord]
    reviewer: str
    started_at: str
    finished_at: str
    summary: str | None


def workflow_to_record(workflow: WorkflowRun) -> WorkflowRecord:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_run_id": workflow.workflow_run_id,
        "state": workflow.state.value,
        "created_at": _datetime_to_string(workflow.created_at),
        "updated_at": _datetime_to_string(workflow.updated_at),
        "attempt": workflow.attempt,
        "max_attempts": workflow.max_attempts,
    }


def workflow_from_record(record: Mapping[str, object]) -> WorkflowRun:
    _validate_schema_version(record)
    try:
        return WorkflowRun.restore(
            workflow_run_id=_required_string(record, "workflow_run_id"),
            state=WorkflowState(_required_string(record, "state")),
            created_at=_datetime_from_value(record["created_at"]),
            updated_at=_datetime_from_value(record["updated_at"]),
            attempt=_required_int(record, "attempt"),
            max_attempts=_required_int(record, "max_attempts"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidPersistenceRecord("invalid WorkflowRecord") from exc


def evidence_to_record(evidence: Evidence) -> EvidenceRecord:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": evidence.evidence_id,
        "workflow_run_id": evidence.workflow_run_id,
        "attempt": evidence.attempt,
        "kind": evidence.kind.value,
        "status": evidence.status.value,
        "producer": evidence.producer,
        "summary": evidence.summary,
        "started_at": _datetime_to_string(evidence.started_at),
        "finished_at": _datetime_to_string(evidence.finished_at),
        "command": evidence.command,
        "cwd": evidence.cwd,
        "execution_backend": evidence.execution_backend,
        "exit_code": evidence.exit_code,
        "stdout_summary": evidence.stdout_summary,
        "stderr_summary": evidence.stderr_summary,
        "artifact_references": list(evidence.artifact_references),
    }


def evidence_from_record(record: Mapping[str, object]) -> Evidence:
    _validate_schema_version(record)
    try:
        return Evidence(
            evidence_id=_required_string(record, "evidence_id"),
            workflow_run_id=_required_string(record, "workflow_run_id"),
            attempt=_required_int(record, "attempt"),
            kind=EvidenceKind(_required_string(record, "kind")),
            status=EvidenceStatus(_required_string(record, "status")),
            producer=_required_string(record, "producer"),
            summary=_required_string(record, "summary"),
            started_at=_datetime_from_value(record["started_at"]),
            finished_at=_datetime_from_value(record["finished_at"]),
            command=_optional_string(record, "command"),
            cwd=_optional_string(record, "cwd"),
            execution_backend=_optional_string(record, "execution_backend"),
            exit_code=_optional_int(record, "exit_code"),
            stdout_summary=_optional_string(record, "stdout_summary"),
            stderr_summary=_optional_string(record, "stderr_summary"),
            artifact_references=_string_list(record, "artifact_references"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidPersistenceRecord("invalid EvidenceRecord") from exc


def verification_to_record(result: VerificationResult) -> VerificationRecord:
    return {
        "schema_version": SCHEMA_VERSION,
        "verification_id": result.verification_id,
        "workflow_run_id": result.workflow_run_id,
        "attempt": result.attempt,
        "checks": [
            {
                "check_id": check.check_id,
                "kind": check.kind.value,
                "status": check.status.value,
                "required": check.required,
                "summary": check.summary,
                "evidence_ids": list(check.evidence_ids),
                "name": check.name,
                "details": check.details,
            }
            for check in result.checks
        ],
        "started_at": _datetime_to_string(result.started_at),
        "finished_at": _datetime_to_string(result.finished_at),
    }


def verification_from_record(
    record: Mapping[str, object],
) -> VerificationResult:
    _validate_schema_version(record)
    try:
        checks = tuple(
            VerificationCheckResult(
                check_id=_required_string(check, "check_id"),
                kind=VerificationCheckKind(
                    _required_string(check, "kind")
                ),
                status=VerificationCheckStatus(
                    _required_string(check, "status")
                ),
                required=_required_bool(check, "required"),
                summary=_required_string(check, "summary"),
                evidence_ids=_string_list(check, "evidence_ids"),
                name=_optional_string(check, "name"),
                details=_optional_string(check, "details"),
            )
            for check in _mapping_list(record, "checks")
        )
        return VerificationResult(
            verification_id=_required_string(record, "verification_id"),
            workflow_run_id=_required_string(record, "workflow_run_id"),
            attempt=_required_int(record, "attempt"),
            checks=checks,
            started_at=_datetime_from_value(record["started_at"]),
            finished_at=_datetime_from_value(record["finished_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidPersistenceRecord("invalid VerificationRecord") from exc


def review_to_record(result: ReviewResult) -> ReviewRecord:
    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": result.review_id,
        "workflow_run_id": result.workflow_run_id,
        "attempt": result.attempt,
        "findings": [
            {
                "finding_id": finding.finding_id,
                "category": finding.category.value,
                "severity": finding.severity.value,
                "message": finding.message,
                "file_path": finding.file_path,
                "line": finding.line,
                "evidence_ids": list(finding.evidence_ids),
                "recommendation": finding.recommendation,
            }
            for finding in result.findings
        ],
        "reviewer": result.reviewer,
        "started_at": _datetime_to_string(result.started_at),
        "finished_at": _datetime_to_string(result.finished_at),
        "summary": result.summary,
    }


def review_from_record(record: Mapping[str, object]) -> ReviewResult:
    _validate_schema_version(record)
    try:
        findings = tuple(
            ReviewFinding(
                finding_id=_required_string(finding, "finding_id"),
                category=ReviewCategory(
                    _required_string(finding, "category")
                ),
                severity=ReviewSeverity(
                    _required_string(finding, "severity")
                ),
                message=_required_string(finding, "message"),
                file_path=_optional_string(finding, "file_path"),
                line=_optional_int(finding, "line"),
                evidence_ids=_string_list(finding, "evidence_ids"),
                recommendation=_optional_string(finding, "recommendation"),
            )
            for finding in _mapping_list(record, "findings")
        )
        return ReviewResult(
            review_id=_required_string(record, "review_id"),
            workflow_run_id=_required_string(record, "workflow_run_id"),
            attempt=_required_int(record, "attempt"),
            findings=findings,
            reviewer=_required_string(record, "reviewer"),
            started_at=_datetime_from_value(record["started_at"]),
            finished_at=_datetime_from_value(record["finished_at"]),
            summary=_optional_string(record, "summary"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidPersistenceRecord("invalid ReviewRecord") from exc


def _validate_schema_version(record: Mapping[str, object]) -> None:
    version = record.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(version)


def _datetime_to_string(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("persisted datetimes must be timezone-aware")
    return value.isoformat()


def _datetime_from_value(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("persisted datetime must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("persisted datetimes must be timezone-aware")
    return parsed


def _required_string(record: Mapping[str, object], key: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_string(record: Mapping[str, object], key: str) -> str | None:
    value = record[key]
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _required_int(record: Mapping[str, object], key: str) -> int:
    value = record[key]
    if type(value) is not int:
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_int(record: Mapping[str, object], key: str) -> int | None:
    value = record[key]
    if value is not None and type(value) is not int:
        raise TypeError(f"{key} must be an integer or null")
    return value


def _required_bool(record: Mapping[str, object], key: str) -> bool:
    value = record[key]
    if type(value) is not bool:
        raise TypeError(f"{key} must be a boolean")
    return value


def _string_list(record: Mapping[str, object], key: str) -> list[str]:
    value = record[key]
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise TypeError(f"{key} must be an array of strings")
    return value


def _mapping_list(
    record: Mapping[str, object], key: str
) -> list[Mapping[str, object]]:
    value = record[key]
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{key} must be an array of records")
    return value
