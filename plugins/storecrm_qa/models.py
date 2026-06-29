"""Typed StoreCRM QA control-plane models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_ENGINEER = "needs_engineer"


class CaseStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_ENGINEER = "needs_engineer"
    EXHAUSTED = "exhausted"


class RunnerOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_ENGINEER = "needs_engineer"


@dataclass(frozen=True)
class PolicyMetadata:
    tenant_id: str
    store_id: str
    risk: str = "low"
    allowed_operations: tuple[str, ...] = ("read",)

    def validate(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id is required")
        if not self.store_id.strip():
            raise ValueError("store_id is required")
        if self.risk not in {"low", "medium", "high"}:
            raise ValueError("risk must be low, medium, or high")
        if not self.allowed_operations:
            raise ValueError("at least one allowed operation is required")
        invalid = [op for op in self.allowed_operations if not op.strip()]
        if invalid:
            raise ValueError("allowed operations must be non-empty strings")


@dataclass(frozen=True)
class QAJob:
    id: int
    name: str
    status: JobStatus
    metadata: PolicyMetadata
    max_attempts: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class QACase:
    id: int
    job_id: int
    name: str
    status: CaseStatus
    input_payload: dict[str, Any]
    attempts: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    metadata: PolicyMetadata
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Lease:
    case_id: int
    job_id: int
    owner: str
    expires_at: datetime
    attempt: int


@dataclass(frozen=True)
class Evidence:
    kind: str
    summary: str
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerResult:
    outcome: RunnerOutcome
    summary: str
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class FinalReportSummary:
    job_id: int
    status: JobStatus
    total_cases: int
    passed: int
    failed: int
    needs_engineer: int
    exhausted: int
    evidence_count: int
