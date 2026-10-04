"""Passive, Hermes-independent verification result facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4


class VerificationCheckKind(str, Enum):
    """The category of a verification check."""

    BUILD = "BUILD"
    TEST = "TEST"
    LINT = "LINT"
    GIT_DIFF = "GIT_DIFF"
    POLICY = "POLICY"
    CUSTOM = "CUSTOM"


class VerificationCheckStatus(str, Enum):
    """Explicit status supplied for one verification check."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class VerificationVerdict(str, Enum):
    """Deterministic aggregate outcome of a verification result."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


class InvalidVerificationResult(ValueError):
    """Raised when a verification result violates a domain invariant."""


@dataclass(frozen=True, slots=True, kw_only=True)
class VerificationCheckResult:
    """An immutable check result referencing externally stored evidence."""

    kind: VerificationCheckKind
    status: VerificationCheckStatus
    required: bool
    summary: str
    check_id: str = field(default_factory=lambda: str(uuid4()))
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    name: str | None = None
    details: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string("check_id", self.check_id)
        if not isinstance(self.kind, VerificationCheckKind):
            raise TypeError("kind must be a VerificationCheckKind")
        if not isinstance(self.status, VerificationCheckStatus):
            raise TypeError("status must be a VerificationCheckStatus")
        if type(self.required) is not bool:
            raise TypeError("required must be a bool")
        _require_non_empty_string("summary", self.summary)

        for field_name in ("name", "details"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")

        if isinstance(self.evidence_ids, str):
            raise TypeError("evidence_ids must be an iterable of strings")
        try:
            evidence_ids = tuple(self.evidence_ids)
        except TypeError as exc:
            raise TypeError(
                "evidence_ids must be an iterable of strings"
            ) from exc
        for evidence_id in evidence_ids:
            _require_non_empty_string("evidence_id", evidence_id)
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class VerificationResult:
    """An immutable verification aggregate for one workflow attempt.

    The verdict is derived only from explicit check statuses. This entity does
    not execute checks, load evidence, or infer status from text or artifacts.
    """

    workflow_run_id: str
    attempt: int
    checks: tuple[VerificationCheckResult, ...]
    started_at: datetime
    finished_at: datetime
    verification_id: str = field(default_factory=lambda: str(uuid4()))
    verdict: VerificationVerdict = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty_string("verification_id", self.verification_id)
        _require_non_empty_string("workflow_run_id", self.workflow_run_id)
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be an integer greater than zero")

        if isinstance(self.checks, VerificationCheckResult):
            raise TypeError(
                "checks must be an iterable of VerificationCheckResult"
            )
        try:
            checks = tuple(self.checks)
        except TypeError as exc:
            raise TypeError(
                "checks must be an iterable of VerificationCheckResult"
            ) from exc
        if not checks:
            raise InvalidVerificationResult(
                "verification result must contain at least one check"
            )
        if not all(isinstance(check, VerificationCheckResult) for check in checks):
            raise TypeError("checks must contain only VerificationCheckResult")
        object.__setattr__(self, "checks", checks)

        _require_aware_datetime("started_at", self.started_at)
        _require_aware_datetime("finished_at", self.finished_at)
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")

        object.__setattr__(self, "verdict", _derive_verdict(checks))

    @property
    def duration(self) -> timedelta:
        """Return the deterministic elapsed time of the verification run."""

        return self.finished_at - self.started_at


def _derive_verdict(
    checks: tuple[VerificationCheckResult, ...],
) -> VerificationVerdict:
    if any(check.status is VerificationCheckStatus.ERROR for check in checks):
        return VerificationVerdict.ERROR
    if any(check.status is VerificationCheckStatus.FAIL for check in checks):
        return VerificationVerdict.FAIL
    if any(
        check.required and check.status is VerificationCheckStatus.SKIPPED
        for check in checks
    ):
        return VerificationVerdict.FAIL
    return VerificationVerdict.PASS


def _require_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_aware_datetime(name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
