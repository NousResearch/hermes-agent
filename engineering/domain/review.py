"""Passive, Hermes-independent engineering review facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4


class ReviewVerdict(str, Enum):
    """Deterministic aggregate outcome of an engineering review."""

    PASS = "PASS"
    NEEDS_WORK = "NEEDS_WORK"
    BLOCKED = "BLOCKED"


class ReviewSeverity(str, Enum):
    """Severity of one structured review finding."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ReviewCategory(str, Enum):
    """Engineering concern represented by a review finding."""

    CORRECTNESS = "CORRECTNESS"
    SECURITY = "SECURITY"
    COMPATIBILITY = "COMPATIBILITY"
    ARCHITECTURE = "ARCHITECTURE"
    MAINTAINABILITY = "MAINTAINABILITY"
    TESTING = "TESTING"
    ERROR_HANDLING = "ERROR_HANDLING"
    LOGGING = "LOGGING"
    SCOPE = "SCOPE"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewFinding:
    """An immutable structured observation produced by a reviewer."""

    category: ReviewCategory
    severity: ReviewSeverity
    message: str
    finding_id: str = field(default_factory=lambda: str(uuid4()))
    file_path: str | None = None
    line: int | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    recommendation: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string("finding_id", self.finding_id)
        if not isinstance(self.category, ReviewCategory):
            raise TypeError("category must be a ReviewCategory")
        if not isinstance(self.severity, ReviewSeverity):
            raise TypeError("severity must be a ReviewSeverity")
        _require_non_empty_string("message", self.message)

        if self.file_path is not None:
            _require_non_empty_string("file_path", self.file_path)
        if self.line is not None and (
            type(self.line) is not int or self.line < 1
        ):
            raise ValueError("line must be a positive integer when supplied")
        if self.recommendation is not None and not isinstance(
            self.recommendation, str
        ):
            raise TypeError("recommendation must be a string or None")

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
class ReviewResult:
    """An immutable aggregate for a review performed elsewhere.

    Unlike verification, an empty review is valid and deterministically passes.
    The verdict depends only on structured finding severity.
    """

    workflow_run_id: str
    attempt: int
    findings: tuple[ReviewFinding, ...]
    reviewer: str
    started_at: datetime
    finished_at: datetime
    review_id: str = field(default_factory=lambda: str(uuid4()))
    summary: str | None = None
    verdict: ReviewVerdict = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty_string("review_id", self.review_id)
        _require_non_empty_string("workflow_run_id", self.workflow_run_id)
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be an integer greater than zero")
        _require_non_empty_string("reviewer", self.reviewer)
        if self.summary is not None and not isinstance(self.summary, str):
            raise TypeError("summary must be a string or None")

        if isinstance(self.findings, ReviewFinding):
            raise TypeError("findings must be an iterable of ReviewFinding")
        try:
            findings = tuple(self.findings)
        except TypeError as exc:
            raise TypeError(
                "findings must be an iterable of ReviewFinding"
            ) from exc
        if not all(isinstance(finding, ReviewFinding) for finding in findings):
            raise TypeError("findings must contain only ReviewFinding")
        object.__setattr__(self, "findings", findings)

        _require_aware_datetime("started_at", self.started_at)
        _require_aware_datetime("finished_at", self.finished_at)
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")

        object.__setattr__(self, "verdict", _derive_verdict(findings))

    @property
    def duration(self) -> timedelta:
        """Return the deterministic elapsed time of the review."""

        return self.finished_at - self.started_at


def _derive_verdict(findings: tuple[ReviewFinding, ...]) -> ReviewVerdict:
    if any(finding.severity is ReviewSeverity.CRITICAL for finding in findings):
        return ReviewVerdict.BLOCKED
    if any(finding.severity is ReviewSeverity.ERROR for finding in findings):
        return ReviewVerdict.NEEDS_WORK
    return ReviewVerdict.PASS


def _require_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_aware_datetime(name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
