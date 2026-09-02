"""Passive, Hermes-independent engineering evidence facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4


class EvidenceKind(str, Enum):
    """The engineering activity that produced an evidence fact."""

    BUILD = "BUILD"
    TEST = "TEST"
    LINT = "LINT"
    GIT_DIFF = "GIT_DIFF"
    POLICY = "POLICY"
    PROJECT_INSPECTION = "PROJECT_INSPECTION"
    COMMAND = "COMMAND"
    OTHER = "OTHER"


class EvidenceStatus(str, Enum):
    """Explicit producer-supplied evidence status, not workflow state."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True, kw_only=True)
class Evidence:
    """An immutable, auditable fact produced by an engineering check.

    This entity is intentionally passive: it performs no execution, inspection,
    status inference, artifact loading, or secret redaction. ``status`` is an
    explicit fact supplied by the trusted producer; text and exit codes never
    change it. Artifact references identify external data without embedding it.
    """

    workflow_run_id: str
    attempt: int
    kind: EvidenceKind
    status: EvidenceStatus
    producer: str
    summary: str
    started_at: datetime
    finished_at: datetime
    evidence_id: str = field(default_factory=lambda: str(uuid4()))
    command: str | None = None
    cwd: str | None = None
    execution_backend: str | None = None
    exit_code: int | None = None
    stdout_summary: str | None = None
    stderr_summary: str | None = None
    artifact_references: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_non_empty_string("evidence_id", self.evidence_id)
        _require_non_empty_string("workflow_run_id", self.workflow_run_id)
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be an integer greater than zero")
        if not isinstance(self.kind, EvidenceKind):
            raise TypeError("kind must be an EvidenceKind")
        if not isinstance(self.status, EvidenceStatus):
            raise TypeError("status must be an EvidenceStatus")
        _require_non_empty_string("producer", self.producer)
        _require_non_empty_string("summary", self.summary)

        _require_aware_datetime("started_at", self.started_at)
        _require_aware_datetime("finished_at", self.finished_at)
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")

        for name in (
            "command",
            "cwd",
            "execution_backend",
            "stdout_summary",
            "stderr_summary",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise TypeError("exit_code must be an integer or None")

        execution_context = (self.command, self.cwd, self.execution_backend)
        if any(value is not None for value in execution_context) and not all(
            value is not None and value.strip() for value in execution_context
        ):
            raise ValueError(
                "command-backed evidence requires command, cwd, and "
                "execution_backend"
            )
        if self.exit_code is not None and self.command is None:
            raise ValueError("exit_code requires command-backed evidence")

        if isinstance(self.artifact_references, str):
            raise TypeError("artifact_references must be an iterable of strings")
        try:
            references = tuple(self.artifact_references)
        except TypeError as exc:
            raise TypeError(
                "artifact_references must be an iterable of strings"
            ) from exc
        for reference in references:
            _require_non_empty_string("artifact reference", reference)
        object.__setattr__(self, "artifact_references", references)

    @property
    def duration(self) -> timedelta:
        """Return the deterministic elapsed time represented by this fact."""

        return self.finished_at - self.started_at


def _require_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_aware_datetime(name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
