"""Types for the disabled-by-default Local Muncho runtime guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


@dataclass(frozen=True)
class RuntimeContext:
    lane: str | None = None
    session_id: str = ""
    platform: str = ""
    user_id: str = ""
    chat_id: str = ""
    thread_id: str = ""
    message_id: str = ""
    profile: str = ""
    case_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeContext:
    scope: str
    text: str
    version: str = ""
    records: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


@dataclass(frozen=True)
class LeaseState:
    lease_owner: str
    active_runtime: str
    expires_at: float | datetime | None
    flags: Sequence[str] = field(default_factory=tuple)
    approval_classes: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return True
        expiry = self.expires_at
        if isinstance(expiry, datetime):
            expiry = expiry.timestamp()
        return float(expiry) <= (utc_ts() if now is None else now)


@dataclass(frozen=True)
class LeaseAssertion:
    allowed: bool
    reason: str = ""
    lease: LeaseState | None = None


@dataclass(frozen=True)
class HeartbeatPayload:
    runtime_id: str
    runtime_kind: str
    status: str
    event_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    emitted_at: float = field(default_factory=utc_ts)


@dataclass(frozen=True)
class HeartbeatResult:
    allowed: bool
    reason: str = ""
    lease: LeaseState | None = None


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reason: str = ""
    message: str = ""
    code: str = ""
    replacement_text: str | None = None

    @classmethod
    def allow(cls, reason: str = "allowed") -> "GuardDecision":
        return cls(True, reason=reason, code="allowed")

    @classmethod
    def block(
        cls,
        reason: str,
        *,
        message: str | None = None,
        code: str = "blocked",
        replacement_text: str | None = None,
    ) -> "GuardDecision":
        return cls(
            False,
            reason=reason,
            message=message or reason,
            code=code,
            replacement_text=replacement_text,
        )


@dataclass(frozen=True)
class EvidenceValidationResult:
    allowed: bool
    reason: str = ""
    missing_fields: Sequence[str] = field(default_factory=tuple)
    evidence_gaps: Sequence[str] = field(default_factory=tuple)
    replacement_text: str | None = None


@dataclass(frozen=True)
class ToolEvidence:
    tool_name: str
    result: Any
    tool_call_id: str = ""
    success: bool = False
    durable_ref: str = ""


@dataclass(frozen=True)
class VisibleSendIntent:
    kind: str
    platform: str = ""
    chat_id: str = ""
    thread_id: str = ""
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisibleSendDecision:
    allowed: bool
    reason: str = ""
    replacement_text: str | None = None


@dataclass(frozen=True)
class WorkerContract:
    goals: Sequence[str] = field(default_factory=tuple)
    task_count: int = 0
    source_task_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LockRequest:
    key: str
    action: str = ""


@dataclass(frozen=True)
class LockHandle:
    key: str
    token: str
    acquired: bool
    expires_at: float | None = None


@dataclass(frozen=True)
class RuntimeEvent:
    event_type: str
    status: str
    context: RuntimeContext
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=utc_ts)


@dataclass(frozen=True)
class AuditEvent:
    action: str
    outcome: str
    context: RuntimeContext
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=utc_ts)


@dataclass(frozen=True)
class SupportCasePatch:
    case_id: str
    fields: Mapping[str, Any]
    context: RuntimeContext


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    session_key: str
    choice: str
    context: RuntimeContext
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodexTaskCreate:
    title: str
    context: RuntimeContext
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodexTaskPatch:
    status: str | None = None
    final_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
