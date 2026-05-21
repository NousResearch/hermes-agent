"""Durable state objects for the Jeeves swarm operator.

Sprint 1 intentionally keeps this module dependency-free: small dataclasses,
explicit JSON conversion, and conservative defaults suitable for shadow mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid5, NAMESPACE_URL


class JobStatus(str, Enum):
    RECEIVED = "received"
    TRIAGING = "triaging"
    PLANNING = "planning"
    AWAITING_PERMISSION = "awaiting_permission"
    RUNNING = "running"
    VERIFYING = "verifying"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    BLOCKED = "blocked"
    AWAITING_PERMISSION = "awaiting_permission"
    RUNNING = "running"
    NEEDS_REVIEW = "needs_review"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


_JOB_STATUSES = {item.value for item in JobStatus}
_TASK_STATUSES = {item.value for item in TaskStatus}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_job_id(*, original_request: str, platform: str = "", user_id: str = "", chat_id: str = "", created_at: str = "") -> str:
    seed = "\x1f".join([platform or "", user_id or "", chat_id or "", created_at or "", original_request or ""])
    return f"swarm_{uuid5(NAMESPACE_URL, seed).hex[:20]}"


def _copy_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass
class AuditEvent:
    event_type: str
    message: str = ""
    timestamp: str = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": _copy_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        return cls(
            event_type=str(data.get("event_type") or "event"),
            message=str(data.get("message") or ""),
            timestamp=str(data.get("timestamp") or _now()),
            metadata=_copy_dict(data.get("metadata")),
        )


@dataclass
class PermissionGrant:
    permission_id: str
    description: str
    status: str = "requested"
    scope: Dict[str, Any] = field(default_factory=dict)
    requested_at: str = field(default_factory=_now)
    decided_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "permission_id": self.permission_id,
            "description": self.description,
            "status": self.status,
            "scope": _copy_dict(self.scope),
            "requested_at": self.requested_at,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionGrant":
        return cls(
            permission_id=str(data.get("permission_id") or data.get("id") or "permission"),
            description=str(data.get("description") or ""),
            status=str(data.get("status") or "requested"),
            scope=_copy_dict(data.get("scope")),
            requested_at=str(data.get("requested_at") or _now()),
            decided_at=data.get("decided_at"),
        )


@dataclass
class EvalResult:
    name: str
    passed: bool
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "metadata": _copy_dict(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalResult":
        return cls(
            name=str(data.get("name") or "eval"),
            passed=bool(data.get("passed", False)),
            details=str(data.get("details") or ""),
            metadata=_copy_dict(data.get("metadata")),
            timestamp=str(data.get("timestamp") or _now()),
        )


@dataclass
class SwarmTask:
    task_id: str
    title: str
    description: str = ""
    status: str = TaskStatus.QUEUED.value
    mode: str = "direct"
    toolsets: List[str] = field(default_factory=list)
    permission_required: bool = False
    result: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _TASK_STATUSES:
            raise ValueError(f"unknown task status: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "mode": self.mode,
            "toolsets": list(self.toolsets),
            "permission_required": self.permission_required,
            "result": _copy_dict(self.result) if self.result is not None else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": _copy_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmTask":
        return cls(
            task_id=str(data.get("task_id") or data.get("id") or "task"),
            title=str(data.get("title") or ""),
            description=str(data.get("description") or ""),
            status=str(data.get("status") or TaskStatus.QUEUED.value),
            mode=str(data.get("mode") or "direct"),
            toolsets=list(data.get("toolsets") or []),
            permission_required=bool(data.get("permission_required", False)),
            result=_copy_dict(data.get("result")) if isinstance(data.get("result"), dict) else data.get("result"),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or data.get("created_at") or _now()),
            metadata=_copy_dict(data.get("metadata")),
        )


@dataclass
class EvidenceRequirement:
    kind: str
    description: str
    required: bool = True
    source: str = "parent"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "required": self.required,
            "source": self.source,
            "metadata": _copy_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRequirement":
        return cls(
            kind=str(data.get("kind") or "artifact"),
            description=str(data.get("description") or ""),
            required=bool(data.get("required", True)),
            source=str(data.get("source") or "parent"),
            metadata=_copy_dict(data.get("metadata")),
        )


@dataclass
class RoutingTelemetry:
    complexity_risk: int = 0
    context_pressure: int = 0
    illusion_risk: int = 0
    required_scaffold: str = "none"
    weak_output_risk: bool = False
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity_risk": self.complexity_risk,
            "context_pressure": self.context_pressure,
            "illusion_risk": self.illusion_risk,
            "required_scaffold": self.required_scaffold,
            "weak_output_risk": self.weak_output_risk,
            "signals": _copy_dict(self.signals),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RoutingTelemetry":
        if not isinstance(data, dict):
            return cls()
        return cls(
            complexity_risk=int(data.get("complexity_risk") or 0),
            context_pressure=int(data.get("context_pressure") or 0),
            illusion_risk=int(data.get("illusion_risk") or 0),
            required_scaffold=str(data.get("required_scaffold") or "none"),
            weak_output_risk=bool(data.get("weak_output_risk", False)),
            signals=_copy_dict(data.get("signals")),
        )


@dataclass
class RoutingPlan:
    mode: str
    reason: str
    suggested_tasks: List[Dict[str, Any]] = field(default_factory=list)
    permission_requests: List[PermissionGrant] = field(default_factory=list)
    verification_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_requirements: List[EvidenceRequirement] = field(default_factory=list)
    routing_telemetry: RoutingTelemetry = field(default_factory=RoutingTelemetry)
    panel_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        evidence_items = [
            item.to_dict() if hasattr(item, "to_dict") else EvidenceRequirement.from_dict(item).to_dict()
            for item in self.evidence_requirements
            if isinstance(item, dict) or hasattr(item, "to_dict")
        ]
        telemetry = (
            self.routing_telemetry.to_dict()
            if hasattr(self.routing_telemetry, "to_dict")
            else RoutingTelemetry.from_dict(self.routing_telemetry).to_dict()
        )
        return {
            "mode": self.mode,
            "reason": self.reason,
            "suggested_tasks": [dict(item) for item in self.suggested_tasks],
            "permission_requests": [item.to_dict() for item in self.permission_requests],
            "verification_required": self.verification_required,
            "metadata": _copy_dict(self.metadata),
            "evidence_requirements": evidence_items,
            "routing_telemetry": telemetry,
            "panel_policy": _copy_dict(self.panel_policy),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["RoutingPlan"]:
        if not data:
            return None
        return cls(
            mode=str(data.get("mode") or "direct"),
            reason=str(data.get("reason") or ""),
            suggested_tasks=[dict(item) for item in data.get("suggested_tasks") or [] if isinstance(item, dict)],
            permission_requests=[PermissionGrant.from_dict(item) for item in data.get("permission_requests") or [] if isinstance(item, dict)],
            verification_required=bool(data.get("verification_required", False)),
            metadata=_copy_dict(data.get("metadata")),
            evidence_requirements=[EvidenceRequirement.from_dict(item) for item in data.get("evidence_requirements") or [] if isinstance(item, dict)],
            routing_telemetry=RoutingTelemetry.from_dict(data.get("routing_telemetry")),
            panel_policy=_copy_dict(data.get("panel_policy")),
        )


@dataclass
class SwarmJob:
    job_id: str
    original_request: str
    status: str = JobStatus.RECEIVED.value
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    platform: str = ""
    user_id: str = ""
    chat_id: str = ""
    session_id: str = ""
    routing_plan: Optional[RoutingPlan] = None
    tasks: List[SwarmTask] = field(default_factory=list)
    permissions: List[PermissionGrant] = field(default_factory=list)
    evals: List[EvalResult] = field(default_factory=list)
    audit: List[AuditEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _JOB_STATUSES:
            raise ValueError(f"unknown job status: {self.status}")

    @classmethod
    def create(
        cls,
        original_request: str,
        *,
        platform: str = "",
        user_id: str = "",
        chat_id: str = "",
        session_id: str = "",
        job_id: Optional[str] = None,
        created_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SwarmJob":
        stamp = created_at or _now()
        resolved_id = job_id or _stable_job_id(
            original_request=original_request,
            platform=platform,
            user_id=user_id,
            chat_id=chat_id,
            created_at=stamp,
        )
        return cls(
            job_id=resolved_id,
            original_request=original_request,
            status=JobStatus.RECEIVED.value,
            created_at=stamp,
            updated_at=stamp,
            platform=platform,
            user_id=user_id,
            chat_id=chat_id,
            session_id=session_id,
            metadata=_copy_dict(metadata),
        )

    def add_task(self, title: str, description: str = "", **kwargs: Any) -> SwarmTask:
        task_id = kwargs.pop("task_id", f"{self.job_id}_task_{len(self.tasks) + 1}")
        task = SwarmTask(task_id=task_id, title=title, description=description, **kwargs)
        self.tasks.append(task)
        self.updated_at = _now()
        self.audit.append(AuditEvent("task_added", f"Added task: {title}", metadata={"task_id": task.task_id}))
        return task

    def transition(self, status: str, message: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        if status not in _JOB_STATUSES:
            raise ValueError(f"unknown job status: {status}")
        old_status = self.status
        self.status = status
        self.updated_at = _now()
        self.audit.append(
            AuditEvent(
                "status_changed",
                message or f"Job status changed from {old_status} to {status}",
                metadata={"from": old_status, "to": status, **_copy_dict(metadata)},
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "original_request": self.original_request,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "platform": self.platform,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "session_id": self.session_id,
            "routing_plan": self.routing_plan.to_dict() if self.routing_plan else None,
            "tasks": [task.to_dict() for task in self.tasks],
            "permissions": [grant.to_dict() for grant in self.permissions],
            "evals": [result.to_dict() for result in self.evals],
            "audit": [event.to_dict() for event in self.audit],
            "metadata": _copy_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmJob":
        job = cls(
            job_id=str(data.get("job_id") or data.get("id") or _stable_job_id(original_request=str(data.get("original_request") or ""))),
            original_request=str(data.get("original_request") or ""),
            status=str(data.get("status") or JobStatus.RECEIVED.value),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or data.get("created_at") or _now()),
            platform=str(data.get("platform") or ""),
            user_id=str(data.get("user_id") or ""),
            chat_id=str(data.get("chat_id") or ""),
            session_id=str(data.get("session_id") or ""),
            routing_plan=RoutingPlan.from_dict(data.get("routing_plan")),
            tasks=[SwarmTask.from_dict(item) for item in data.get("tasks") or [] if isinstance(item, dict)],
            permissions=[PermissionGrant.from_dict(item) for item in data.get("permissions") or [] if isinstance(item, dict)],
            evals=[EvalResult.from_dict(item) for item in data.get("evals") or [] if isinstance(item, dict)],
            audit=[AuditEvent.from_dict(item) for item in data.get("audit") or [] if isinstance(item, dict)],
            metadata=_copy_dict(data.get("metadata")),
        )
        return job


# Convenience aliases requested by the implementation plan.
def new_job(original_request: str, **kwargs: Any) -> SwarmJob:
    return SwarmJob.create(original_request, **kwargs)


def add_task(job: SwarmJob, title: str, description: str = "", **kwargs: Any) -> SwarmTask:
    return job.add_task(title, description, **kwargs)


def transition(job: SwarmJob, status: str, message: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
    job.transition(status, message=message, metadata=metadata)


__all__ = [
    "AuditEvent",
    "EvidenceRequirement",
    "EvalResult",
    "JobStatus",
    "PermissionGrant",
    "RoutingPlan",
    "RoutingTelemetry",
    "SwarmJob",
    "SwarmTask",
    "TaskStatus",
    "add_task",
    "new_job",
    "transition",
]
