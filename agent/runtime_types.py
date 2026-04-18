from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

RunState = Literal[
    "intake",
    "context_recovery",
    "planning",
    "executing",
    "waiting_tool",
    "waiting_human",
    "waiting_external",
    "reviewing",
    "completed",
    "failed",
    "cancelled",
]

NextStep = Literal[
    "run_again",
    "call_tool",
    "delegate",
    "request_approval",
    "request_clarification",
    "pause",
    "resume",
    "finalize",
    "fail",
]

StepType = Literal[
    "context_recovery",
    "model_call",
    "tool_execution",
    "delegation",
    "approval",
    "finalization",
    "interruption",
]

StepStatus = Literal["started", "completed", "failed", "skipped"]
InterruptionReason = Literal[
    "waiting_user",
    "waiting_approval",
    "waiting_process",
    "waiting_external",
    "waiting_cron",
]
InterruptionStatus = Literal["open", "resumed", "expired", "cancelled"]
DelegationVerificationStatus = Literal["pending", "verified", "failed", "skipped"]
DelegationStatus = Literal["started", "completed", "failed"]


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(slots=True)
class RunRecord:
    id: str
    session_id: str | None
    parent_run_id: str | None
    source: str | None
    user_intent: str | None
    state: RunState
    next_step: NextStep | None
    started_at: float
    ended_at: float | None = None
    final_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, **kwargs: Any) -> "RunRecord":
        return cls(
            id=kwargs.pop("id", _new_id()),
            started_at=kwargs.pop("started_at", time.time()),
            **kwargs,
        )

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "parent_run_id": self.parent_run_id,
            "source": self.source,
            "user_intent": self.user_intent,
            "state": self.state,
            "next_step": self.next_step,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "final_status": self.final_status,
            "metadata_json": json.dumps(self.metadata, ensure_ascii=False, sort_keys=True),
        }


@dataclass(slots=True)
class RunStepRecord:
    id: str
    run_id: str
    step_index: int
    step_type: StepType
    status: StepStatus
    started_at: float
    ended_at: float | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    tool_name: str | None = None
    error: str | None = None

    @classmethod
    def create(cls, **kwargs: Any) -> "RunStepRecord":
        return cls(
            id=kwargs.pop("id", _new_id()),
            started_at=kwargs.pop("started_at", time.time()),
            **kwargs,
        )

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "step_type": self.step_type,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "tool_name": self.tool_name,
            "error": self.error,
        }


@dataclass(slots=True)
class RunEventRecord:
    id: str
    run_id: str
    step_id: str | None
    event_type: str
    payload: dict[str, Any]
    timestamp: float

    @classmethod
    def create(cls, **kwargs: Any) -> "RunEventRecord":
        return cls(
            id=kwargs.pop("id", _new_id()),
            timestamp=kwargs.pop("timestamp", time.time()),
            **kwargs,
        )

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "event_type": self.event_type,
            "payload_json": json.dumps(self.payload, ensure_ascii=False, sort_keys=True),
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class InterruptionRecord:
    id: str
    run_id: str
    step_id: str | None
    reason_type: InterruptionReason
    waiting_on: str | None
    snapshot: dict[str, Any]
    resumable: bool
    created_at: float
    resumed_at: float | None = None
    status: InterruptionStatus = "open"

    @classmethod
    def create(cls, **kwargs: Any) -> "InterruptionRecord":
        return cls(
            id=kwargs.pop("id", _new_id()),
            created_at=kwargs.pop("created_at", time.time()),
            **kwargs,
        )

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "reason_type": self.reason_type,
            "waiting_on": self.waiting_on,
            "snapshot_json": json.dumps(self.snapshot, ensure_ascii=False, sort_keys=True),
            "resumable": 1 if self.resumable else 0,
            "created_at": self.created_at,
            "resumed_at": self.resumed_at,
            "status": self.status,
        }


@dataclass(slots=True)
class DelegationRecord:
    id: str
    parent_run_id: str
    child_session_id: str | None
    goal: str
    context_summary: str | None
    allowed_toolsets: list[str]
    side_effect_policy: str | None
    expected_output_type: str | None
    verification_status: DelegationVerificationStatus
    status: DelegationStatus
    created_at: float
    ended_at: float | None = None

    @classmethod
    def create(cls, **kwargs: Any) -> "DelegationRecord":
        return cls(
            id=kwargs.pop("id", _new_id()),
            created_at=kwargs.pop("created_at", time.time()),
            **kwargs,
        )

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_run_id": self.parent_run_id,
            "child_session_id": self.child_session_id,
            "goal": self.goal,
            "context_summary": self.context_summary,
            "allowed_toolsets_json": json.dumps(self.allowed_toolsets, ensure_ascii=False),
            "side_effect_policy": self.side_effect_policy,
            "expected_output_type": self.expected_output_type,
            "verification_status": self.verification_status,
            "status": self.status,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
        }


@dataclass(slots=True)
class ArtifactRecord:
    id: str
    run_id: str
    step_id: str | None
    artifact_type: str
    path_or_ref: str
    produced_by: str | None
    purpose: str | None
    is_final: bool
    delivered: bool
    created_at: float

    @classmethod
    def create(cls, **kwargs: Any) -> "ArtifactRecord":
        return cls(
            id=kwargs.pop("id", _new_id()),
            created_at=kwargs.pop("created_at", time.time()),
            **kwargs,
        )

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "artifact_type": self.artifact_type,
            "path_or_ref": self.path_or_ref,
            "produced_by": self.produced_by,
            "purpose": self.purpose,
            "is_final": 1 if self.is_final else 0,
            "delivered": 1 if self.delivered else 0,
            "created_at": self.created_at,
        }
