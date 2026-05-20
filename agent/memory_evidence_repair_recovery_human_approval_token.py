"""Read-only human approval token draft for memory recovery execution.

The recovery token layer creates a dedicated approval-token draft only after
a recovery execution preview is ready and requires human approval. It never
writes to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from agent.memory_evidence_repair_recovery_execution_preview import (
    RECOVERY_EXECUTION_STATUS_BLOCKED,
    RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED,
    RECOVERY_EXECUTION_STATUS_READY,
)


RECOVERY_APPROVAL_TOKEN_TYPE = "memory_evidence_repair_recovery_human_approval"
RECOVERY_APPROVAL_TOKEN_SCOPE = "manual_memory_evidence_repair_recovery"
RECOVERY_TOKEN_STATUS_DRAFT_READY = "draft_ready_for_recovery_human_approval"
RECOVERY_TOKEN_STATUS_BLOCKED = "blocked"
RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED = "no_action_needed"
DEFAULT_EXPIRES_IN_MINUTES = 15


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryHumanApprovalToken:
    """One read-only recovery human approval token draft."""

    id: str
    token_type: str
    status: str
    scope: str
    approver: str
    approval_reason: str
    execution_preview_status: str
    execution_preview_id: str
    execution_preview_digest: str
    decision_id: str
    route: str
    priority: str
    token_digest: str
    token_preview: str
    expires_in_minutes: int
    operation_ids: tuple[str, ...]
    failed_audit_step_ids: tuple[str, ...]
    step_ids: tuple[str, ...]
    future_mutation_step_ids: tuple[str, ...]
    required_confirmation_text: str
    one_time_constraints: dict[str, Any]
    safety_note: str
    source_execution_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "token_type": self.token_type,
            "status": self.status,
            "scope": self.scope,
            "approver": self.approver,
            "approval_reason": self.approval_reason,
            "execution_preview_status": self.execution_preview_status,
            "execution_preview_id": self.execution_preview_id,
            "execution_preview_digest": self.execution_preview_digest,
            "decision_id": self.decision_id,
            "route": self.route,
            "priority": self.priority,
            "token_digest": self.token_digest,
            "token_preview": self.token_preview,
            "expires_in_minutes": self.expires_in_minutes,
            "operation_ids": list(self.operation_ids),
            "failed_audit_step_ids": list(self.failed_audit_step_ids),
            "step_ids": list(self.step_ids),
            "future_mutation_step_ids": list(self.future_mutation_step_ids),
            "required_confirmation_text": self.required_confirmation_text,
            "one_time_constraints": dict(self.one_time_constraints),
            "safety_note": self.safety_note,
            "source_execution_preview": dict(self.source_execution_preview),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryHumanApprovalTokenReport:
    """Read-only recovery human approval token draft report."""

    generated_at: str
    status: str
    tokens: tuple[MemoryEvidenceRepairRecoveryHumanApprovalToken, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_execution_preview_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "token_count": len(self.tokens),
            "ready_count": 1 if self.status == RECOVERY_TOKEN_STATUS_DRAFT_READY else 0,
            "blocked_count": 1 if self.status == RECOVERY_TOKEN_STATUS_BLOCKED else 0,
            "no_action_count": 1 if self.status == RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED else 0,
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_human_approval_token_available": bool(self.tokens),
            "requires_followup": bool(self.required_actions),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "tokens": [token.to_dict() for token in self.tokens],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_recovery_execution_preview_report": dict(
                self.source_recovery_execution_preview_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_human_approval_token(
    *,
    recovery_execution: Mapping[str, Any] | None = None,
    approver: str = "human",
    approval_reason: str = "",
    expires_in_minutes: int = DEFAULT_EXPIRES_IN_MINUTES,
) -> MemoryEvidenceRepairRecoveryHumanApprovalTokenReport:
    """Build a read-only recovery approval token draft from an execution preview."""

    recovery_execution = (
        recovery_execution if isinstance(recovery_execution, Mapping) else {}
    )
    execution_status = _str(recovery_execution.get("status"))
    preview = _extract_execution_preview(recovery_execution)

    if not recovery_execution or execution_status == RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED:
        return _empty_report(recovery_execution=recovery_execution)

    if execution_status == RECOVERY_EXECUTION_STATUS_BLOCKED and not preview:
        blocking_reasons = tuple(_list_of_str(recovery_execution.get("blocking_reasons")))
        required_actions = tuple(_list_of_str(recovery_execution.get("required_actions")))
        return MemoryEvidenceRepairRecoveryHumanApprovalTokenReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_TOKEN_STATUS_BLOCKED,
            tokens=(),
            blocking_reasons=blocking_reasons
            or ("Recovery execution preview is blocked.",),
            required_actions=required_actions
            or ("produce_ready_recovery_execution_preview",),
            source_recovery_execution_preview_report=dict(recovery_execution),
        )

    if execution_status != RECOVERY_EXECUTION_STATUS_READY or not preview:
        return MemoryEvidenceRepairRecoveryHumanApprovalTokenReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_TOKEN_STATUS_BLOCKED,
            tokens=(),
            blocking_reasons=("Recovery execution preview is not ready.",),
            required_actions=("produce_ready_recovery_execution_preview",),
            source_recovery_execution_preview_report=dict(recovery_execution),
        )

    if not _requires_recovery_approval(preview):
        return MemoryEvidenceRepairRecoveryHumanApprovalTokenReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED,
            tokens=(),
            blocking_reasons=(),
            required_actions=(),
            source_recovery_execution_preview_report=dict(recovery_execution),
        )

    token = _token_from_execution_preview(
        preview=preview,
        approver=approver,
        approval_reason=approval_reason,
        expires_in_minutes=expires_in_minutes,
    )
    return MemoryEvidenceRepairRecoveryHumanApprovalTokenReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_TOKEN_STATUS_DRAFT_READY,
        tokens=(token,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_execution_preview_report=dict(recovery_execution),
    )


def empty_evidence_repair_recovery_human_approval_token() -> MemoryEvidenceRepairRecoveryHumanApprovalTokenReport:
    """Return an empty read-only recovery human approval token report."""

    return _empty_report(recovery_execution={})


def _token_from_execution_preview(
    *,
    preview: Mapping[str, Any],
    approver: str,
    approval_reason: str,
    expires_in_minutes: int,
) -> MemoryEvidenceRepairRecoveryHumanApprovalToken:
    execution_preview_digest = _execution_preview_digest(preview)
    step_ids = tuple(_list_of_str(preview.get("step_ids")))
    future_mutation_step_ids = tuple(
        _str(step.get("id"))
        for step in _list(preview.get("steps"))
        if isinstance(step, Mapping) and step.get("future_would_mutate_memory") is True
    )
    operation_ids = tuple(_list_of_str(preview.get("operation_ids")))
    failed_step_ids = tuple(_list_of_str(preview.get("failed_audit_step_ids")))
    token_seed = {
        "token_type": RECOVERY_APPROVAL_TOKEN_TYPE,
        "scope": RECOVERY_APPROVAL_TOKEN_SCOPE,
        "approver": _str(approver) or "human",
        "approval_reason": _str(approval_reason),
        "execution_preview_digest": execution_preview_digest,
        "execution_preview_id": _str(preview.get("id")),
        "decision_id": _str(preview.get("decision_id")),
        "route": _str(preview.get("route")),
        "operation_ids": operation_ids,
        "failed_audit_step_ids": failed_step_ids,
        "step_ids": step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "expires_in_minutes": _expires_in_minutes(expires_in_minutes),
    }
    token_digest = _digest(token_seed)
    token_id = f"recovery-approval-token-{token_digest[:16]}"
    required_confirmation_text = (
        f"APPROVE {RECOVERY_APPROVAL_TOKEN_SCOPE} {token_id}"
    )
    return MemoryEvidenceRepairRecoveryHumanApprovalToken(
        id=token_id,
        token_type=RECOVERY_APPROVAL_TOKEN_TYPE,
        status=RECOVERY_TOKEN_STATUS_DRAFT_READY,
        scope=RECOVERY_APPROVAL_TOKEN_SCOPE,
        approver=_str(approver) or "human",
        approval_reason=_str(approval_reason),
        execution_preview_status=_str(preview.get("status")),
        execution_preview_id=_str(preview.get("id")),
        execution_preview_digest=execution_preview_digest,
        decision_id=_str(preview.get("decision_id")),
        route=_str(preview.get("route")),
        priority=_str(preview.get("priority")),
        token_digest=token_digest,
        token_preview=f"{token_id}:{token_digest[:12]}",
        expires_in_minutes=_expires_in_minutes(expires_in_minutes),
        operation_ids=operation_ids,
        failed_audit_step_ids=failed_step_ids,
        step_ids=step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        required_confirmation_text=required_confirmation_text,
        one_time_constraints={
            "one_time_use": True,
            "requires_exact_execution_preview_digest": execution_preview_digest,
            "requires_exact_execution_preview_id": _str(preview.get("id")),
            "requires_exact_decision_id": _str(preview.get("decision_id")),
            "requires_exact_route": _str(preview.get("route")),
            "requires_exact_confirmation_text": required_confirmation_text,
            "requires_same_future_mutation_step_ids": list(future_mutation_step_ids),
            "expires_in_minutes": _expires_in_minutes(expires_in_minutes),
        },
        safety_note=(
            "Read-only recovery approval token draft. This is not a persisted "
            "credential and does not execute recovery or mutate durable memory by itself."
        ),
        source_execution_preview=dict(preview),
    )


def _extract_execution_preview(recovery_execution: Mapping[str, Any]) -> dict[str, Any]:
    previews = recovery_execution.get("previews")
    if isinstance(previews, list):
        for preview in previews:
            if isinstance(preview, Mapping):
                return dict(preview)
    if _str(recovery_execution.get("id")).startswith("recovery-execution-preview-"):
        return dict(recovery_execution)
    return {}


def _requires_recovery_approval(preview: Mapping[str, Any]) -> bool:
    if preview.get("future_would_mutate_memory") is True:
        return True
    return preview.get("human_approval_required") is True


def _execution_preview_digest(preview: Mapping[str, Any]) -> str:
    existing = _str(preview.get("preview_digest"))
    if existing:
        return existing
    return _digest(
        {
            "id": _str(preview.get("id")),
            "route": _str(preview.get("route")),
            "decision_id": _str(preview.get("decision_id")),
            "operation_ids": tuple(_list_of_str(preview.get("operation_ids"))),
            "failed_audit_step_ids": tuple(_list_of_str(preview.get("failed_audit_step_ids"))),
            "step_ids": tuple(_list_of_str(preview.get("step_ids"))),
        }
    )


def _empty_report(
    *,
    recovery_execution: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryHumanApprovalTokenReport:
    return MemoryEvidenceRepairRecoveryHumanApprovalTokenReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED,
        tokens=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_execution_preview_report=dict(recovery_execution),
    )


def _expires_in_minutes(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_EXPIRES_IN_MINUTES
    return parsed if parsed > 0 else DEFAULT_EXPIRES_IN_MINUTES


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_str(value: Any) -> list[str]:
    return [_str(item) for item in value] if isinstance(value, list) else []


def _str(value: Any) -> str:
    return str(value or "").strip()
