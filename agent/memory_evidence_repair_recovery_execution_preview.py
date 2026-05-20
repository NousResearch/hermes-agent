"""Read-only recovery execution preview for memory evidence repair.

The recovery execution preview expands a recovery decision into an ordered
manual execution plan. It never executes rollback, never mutates durable
memory, and never changes receipt, token, or lock state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
)
from agent.memory_evidence_repair_recovery_decision_gate import (
    DECISION_STATUS_PLANNED,
    RECOVERY_DECISION_STATUS_BLOCKED,
    RECOVERY_DECISION_STATUS_NO_ACTION_NEEDED,
    RECOVERY_DECISION_STATUS_READY,
    RECOVERY_ROUTE_ISOLATE_AND_REVIEW,
    RECOVERY_ROUTE_MANUAL_ROLLBACK,
    RECOVERY_ROUTE_PREPAREDNESS_ONLY,
    RECOVERY_ROUTE_RERUN_AUDIT,
)


RECOVERY_EXECUTION_STATUS_READY = "recovery_execution_preview_ready"
RECOVERY_EXECUTION_STATUS_BLOCKED = "blocked"
RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED = "no_action_needed"

CHECK_RECOVERY_DECISION_READY = "recovery_decision_ready"
CHECK_RECOVERY_DECISION_INTEGRITY = "recovery_decision_integrity"
CHECK_EXECUTION_CONTROLS = "execution_controls"

EXECUTION_STEP_STATUS_PLANNED = "planned"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryExecutionCheck:
    """One read-only recovery execution preview check."""

    id: str
    status: str
    reason: str
    required_actions: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "reason": self.reason,
            "required_actions": list(self.required_actions),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryExecutionStep:
    """One future manual recovery execution step preview."""

    id: str
    sequence: int
    phase: str
    action: str
    description: str
    status: str
    route: str
    operation_id: str
    receipt_id: str
    token_id: str
    lock_id: str
    required_precondition: str
    expected_result: str
    source_decision_id: str
    human_approval_required: bool
    future_would_mutate_memory: bool
    rollback_safe_point: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "phase": self.phase,
            "action": self.action,
            "description": self.description,
            "status": self.status,
            "route": self.route,
            "operation_id": self.operation_id,
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "required_precondition": self.required_precondition,
            "expected_result": self.expected_result,
            "source_decision_id": self.source_decision_id,
            "human_approval_required": self.human_approval_required,
            "future_would_mutate_memory": self.future_would_mutate_memory,
            "rollback_safe_point": self.rollback_safe_point,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryExecutionPreview:
    """One read-only recovery execution preview."""

    id: str
    status: str
    route: str
    priority: str
    decision_id: str
    operation_ids: tuple[str, ...]
    failed_audit_step_ids: tuple[str, ...]
    receipt_id: str
    token_id: str
    lock_id: str
    step_ids: tuple[str, ...]
    preview_digest: str
    execution_preview: str
    steps: tuple[MemoryEvidenceRepairRecoveryExecutionStep, ...]
    future_would_mutate_memory: bool
    human_approval_required: bool
    blocked_actions: tuple[str, ...]
    safety_note: str
    source_recovery_decision: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "route": self.route,
            "priority": self.priority,
            "decision_id": self.decision_id,
            "operation_ids": list(self.operation_ids),
            "failed_audit_step_ids": list(self.failed_audit_step_ids),
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "step_ids": list(self.step_ids),
            "preview_digest": self.preview_digest,
            "execution_preview": self.execution_preview,
            "steps": [step.to_dict() for step in self.steps],
            "future_would_mutate_memory": self.future_would_mutate_memory,
            "human_approval_required": self.human_approval_required,
            "blocked_actions": list(self.blocked_actions),
            "safety_note": self.safety_note,
            "source_recovery_decision": dict(self.source_recovery_decision),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryExecutionPreviewReport:
    """Read-only recovery execution preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryExecutionCheck, ...]
    previews: tuple[MemoryEvidenceRepairRecoveryExecutionPreview, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_decision_gate_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        by_route: dict[str, int] = {}
        future_mutation_count = 0
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        for preview in self.previews:
            by_route[preview.route] = by_route.get(preview.route, 0) + 1
            future_mutation_count += sum(
                1 for step in preview.steps if step.future_would_mutate_memory
            )
        step_count = sum(len(preview.steps) for preview in self.previews)
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "preview_count": len(self.previews),
            "execution_step_count": step_count,
            "future_mutation_step_count": future_mutation_count,
            "manual_rollback_preview_count": by_route.get(
                RECOVERY_ROUTE_MANUAL_ROLLBACK,
                0,
            ),
            "preparedness_preview_count": by_route.get(
                RECOVERY_ROUTE_PREPAREDNESS_ONLY,
                0,
            ),
            "isolation_review_preview_count": by_route.get(
                RECOVERY_ROUTE_ISOLATE_AND_REVIEW,
                0,
            ),
            "rerun_audit_preview_count": by_route.get(RECOVERY_ROUTE_RERUN_AUDIT, 0),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_execution_preview_available": bool(self.previews),
            "recovery_execution_ready": self.status == RECOVERY_EXECUTION_STATUS_READY,
            "has_blocks": self.status == RECOVERY_EXECUTION_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
            "by_route": by_route,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "previews": [preview.to_dict() for preview in self.previews],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_recovery_decision_gate_report": dict(
                self.source_recovery_decision_gate_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_execution_preview(
    *,
    recovery_decision: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryExecutionPreviewReport:
    """Build a read-only recovery execution preview from a decision gate."""

    recovery_decision = recovery_decision if isinstance(recovery_decision, Mapping) else {}
    decision_status = _str(recovery_decision.get("status"))
    decision = _extract_decision(recovery_decision)

    if not recovery_decision or decision_status == RECOVERY_DECISION_STATUS_NO_ACTION_NEEDED:
        return _empty_report(recovery_decision=recovery_decision)

    if decision_status == RECOVERY_DECISION_STATUS_BLOCKED and not decision:
        return _blocked_report(
            recovery_decision=recovery_decision,
            checks=(
                _fail(
                    CHECK_RECOVERY_DECISION_READY,
                    "Recovery decision gate is blocked and cannot drive execution preview.",
                    tuple(_list_of_str(recovery_decision.get("required_actions")))
                    or ("produce_ready_recovery_decision_gate",),
                    {"recovery_decision_status": decision_status},
                ),
            ),
        )

    if not decision:
        return _blocked_report(
            recovery_decision=recovery_decision,
            checks=(
                _fail(
                    CHECK_RECOVERY_DECISION_READY,
                    "No recovery decision was supplied.",
                    ("produce_recovery_decision_gate",),
                    {"recovery_decision_status": decision_status},
                ),
            ),
        )

    checks = (
        _recovery_decision_ready_check(
            recovery_decision=recovery_decision,
            decision=decision,
        ),
        _recovery_decision_integrity_check(decision),
        _execution_controls_check(decision),
    )
    blocking_reasons = tuple(
        _dedupe_strings(check.reason for check in checks if check.status == CHECK_STATUS_FAIL)
    )
    required_actions = tuple(
        _dedupe_strings(
            action
            for check in checks
            if check.status == CHECK_STATUS_FAIL
            for action in check.required_actions
        )
    )
    if blocking_reasons:
        return MemoryEvidenceRepairRecoveryExecutionPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_EXECUTION_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_decision_gate_report=dict(recovery_decision),
        )

    preview = _preview_from_decision(decision)
    return MemoryEvidenceRepairRecoveryExecutionPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_EXECUTION_STATUS_READY,
        checks=checks,
        previews=(preview,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_decision_gate_report=dict(recovery_decision),
    )


def empty_evidence_repair_recovery_execution_preview() -> MemoryEvidenceRepairRecoveryExecutionPreviewReport:
    """Return an empty read-only recovery execution preview."""

    return _empty_report(recovery_decision={})


def _recovery_decision_ready_check(
    *,
    recovery_decision: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutionCheck:
    report_status = _str(recovery_decision.get("status"))
    decision_status = _str(decision.get("status"))
    if report_status and report_status != RECOVERY_DECISION_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_DECISION_READY,
            "Recovery decision gate report is not ready for execution preview.",
            tuple(_list_of_str(recovery_decision.get("required_actions")))
            or ("produce_ready_recovery_decision_gate",),
            {"report_status": report_status, "decision_status": decision_status},
        )
    if decision_status != DECISION_STATUS_PLANNED:
        return _fail(
            CHECK_RECOVERY_DECISION_READY,
            "Recovery decision entry is not planned.",
            ("regenerate_recovery_decision_gate",),
            {"report_status": report_status, "decision_status": decision_status},
        )
    return _pass(
        CHECK_RECOVERY_DECISION_READY,
        "Recovery decision is ready for execution preview planning.",
        {"decision_id": _str(decision.get("id"))},
    )


def _recovery_decision_integrity_check(
    decision: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutionCheck:
    route = _str(decision.get("route"))
    missing_fields: list[str] = []
    for field_name in ("id", "route", "priority", "recommended_actions", "blocked_actions"):
        value = decision.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    if route not in _known_routes() or missing_fields:
        return _fail(
            CHECK_RECOVERY_DECISION_INTEGRITY,
            "Recovery decision route or required fields are invalid.",
            ("regenerate_recovery_decision_gate",),
            {"route": route, "missing_fields": missing_fields},
        )
    return _pass(
        CHECK_RECOVERY_DECISION_INTEGRITY,
        "Recovery decision structure is usable for execution preview planning.",
        {"decision_id": _str(decision.get("id")), "route": route},
    )


def _execution_controls_check(
    decision: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutionCheck:
    preconditions = set(_list_of_str(decision.get("required_preconditions")))
    blocked = set(_list_of_str(decision.get("blocked_actions")))
    missing: list[str] = []
    if "no_automatic_memory_mutation" not in preconditions:
        missing.append("no_automatic_memory_mutation")
    if "automatic_memory_rollback" not in blocked:
        missing.append("automatic_memory_rollback_block")
    if missing:
        return _fail(
            CHECK_EXECUTION_CONTROLS,
            "Recovery decision does not preserve read-only execution controls.",
            ("regenerate_recovery_decision_gate_with_controls",),
            {"missing_controls": missing},
        )
    return _pass(
        CHECK_EXECUTION_CONTROLS,
        "Execution preview remains read-only and automatic recovery is blocked.",
        {
            "decision_id": _str(decision.get("id")),
            "automatic_memory_rollback_blocked": True,
        },
    )


def _preview_from_decision(
    decision: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutionPreview:
    route = _str(decision.get("route"))
    steps = tuple(_steps_for_decision(decision))
    step_ids = tuple(step.id for step in steps)
    operation_ids = tuple(_list_of_str(decision.get("operation_ids")))
    failed_step_ids = tuple(_list_of_str(decision.get("failed_audit_step_ids")))
    seed = {
        "decision_id": _str(decision.get("id")),
        "route": route,
        "priority": _str(decision.get("priority")),
        "operation_ids": operation_ids,
        "failed_audit_step_ids": failed_step_ids,
        "step_ids": step_ids,
    }
    preview_digest = _digest(seed)
    preview_id = f"recovery-execution-preview-{preview_digest[:16]}"
    return MemoryEvidenceRepairRecoveryExecutionPreview(
        id=preview_id,
        status=RECOVERY_EXECUTION_STATUS_READY,
        route=route,
        priority=_str(decision.get("priority")),
        decision_id=_str(decision.get("id")),
        operation_ids=operation_ids,
        failed_audit_step_ids=failed_step_ids,
        receipt_id=_str(decision.get("receipt_id")),
        token_id=_str(decision.get("token_id")),
        lock_id=_str(decision.get("lock_id")),
        step_ids=step_ids,
        preview_digest=preview_digest,
        execution_preview=f"{preview_id}:{preview_digest[:12]}",
        steps=steps,
        future_would_mutate_memory=any(step.future_would_mutate_memory for step in steps),
        human_approval_required=any(step.human_approval_required for step in steps),
        blocked_actions=tuple(_list_of_str(decision.get("blocked_actions"))),
        safety_note=(
            "Read-only recovery execution preview. It orders future manual "
            "recovery actions but does not execute rollback or mutate durable memory."
        ),
        source_recovery_decision=dict(decision),
    )


def _steps_for_decision(
    decision: Mapping[str, Any],
) -> list[MemoryEvidenceRepairRecoveryExecutionStep]:
    route = _str(decision.get("route"))
    if route == RECOVERY_ROUTE_MANUAL_ROLLBACK:
        actions = (
            ("preflight", "verify_recovery_decision_gate", "decision_gate_ready"),
            ("preserve", "preserve_failed_audit_context", "failed_audit_context_preserved"),
            ("isolate", "isolate_impacted_memory_records", "impacted_records_isolated"),
            ("verify", "verify_snapshot_or_rollback_plan", "snapshot_or_rollback_plan_verified"),
            ("restore", "apply_manual_rollback_restore", "pre_commit_state_restored"),
            ("reconcile", "reconcile_receipt_token_lock_state", "receipt_token_lock_state_consistent"),
            ("audit", "rerun_post_commit_audit", "post_commit_audit_passes_after_recovery"),
        )
    elif route == RECOVERY_ROUTE_ISOLATE_AND_REVIEW:
        actions = (
            ("preflight", "verify_recovery_decision_gate", "decision_gate_ready"),
            ("preserve", "preserve_failed_audit_context", "failed_audit_context_preserved"),
            ("isolate", "isolate_impacted_memory_records", "impacted_records_isolated"),
            ("review", "inspect_failed_audit_signals", "human_review_route_chosen"),
            ("audit", "rerun_post_commit_audit_if_needed", "post_commit_audit_rechecked"),
        )
    elif route == RECOVERY_ROUTE_PREPAREDNESS_ONLY:
        actions = (
            ("preflight", "verify_recovery_decision_gate", "decision_gate_ready"),
            ("prepare", "keep_rollback_drill_available", "rollback_drill_available"),
            ("verify", "verify_snapshot_or_rollback_plan_before_future_commit", "future_rollback_source_verified"),
            ("audit", "rerun_post_commit_audit_after_future_commit", "future_audit_plan_ready"),
        )
    else:
        actions = (
            ("preflight", "verify_recovery_decision_gate", "decision_gate_ready"),
            ("audit", "rerun_post_commit_audit", "post_commit_audit_rechecked"),
        )

    operation_ids = _list_of_str(decision.get("operation_ids"))
    operation_text = ",".join(operation_ids)
    return [
        _step(
            sequence=index,
            phase=phase,
            action=action,
            expected_result=expected,
            decision=decision,
            operation_id=operation_text if action in _operation_scoped_actions() else "",
            future_would_mutate_memory=action == "apply_manual_rollback_restore",
            rollback_safe_point=phase in {"preflight", "preserve", "verify", "audit"},
        )
        for index, (phase, action, expected) in enumerate(actions, start=1)
    ]


def _step(
    *,
    sequence: int,
    phase: str,
    action: str,
    expected_result: str,
    decision: Mapping[str, Any],
    operation_id: str,
    future_would_mutate_memory: bool,
    rollback_safe_point: bool,
) -> MemoryEvidenceRepairRecoveryExecutionStep:
    route = _str(decision.get("route"))
    human_approval_required = future_would_mutate_memory or route in {
        RECOVERY_ROUTE_MANUAL_ROLLBACK,
        RECOVERY_ROUTE_ISOLATE_AND_REVIEW,
    }
    return MemoryEvidenceRepairRecoveryExecutionStep(
        id=f"recovery-execution-step-{sequence}-{action}",
        sequence=sequence,
        phase=phase,
        action=action,
        description=_description(action),
        status=EXECUTION_STEP_STATUS_PLANNED,
        route=route,
        operation_id=operation_id,
        receipt_id=_str(decision.get("receipt_id")),
        token_id=_str(decision.get("token_id")),
        lock_id=_str(decision.get("lock_id")),
        required_precondition=_required_precondition(action, decision),
        expected_result=expected_result,
        source_decision_id=_str(decision.get("id")),
        human_approval_required=human_approval_required,
        future_would_mutate_memory=future_would_mutate_memory,
        rollback_safe_point=rollback_safe_point,
    )


def _description(action: str) -> str:
    descriptions = {
        "verify_recovery_decision_gate": "Verify the recovery decision gate, route, priority, and blocked actions.",
        "preserve_failed_audit_context": "Preserve failed audit evidence before any future manual recovery work.",
        "isolate_impacted_memory_records": "Mark impacted records for human review before further writes.",
        "verify_snapshot_or_rollback_plan": "Verify a pre-commit snapshot or rollback plan before restore.",
        "apply_manual_rollback_restore": "Manually restore impacted evidence from the verified rollback source.",
        "reconcile_receipt_token_lock_state": "Reconcile receipt, approval-token, and write-lock state after recovery.",
        "rerun_post_commit_audit": "Rerun post-commit audit after recovery handling.",
        "inspect_failed_audit_signals": "Inspect failed audit signals before choosing the final recovery route.",
        "rerun_post_commit_audit_if_needed": "Rerun post-commit audit if human review finds recoverable state.",
        "keep_rollback_drill_available": "Keep the rollback drill available as preparedness coverage.",
        "verify_snapshot_or_rollback_plan_before_future_commit": "Verify rollback sources before a future commit.",
        "rerun_post_commit_audit_after_future_commit": "Run post-commit audit after the future commit occurs.",
    }
    return descriptions.get(action, "Preview the future manual recovery action.")


def _required_precondition(action: str, decision: Mapping[str, Any]) -> str:
    if action == "apply_manual_rollback_restore":
        return "explicit_human_approval_before_recovery_execution"
    if action == "verify_snapshot_or_rollback_plan":
        return "snapshot_or_rollback_plan_verified"
    if action == "isolate_impacted_memory_records":
        return "failed_audit_context_preserved"
    preconditions = _list_of_str(decision.get("required_preconditions"))
    return preconditions[0] if preconditions else "no_automatic_memory_mutation"


def _operation_scoped_actions() -> set[str]:
    return {
        "isolate_impacted_memory_records",
        "verify_snapshot_or_rollback_plan",
        "apply_manual_rollback_restore",
        "verify_snapshot_or_rollback_plan_before_future_commit",
    }


def _extract_decision(recovery_decision: Mapping[str, Any]) -> dict[str, Any]:
    decisions = recovery_decision.get("decisions")
    if isinstance(decisions, list):
        for decision in decisions:
            if isinstance(decision, Mapping):
                return dict(decision)
    if _str(recovery_decision.get("id")).startswith("recovery-decision-"):
        return dict(recovery_decision)
    return {}


def _known_routes() -> set[str]:
    return {
        RECOVERY_ROUTE_MANUAL_ROLLBACK,
        RECOVERY_ROUTE_ISOLATE_AND_REVIEW,
        RECOVERY_ROUTE_PREPAREDNESS_ONLY,
        RECOVERY_ROUTE_RERUN_AUDIT,
    }


def _empty_report(
    *,
    recovery_decision: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutionPreviewReport:
    return MemoryEvidenceRepairRecoveryExecutionPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_decision_gate_report=dict(recovery_decision),
    )


def _blocked_report(
    *,
    recovery_decision: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryExecutionCheck, ...],
) -> MemoryEvidenceRepairRecoveryExecutionPreviewReport:
    return MemoryEvidenceRepairRecoveryExecutionPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_EXECUTION_STATUS_BLOCKED,
        checks=checks,
        previews=(),
        blocking_reasons=tuple(
            _dedupe_strings(check.reason for check in checks if check.status == CHECK_STATUS_FAIL)
        ),
        required_actions=tuple(
            _dedupe_strings(
                action
                for check in checks
                if check.status == CHECK_STATUS_FAIL
                for action in check.required_actions
            )
        ),
        source_recovery_decision_gate_report=dict(recovery_decision),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutionCheck:
    return MemoryEvidenceRepairRecoveryExecutionCheck(
        id=check_id,
        status=CHECK_STATUS_PASS,
        reason=reason,
        required_actions=(),
        details=dict(details),
    )


def _fail(
    check_id: str,
    reason: str,
    required_actions: Sequence[str],
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutionCheck:
    return MemoryEvidenceRepairRecoveryExecutionCheck(
        id=check_id,
        status=CHECK_STATUS_FAIL,
        reason=reason,
        required_actions=tuple(required_actions),
        details=dict(details),
    )


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _list_of_str(value: Any) -> list[str]:
    return [_str(item) for item in value] if isinstance(value, list) else []


def _dedupe_strings(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _str(value: Any) -> str:
    return str(value or "").strip()
