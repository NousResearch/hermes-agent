"""Read-only recovery decision gate for memory evidence repair.

The recovery decision gate turns a rollback drill preview into a controlled
decision about the next recovery route. It never performs rollback, never
mutates durable memory, and never changes receipt, token, or lock state.
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
from agent.memory_evidence_repair_rollback_drill_preview import (
    DRILL_STEP_STATUS_PLANNED,
    ROLLBACK_DRILL_STATUS_BLOCKED,
    ROLLBACK_DRILL_STATUS_NO_ACTION_NEEDED,
    ROLLBACK_DRILL_STATUS_READY,
    ROLLBACK_DRILL_TRIGGER_AUDIT_FAILURE,
    ROLLBACK_DRILL_TRIGGER_PREPAREDNESS,
)


RECOVERY_DECISION_STATUS_READY = "recovery_decision_gate_ready"
RECOVERY_DECISION_STATUS_BLOCKED = "blocked"
RECOVERY_DECISION_STATUS_NO_ACTION_NEEDED = "no_action_needed"

RECOVERY_ROUTE_PREPAREDNESS_ONLY = "preparedness_review_only"
RECOVERY_ROUTE_ISOLATE_AND_REVIEW = "isolate_and_review"
RECOVERY_ROUTE_MANUAL_ROLLBACK = "manual_rollback_required"
RECOVERY_ROUTE_RERUN_AUDIT = "rerun_post_commit_audit"

CHECK_ROLLBACK_DRILL_READY = "rollback_drill_ready"
CHECK_ROLLBACK_DRILL_INTEGRITY = "rollback_drill_integrity"
CHECK_OPERATOR_CONTROLS = "operator_controls"

DECISION_STATUS_PLANNED = "planned"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryDecisionCheck:
    """One read-only recovery decision readiness check."""

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
class MemoryEvidenceRepairRecoveryDecision:
    """One read-only recovery route decision."""

    id: str
    status: str
    route: str
    priority: str
    reason: str
    rollback_drill_id: str
    trigger: str
    operation_ids: tuple[str, ...]
    failed_audit_step_ids: tuple[str, ...]
    receipt_id: str
    token_id: str
    lock_id: str
    recommended_actions: tuple[str, ...]
    required_preconditions: tuple[str, ...]
    blocked_actions: tuple[str, ...]
    next_tool_hint: str
    decision_digest: str
    source_rollback_drill: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "route": self.route,
            "priority": self.priority,
            "reason": self.reason,
            "rollback_drill_id": self.rollback_drill_id,
            "trigger": self.trigger,
            "operation_ids": list(self.operation_ids),
            "failed_audit_step_ids": list(self.failed_audit_step_ids),
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "recommended_actions": list(self.recommended_actions),
            "required_preconditions": list(self.required_preconditions),
            "blocked_actions": list(self.blocked_actions),
            "next_tool_hint": self.next_tool_hint,
            "decision_digest": self.decision_digest,
            "source_rollback_drill": dict(self.source_rollback_drill),
            "future_would_mutate_memory": False,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryDecisionGateReport:
    """Read-only recovery decision gate report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryDecisionCheck, ...]
    decisions: tuple[MemoryEvidenceRepairRecoveryDecision, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_rollback_drill_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        by_route: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        for decision in self.decisions:
            by_route[decision.route] = by_route.get(decision.route, 0) + 1
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "decision_count": len(self.decisions),
            "manual_rollback_required_count": by_route.get(
                RECOVERY_ROUTE_MANUAL_ROLLBACK,
                0,
            ),
            "isolate_and_review_count": by_route.get(
                RECOVERY_ROUTE_ISOLATE_AND_REVIEW,
                0,
            ),
            "preparedness_review_count": by_route.get(
                RECOVERY_ROUTE_PREPAREDNESS_ONLY,
                0,
            ),
            "rerun_audit_count": by_route.get(RECOVERY_ROUTE_RERUN_AUDIT, 0),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_decision_available": bool(self.decisions),
            "recovery_decision_ready": self.status == RECOVERY_DECISION_STATUS_READY,
            "has_blocks": self.status == RECOVERY_DECISION_STATUS_BLOCKED,
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
            "decisions": [decision.to_dict() for decision in self.decisions],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_rollback_drill_report": dict(self.source_rollback_drill_report),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_decision_gate(
    *,
    rollback_drill: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryDecisionGateReport:
    """Build a read-only recovery decision gate from a rollback drill preview."""

    rollback_drill = rollback_drill if isinstance(rollback_drill, Mapping) else {}
    drill_status = _str(rollback_drill.get("status"))
    drill_preview = _extract_drill_preview(rollback_drill)

    if not rollback_drill or drill_status == ROLLBACK_DRILL_STATUS_NO_ACTION_NEEDED:
        return _empty_report(rollback_drill=rollback_drill)

    if drill_status == ROLLBACK_DRILL_STATUS_BLOCKED and not drill_preview:
        return _blocked_report(
            rollback_drill=rollback_drill,
            checks=(
                _fail(
                    CHECK_ROLLBACK_DRILL_READY,
                    "Rollback drill preview is blocked and cannot drive a recovery decision.",
                    tuple(_list_of_str(rollback_drill.get("required_actions")))
                    or ("produce_ready_rollback_drill_preview",),
                    {"rollback_drill_status": drill_status},
                ),
            ),
        )

    if not drill_preview:
        return _blocked_report(
            rollback_drill=rollback_drill,
            checks=(
                _fail(
                    CHECK_ROLLBACK_DRILL_READY,
                    "No rollback drill preview was supplied.",
                    ("produce_rollback_drill_preview",),
                    {"rollback_drill_status": drill_status},
                ),
            ),
        )

    checks = (
        _rollback_drill_ready_check(rollback_drill=rollback_drill, drill_preview=drill_preview),
        _rollback_drill_integrity_check(drill_preview),
        _operator_controls_check(drill_preview),
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
        return MemoryEvidenceRepairRecoveryDecisionGateReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_DECISION_STATUS_BLOCKED,
            checks=checks,
            decisions=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_rollback_drill_report=dict(rollback_drill),
        )

    decision = _decision_from_drill(drill_preview)
    return MemoryEvidenceRepairRecoveryDecisionGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_DECISION_STATUS_READY,
        checks=checks,
        decisions=(decision,),
        blocking_reasons=(),
        required_actions=(),
        source_rollback_drill_report=dict(rollback_drill),
    )


def empty_evidence_repair_recovery_decision_gate() -> MemoryEvidenceRepairRecoveryDecisionGateReport:
    """Return an empty read-only recovery decision gate."""

    return _empty_report(rollback_drill={})


def _rollback_drill_ready_check(
    *,
    rollback_drill: Mapping[str, Any],
    drill_preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryDecisionCheck:
    report_status = _str(rollback_drill.get("status"))
    preview_status = _str(drill_preview.get("status"))
    if report_status and report_status != ROLLBACK_DRILL_STATUS_READY:
        return _fail(
            CHECK_ROLLBACK_DRILL_READY,
            "Rollback drill report is not ready for recovery decision planning.",
            tuple(_list_of_str(rollback_drill.get("required_actions")))
            or ("produce_ready_rollback_drill_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    if preview_status != ROLLBACK_DRILL_STATUS_READY:
        return _fail(
            CHECK_ROLLBACK_DRILL_READY,
            "Rollback drill preview entry is not ready.",
            ("produce_ready_rollback_drill_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    return _pass(
        CHECK_ROLLBACK_DRILL_READY,
        "Rollback drill preview is ready for recovery decision planning.",
        {"rollback_drill_id": _str(drill_preview.get("id"))},
    )


def _rollback_drill_integrity_check(
    drill_preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryDecisionCheck:
    missing_fields: list[str] = []
    for field_name in ("id", "trigger", "rollback_mode", "steps"):
        value = drill_preview.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    step_statuses = [
        _str(step.get("status"))
        for step in _list(drill_preview.get("steps"))
        if isinstance(step, Mapping)
    ]
    non_planned = [status for status in step_statuses if status != DRILL_STEP_STATUS_PLANNED]
    if missing_fields or non_planned:
        return _fail(
            CHECK_ROLLBACK_DRILL_INTEGRITY,
            "Rollback drill preview has missing fields or non-planned steps.",
            ("regenerate_rollback_drill_preview",),
            {
                "missing_fields": missing_fields,
                "non_planned_step_statuses": non_planned,
            },
        )
    return _pass(
        CHECK_ROLLBACK_DRILL_INTEGRITY,
        "Rollback drill preview structure is usable for decision planning.",
        {"rollback_drill_id": _str(drill_preview.get("id")), "step_count": len(step_statuses)},
    )


def _operator_controls_check(
    drill_preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryDecisionCheck:
    return _pass(
        CHECK_OPERATOR_CONTROLS,
        "Future recovery execution remains gated by explicit human approval.",
        {
            "rollback_drill_id": _str(drill_preview.get("id")),
            "requires_human_approval_before_execution": True,
            "read_only_gate": True,
        },
    )


def _decision_from_drill(
    drill_preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryDecision:
    route = _route_for_drill(drill_preview)
    recommended_actions = tuple(_recommended_actions(route))
    required_preconditions = tuple(_required_preconditions(route))
    blocked_actions = tuple(_blocked_actions(route))
    operation_ids = tuple(_list_of_str(drill_preview.get("operation_ids")))
    failed_step_ids = tuple(_list_of_str(drill_preview.get("failed_audit_step_ids")))
    trigger = _str(drill_preview.get("trigger"))
    seed = {
        "route": route,
        "rollback_drill_id": _str(drill_preview.get("id")),
        "trigger": trigger,
        "operation_ids": operation_ids,
        "failed_audit_step_ids": failed_step_ids,
        "recommended_actions": recommended_actions,
        "required_preconditions": required_preconditions,
    }
    decision_digest = _digest(seed)
    return MemoryEvidenceRepairRecoveryDecision(
        id=f"recovery-decision-{decision_digest[:16]}",
        status=DECISION_STATUS_PLANNED,
        route=route,
        priority=_priority(route),
        reason=_reason(route, drill_preview),
        rollback_drill_id=_str(drill_preview.get("id")),
        trigger=trigger,
        operation_ids=operation_ids,
        failed_audit_step_ids=failed_step_ids,
        receipt_id=_str(drill_preview.get("receipt_id")),
        token_id=_str(drill_preview.get("token_id")),
        lock_id=_str(drill_preview.get("lock_id")),
        recommended_actions=recommended_actions,
        required_preconditions=required_preconditions,
        blocked_actions=blocked_actions,
        next_tool_hint=_next_tool_hint(route),
        decision_digest=decision_digest,
        source_rollback_drill=dict(drill_preview),
    )


def _route_for_drill(drill_preview: Mapping[str, Any]) -> str:
    trigger = _str(drill_preview.get("trigger"))
    future_restore_steps = [
        step
        for step in _list(drill_preview.get("steps"))
        if isinstance(step, Mapping) and step.get("future_would_restore_memory") is True
    ]
    failed_steps = _list_of_str(drill_preview.get("failed_audit_step_ids"))
    if trigger == ROLLBACK_DRILL_TRIGGER_PREPAREDNESS:
        return RECOVERY_ROUTE_PREPAREDNESS_ONLY
    if trigger == ROLLBACK_DRILL_TRIGGER_AUDIT_FAILURE and future_restore_steps:
        return RECOVERY_ROUTE_MANUAL_ROLLBACK
    if trigger == ROLLBACK_DRILL_TRIGGER_AUDIT_FAILURE or failed_steps:
        return RECOVERY_ROUTE_ISOLATE_AND_REVIEW
    return RECOVERY_ROUTE_RERUN_AUDIT


def _recommended_actions(route: str) -> list[str]:
    if route == RECOVERY_ROUTE_MANUAL_ROLLBACK:
        return [
            "preserve_failed_audit_context",
            "isolate_impacted_memory_records",
            "verify_pre_commit_snapshot_or_rollback_plan",
            "request_explicit_human_recovery_approval",
            "rerun_post_commit_audit_after_manual_recovery",
        ]
    if route == RECOVERY_ROUTE_ISOLATE_AND_REVIEW:
        return [
            "preserve_failed_audit_context",
            "isolate_impacted_memory_records",
            "inspect_failed_audit_signals",
            "choose_manual_rollback_or_patch_regeneration",
        ]
    if route == RECOVERY_ROUTE_PREPAREDNESS_ONLY:
        return [
            "keep_rollback_drill_available",
            "verify_pre_commit_snapshot_or_rollback_plan_before_future_commit",
            "rerun_post_commit_audit_after_future_commit",
        ]
    return ["rerun_post_commit_audit"]


def _required_preconditions(route: str) -> list[str]:
    common = [
        "no_automatic_memory_mutation",
        "explicit_human_approval_before_recovery_execution",
    ]
    if route == RECOVERY_ROUTE_MANUAL_ROLLBACK:
        return [
            *common,
            "failed_audit_context_preserved",
            "impacted_records_isolated",
            "snapshot_or_rollback_plan_verified",
        ]
    if route == RECOVERY_ROUTE_ISOLATE_AND_REVIEW:
        return [
            *common,
            "failed_audit_context_preserved",
            "impacted_records_isolated",
        ]
    if route == RECOVERY_ROUTE_PREPAREDNESS_ONLY:
        return [
            "no_automatic_memory_mutation",
            "rollback_drill_kept_read_only",
        ]
    return common


def _blocked_actions(route: str) -> list[str]:
    blocked = [
        "automatic_memory_rollback",
        "automatic_receipt_rewrite",
        "automatic_token_state_change",
        "automatic_lock_state_change",
    ]
    if route in {RECOVERY_ROUTE_MANUAL_ROLLBACK, RECOVERY_ROUTE_ISOLATE_AND_REVIEW}:
        blocked.append("future_memory_write_until_human_review")
    return blocked


def _next_tool_hint(route: str) -> str:
    if route == RECOVERY_ROUTE_MANUAL_ROLLBACK:
        return "memory_evidence_repair_recovery_execution_preview"
    if route == RECOVERY_ROUTE_ISOLATE_AND_REVIEW:
        return "memory_evidence_repair_post_commit_audit_preview"
    if route == RECOVERY_ROUTE_PREPAREDNESS_ONLY:
        return "memory_evidence_repair_post_commit_audit_preview"
    return "memory_evidence_repair_post_commit_audit_preview"


def _priority(route: str) -> str:
    if route == RECOVERY_ROUTE_MANUAL_ROLLBACK:
        return "p0"
    if route == RECOVERY_ROUTE_ISOLATE_AND_REVIEW:
        return "p1"
    return "p2"


def _reason(route: str, drill_preview: Mapping[str, Any]) -> str:
    failed_count = len(_list_of_str(drill_preview.get("failed_audit_step_ids")))
    if route == RECOVERY_ROUTE_MANUAL_ROLLBACK:
        return f"{failed_count} failed audit step(s) require isolated manual rollback planning."
    if route == RECOVERY_ROUTE_ISOLATE_AND_REVIEW:
        return f"{failed_count} failed audit step(s) require isolation and human review."
    if route == RECOVERY_ROUTE_PREPAREDNESS_ONLY:
        return "No observed audit failure; keep the rollback drill as preparedness coverage."
    return "Recovery can proceed by rerunning the post-commit audit."


def _extract_drill_preview(rollback_drill: Mapping[str, Any]) -> dict[str, Any]:
    previews = rollback_drill.get("previews")
    if isinstance(previews, list):
        for preview in previews:
            if isinstance(preview, Mapping):
                return dict(preview)
    if _str(rollback_drill.get("id")).startswith("rollback-drill-"):
        return dict(rollback_drill)
    return {}


def _empty_report(
    *,
    rollback_drill: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryDecisionGateReport:
    return MemoryEvidenceRepairRecoveryDecisionGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_DECISION_STATUS_NO_ACTION_NEEDED,
        checks=(),
        decisions=(),
        blocking_reasons=(),
        required_actions=(),
        source_rollback_drill_report=dict(rollback_drill),
    )


def _blocked_report(
    *,
    rollback_drill: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryDecisionCheck, ...],
) -> MemoryEvidenceRepairRecoveryDecisionGateReport:
    return MemoryEvidenceRepairRecoveryDecisionGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_DECISION_STATUS_BLOCKED,
        checks=checks,
        decisions=(),
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
        source_rollback_drill_report=dict(rollback_drill),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryDecisionCheck:
    return MemoryEvidenceRepairRecoveryDecisionCheck(
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
) -> MemoryEvidenceRepairRecoveryDecisionCheck:
    return MemoryEvidenceRepairRecoveryDecisionCheck(
        id=check_id,
        status=CHECK_STATUS_FAIL,
        reason=reason,
        required_actions=tuple(required_actions),
        details=dict(details),
    )


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
