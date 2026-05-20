"""Read-only recovery closure gate for memory evidence repair.

The recovery closure gate verifies a recovery completion audit preview and only
allows a future closure draft when observed recovery-completion state has been
supplied and every audit step passes. It never writes durable memory and never
marks recovery as closed.
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
from agent.memory_evidence_repair_recovery_completion_audit_preview import (
    RECOVERY_AUDIT_STEP_STATUS_FAIL,
    RECOVERY_AUDIT_STEP_STATUS_PASS,
    RECOVERY_AUDIT_STEP_STATUS_PLANNED,
    RECOVERY_COMPLETION_AUDIT_STATUS_NO_ACTION_NEEDED,
    RECOVERY_COMPLETION_AUDIT_STATUS_READY,
)


RECOVERY_CLOSURE_STATUS_READY = "recovery_closure_ready"
RECOVERY_CLOSURE_STATUS_BLOCKED = "blocked"
RECOVERY_CLOSURE_STATUS_NO_ACTION_NEEDED = "no_action_needed"
RECOVERY_CLOSURE_DRAFT_STATUS_READY = "recovery_closure_draft_ready"

CHECK_RECOVERY_COMPLETION_AUDIT_READY = "recovery_completion_audit_ready"
CHECK_RECOVERY_COMPLETION_AUDIT_INTEGRITY = "recovery_completion_audit_integrity"
CHECK_RECOVERY_COMPLETION_AUDIT_OBSERVED_PASS = "recovery_completion_audit_observed_pass"
CHECK_RECOVERY_CLOSURE_REQUIREMENTS = "recovery_closure_requirements"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureCheck:
    """One read-only recovery closure gate check."""

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
class MemoryEvidenceRepairRecoveryClosureDraft:
    """One read-only future recovery closure draft."""

    id: str
    status: str
    closure_type: str
    closure_decision: str
    audit_preview_id: str
    receipt_id: str
    executor_preview_id: str
    decision_id: str
    execution_preview_id: str
    token_id: str
    lock_id: str
    route: str
    operation_ids: tuple[str, ...]
    recovery_step_ids: tuple[str, ...]
    future_mutation_step_ids: tuple[str, ...]
    audit_step_ids: tuple[str, ...]
    closure_digest: str
    closure_preview: str
    would_close_recovery_loop: bool
    would_mark_recovery_resolved: bool
    would_preserve_audit_evidence: bool
    safety_note: str
    source_recovery_completion_audit_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "closure_type": self.closure_type,
            "closure_decision": self.closure_decision,
            "audit_preview_id": self.audit_preview_id,
            "receipt_id": self.receipt_id,
            "executor_preview_id": self.executor_preview_id,
            "decision_id": self.decision_id,
            "execution_preview_id": self.execution_preview_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "route": self.route,
            "operation_ids": list(self.operation_ids),
            "recovery_step_ids": list(self.recovery_step_ids),
            "future_mutation_step_ids": list(self.future_mutation_step_ids),
            "audit_step_ids": list(self.audit_step_ids),
            "closure_digest": self.closure_digest,
            "closure_preview": self.closure_preview,
            "would_close_recovery_loop": self.would_close_recovery_loop,
            "would_mark_recovery_resolved": self.would_mark_recovery_resolved,
            "would_preserve_audit_evidence": self.would_preserve_audit_evidence,
            "safety_note": self.safety_note,
            "source_recovery_completion_audit_preview": dict(
                self.source_recovery_completion_audit_preview
            ),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureGateReport:
    """Read-only recovery closure gate report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryClosureCheck, ...]
    closures: tuple[MemoryEvidenceRepairRecoveryClosureDraft, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_completion_audit_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "closure_count": len(self.closures),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_closure_available": bool(self.closures),
            "recovery_closure_ready": self.status == RECOVERY_CLOSURE_STATUS_READY,
            "would_close_recovery_loop_count": sum(
                1 for closure in self.closures if closure.would_close_recovery_loop
            ),
            "has_blocks": self.status == RECOVERY_CLOSURE_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "closures": [closure.to_dict() for closure in self.closures],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_recovery_completion_audit_report": dict(
                self.source_recovery_completion_audit_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_closure_gate(
    *,
    recovery_completion_audit: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryClosureGateReport:
    """Build a read-only recovery closure gate from a completion audit."""

    recovery_completion_audit = (
        recovery_completion_audit if isinstance(recovery_completion_audit, Mapping) else {}
    )
    report_status = _str(recovery_completion_audit.get("status"))
    preview = _extract_preview(recovery_completion_audit)

    if not preview:
        if (
            not recovery_completion_audit
            or report_status == RECOVERY_COMPLETION_AUDIT_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(recovery_completion_audit=recovery_completion_audit)
        source_blocking = tuple(_list_of_str(recovery_completion_audit.get("blocking_reasons")))
        source_actions = tuple(_list_of_str(recovery_completion_audit.get("required_actions")))
        return _blocked_report(
            recovery_completion_audit=recovery_completion_audit,
            checks=(
                _fail(
                    CHECK_RECOVERY_COMPLETION_AUDIT_READY,
                    "No recovery completion audit preview was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions or ("produce_recovery_completion_audit_preview",),
                    {"recovery_completion_audit_status": report_status},
                ),
            ),
        )

    checks = (
        _audit_ready_check(recovery_completion_audit, preview),
        _audit_integrity_check(preview),
        _audit_observed_pass_check(preview),
        _closure_requirements_check(preview),
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
        return MemoryEvidenceRepairRecoveryClosureGateReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_CLOSURE_STATUS_BLOCKED,
            checks=checks,
            closures=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_completion_audit_report=dict(recovery_completion_audit),
        )

    closure = _closure_from_audit(preview)
    return MemoryEvidenceRepairRecoveryClosureGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_STATUS_READY,
        checks=checks,
        closures=(closure,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_completion_audit_report=dict(recovery_completion_audit),
    )


def empty_evidence_repair_recovery_closure_gate() -> MemoryEvidenceRepairRecoveryClosureGateReport:
    """Return an empty read-only recovery closure gate report."""

    return _empty_report(recovery_completion_audit={})


def _extract_preview(recovery_completion_audit: Mapping[str, Any]) -> dict[str, Any]:
    previews = recovery_completion_audit.get("previews")
    if isinstance(previews, list):
        for preview in previews:
            if isinstance(preview, Mapping):
                return dict(preview)
    if _str(recovery_completion_audit.get("id")).startswith("recovery-completion-audit-"):
        return dict(recovery_completion_audit)
    return {}


def _audit_ready_check(
    recovery_completion_audit: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureCheck:
    report_status = _str(recovery_completion_audit.get("status"))
    preview_status = _str(preview.get("status"))
    if report_status and report_status != RECOVERY_COMPLETION_AUDIT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_COMPLETION_AUDIT_READY,
            "Recovery completion audit report is not ready for closure.",
            tuple(_list_of_str(recovery_completion_audit.get("required_actions")))
            or ("produce_ready_recovery_completion_audit_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    if preview_status != RECOVERY_COMPLETION_AUDIT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_COMPLETION_AUDIT_READY,
            "Recovery completion audit preview entry is not ready.",
            ("produce_ready_recovery_completion_audit_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    return _pass(
        CHECK_RECOVERY_COMPLETION_AUDIT_READY,
        "Recovery completion audit preview is ready for closure gating.",
        {"audit_preview_id": _str(preview.get("id"))},
    )


def _audit_integrity_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureCheck:
    expected_digest = _expected_audit_digest(preview)
    actual_digest = _str(preview.get("audit_digest"))
    expected_id = f"recovery-completion-audit-{expected_digest[:16]}" if expected_digest else ""
    missing_fields: list[str] = []
    for field_name in (
        "id",
        "receipt_id",
        "executor_preview_id",
        "decision_id",
        "execution_preview_id",
        "lock_id",
        "token_id",
        "operation_ids",
        "recovery_step_ids",
        "future_mutation_step_ids",
        "audit_step_ids",
        "audit_steps",
    ):
        value = preview.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(preview.get("id")) != expected_id
    ):
        return _fail(
            CHECK_RECOVERY_COMPLETION_AUDIT_INTEGRITY,
            "Recovery completion audit digest, id, or required fields are invalid.",
            ("regenerate_recovery_completion_audit_preview",),
            {
                "expected_audit_digest": expected_digest,
                "actual_audit_digest": actual_digest,
                "expected_audit_id": expected_id,
                "actual_audit_id": _str(preview.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_COMPLETION_AUDIT_INTEGRITY,
        "Recovery completion audit integrity checks pass.",
        {"audit_preview_id": expected_id, "audit_digest": actual_digest},
    )


def _audit_observed_pass_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureCheck:
    audit_steps = _audit_steps(preview)
    statuses = [_str(step.get("status")) for step in audit_steps]
    planned = [step for step in audit_steps if _str(step.get("status")) == RECOVERY_AUDIT_STEP_STATUS_PLANNED]
    failed = [step for step in audit_steps if _str(step.get("status")) == RECOVERY_AUDIT_STEP_STATUS_FAIL]
    passed = [step for step in audit_steps if _str(step.get("status")) == RECOVERY_AUDIT_STEP_STATUS_PASS]
    if preview.get("observed_state_supplied") is not True:
        return _fail(
            CHECK_RECOVERY_COMPLETION_AUDIT_OBSERVED_PASS,
            "Recovery closure requires observed recovery-completion state; planned audit is not enough.",
            ("provide_observed_recovery_completion_state",),
            {"observed_state_supplied": bool(preview.get("observed_state_supplied"))},
        )
    if planned or failed or len(passed) != len(audit_steps):
        return _fail(
            CHECK_RECOVERY_COMPLETION_AUDIT_OBSERVED_PASS,
            "Recovery closure requires every completion audit step to pass.",
            ("investigate_recovery_completion_audit_failures",),
            {
                "audit_step_count": len(audit_steps),
                "pass_count": len(passed),
                "planned_count": len(planned),
                "fail_count": len(failed),
                "statuses": statuses,
            },
        )
    return _pass(
        CHECK_RECOVERY_COMPLETION_AUDIT_OBSERVED_PASS,
        "Observed recovery-completion audit has passed every step.",
        {"audit_step_count": len(audit_steps), "pass_count": len(passed)},
    )


def _closure_requirements_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureCheck:
    flags = {
        "would_verify_completion_receipt_recorded": preview.get(
            "would_verify_completion_receipt_recorded"
        )
        is True,
        "would_verify_recovery_token_used": preview.get("would_verify_recovery_token_used")
        is True,
        "would_verify_recovery_lock_released": preview.get(
            "would_verify_recovery_lock_released"
        )
        is True,
        "would_verify_recovery_steps_completed": preview.get(
            "would_verify_recovery_steps_completed"
        )
        is True,
        "would_verify_post_recovery_audit_clear": preview.get(
            "would_verify_post_recovery_audit_clear"
        )
        is True,
        "would_verify_no_secondary_memory_contamination": preview.get(
            "would_verify_no_secondary_memory_contamination"
        )
        is True,
    }
    missing = [name for name, ok in flags.items() if not ok]
    required_categories = {
        "completion_receipt",
        "token",
        "lock",
        "recovery_step",
        "post_recovery_audit",
        "contamination_guard",
    }
    categories = {_str(step.get("category")) for step in _audit_steps(preview)}
    missing_categories = sorted(required_categories - categories)
    if missing or missing_categories or not _list_of_str(preview.get("future_mutation_step_ids")):
        return _fail(
            CHECK_RECOVERY_CLOSURE_REQUIREMENTS,
            "Recovery closure requirements are incomplete.",
            ("regenerate_recovery_completion_audit_preview",),
            {
                "missing_flags": missing,
                "missing_categories": missing_categories,
                "future_mutation_step_count": len(
                    _list_of_str(preview.get("future_mutation_step_ids"))
                ),
            },
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_REQUIREMENTS,
        "Recovery closure requirements cover receipt, token, lock, steps, audit, and contamination guard.",
        {
            "category_count": len(categories),
            "future_mutation_step_count": len(
                _list_of_str(preview.get("future_mutation_step_ids"))
            ),
        },
    )


def _closure_from_audit(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureDraft:
    operation_ids = tuple(_list_of_str(preview.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(preview.get("recovery_step_ids")))
    future_mutation_step_ids = tuple(_list_of_str(preview.get("future_mutation_step_ids")))
    audit_step_ids = tuple(_list_of_str(preview.get("audit_step_ids")))
    seed = {
        "closure_type": "memory_evidence_repair_recovery_closure",
        "closure_decision": "close_recovery_loop",
        "audit_preview_id": _str(preview.get("id")),
        "receipt_id": _str(preview.get("receipt_id")),
        "executor_preview_id": _str(preview.get("executor_preview_id")),
        "decision_id": _str(preview.get("decision_id")),
        "execution_preview_id": _str(preview.get("execution_preview_id")),
        "token_id": _str(preview.get("token_id")),
        "lock_id": _str(preview.get("lock_id")),
        "route": _str(preview.get("route")),
        "operation_ids": operation_ids,
        "recovery_step_ids": recovery_step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "audit_step_ids": audit_step_ids,
    }
    closure_digest = _digest(seed)
    closure_id = f"recovery-closure-{closure_digest[:16]}"
    return MemoryEvidenceRepairRecoveryClosureDraft(
        id=closure_id,
        status=RECOVERY_CLOSURE_DRAFT_STATUS_READY,
        closure_type="memory_evidence_repair_recovery_closure",
        closure_decision="close_recovery_loop",
        audit_preview_id=_str(preview.get("id")),
        receipt_id=_str(preview.get("receipt_id")),
        executor_preview_id=_str(preview.get("executor_preview_id")),
        decision_id=_str(preview.get("decision_id")),
        execution_preview_id=_str(preview.get("execution_preview_id")),
        token_id=_str(preview.get("token_id")),
        lock_id=_str(preview.get("lock_id")),
        route=_str(preview.get("route")),
        operation_ids=operation_ids,
        recovery_step_ids=recovery_step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        audit_step_ids=audit_step_ids,
        closure_digest=closure_digest,
        closure_preview=f"{closure_id}:{closure_digest[:12]}",
        would_close_recovery_loop=True,
        would_mark_recovery_resolved=True,
        would_preserve_audit_evidence=True,
        safety_note=(
            "Read-only recovery closure draft. It states that a future closure "
            "would be allowed, but does not persist closure state or mutate memory."
        ),
        source_recovery_completion_audit_preview=dict(preview),
    )


def _expected_audit_digest(preview: Mapping[str, Any]) -> str:
    seed = {
        "receipt_id": _str(preview.get("receipt_id")),
        "executor_preview_id": _str(preview.get("executor_preview_id")),
        "decision_id": _str(preview.get("decision_id")),
        "execution_preview_id": _str(preview.get("execution_preview_id")),
        "lock_id": _str(preview.get("lock_id")),
        "token_id": _str(preview.get("token_id")),
        "route": _str(preview.get("route")),
        "operation_ids": tuple(_list_of_str(preview.get("operation_ids"))),
        "recovery_step_ids": tuple(_list_of_str(preview.get("recovery_step_ids"))),
        "future_mutation_step_ids": tuple(
            _list_of_str(preview.get("future_mutation_step_ids"))
        ),
        "audit_step_ids": tuple(_list_of_str(preview.get("audit_step_ids"))),
        "observed_state_supplied": preview.get("observed_state_supplied") is True,
    }
    return _digest(seed)


def _audit_steps(preview: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        step
        for step in _list(preview.get("audit_steps"))
        if isinstance(step, Mapping)
    ]


def _empty_report(
    *,
    recovery_completion_audit: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGateReport:
    return MemoryEvidenceRepairRecoveryClosureGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_STATUS_NO_ACTION_NEEDED,
        checks=(),
        closures=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_completion_audit_report=dict(recovery_completion_audit),
    )


def _blocked_report(
    *,
    recovery_completion_audit: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryClosureCheck, ...],
) -> MemoryEvidenceRepairRecoveryClosureGateReport:
    return MemoryEvidenceRepairRecoveryClosureGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_STATUS_BLOCKED,
        checks=checks,
        closures=(),
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
        source_recovery_completion_audit_report=dict(recovery_completion_audit),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureCheck:
    return MemoryEvidenceRepairRecoveryClosureCheck(
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
) -> MemoryEvidenceRepairRecoveryClosureCheck:
    return MemoryEvidenceRepairRecoveryClosureCheck(
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
