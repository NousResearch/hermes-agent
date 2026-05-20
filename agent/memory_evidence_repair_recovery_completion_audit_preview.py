"""Read-only recovery completion audit preview for memory evidence repair.

The recovery completion audit preview turns a recovery completion receipt into
an ordered verification plan for after a future manual recovery. It can also
validate optional observed recovery-completion state. It never writes to durable
memory stores.
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
from agent.memory_evidence_repair_recovery_completion_receipt import (
    RECOVERY_COMPLETION_RECEIPT_SCOPE,
    RECOVERY_COMPLETION_RECEIPT_STATUS_NO_ACTION_NEEDED,
    RECOVERY_COMPLETION_RECEIPT_STATUS_READY,
    RECOVERY_COMPLETION_RECEIPT_TYPE,
)


RECOVERY_COMPLETION_AUDIT_STATUS_READY = "recovery_completion_audit_preview_ready"
RECOVERY_COMPLETION_AUDIT_STATUS_BLOCKED = "blocked"
RECOVERY_COMPLETION_AUDIT_STATUS_NO_ACTION_NEEDED = "no_action_needed"

RECOVERY_AUDIT_STEP_STATUS_PLANNED = "planned"
RECOVERY_AUDIT_STEP_STATUS_PASS = "pass"
RECOVERY_AUDIT_STEP_STATUS_FAIL = "fail"

CHECK_RECOVERY_COMPLETION_RECEIPT_READY = "recovery_completion_receipt_ready"
CHECK_RECOVERY_COMPLETION_RECEIPT_INTEGRITY = "recovery_completion_receipt_integrity"
CHECK_RECOVERY_COMPLETION_AUDIT_REQUIREMENTS = "recovery_completion_audit_requirements"
CHECK_OBSERVED_RECOVERY_COMPLETION_STATE = "observed_recovery_completion_state"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryCompletionAuditCheck:
    """One read-only recovery completion audit readiness check."""

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
class MemoryEvidenceRepairRecoveryCompletionAuditStep:
    """One future recovery completion audit step preview."""

    id: str
    sequence: int
    category: str
    assertion: str
    operation_id: str
    recovery_step_id: str
    receipt_id: str
    token_id: str
    lock_id: str
    expected_signal: dict[str, Any]
    observed_signal: dict[str, Any]
    status: str
    required_action_on_fail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "category": self.category,
            "assertion": self.assertion,
            "operation_id": self.operation_id,
            "recovery_step_id": self.recovery_step_id,
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "expected_signal": dict(self.expected_signal),
            "observed_signal": dict(self.observed_signal),
            "status": self.status,
            "required_action_on_fail": self.required_action_on_fail,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryCompletionAuditPreview:
    """One read-only recovery completion audit preview."""

    id: str
    status: str
    receipt_id: str
    executor_preview_id: str
    decision_id: str
    execution_preview_id: str
    lock_id: str
    token_id: str
    route: str
    operation_ids: tuple[str, ...]
    recovery_step_ids: tuple[str, ...]
    future_mutation_step_ids: tuple[str, ...]
    audit_step_ids: tuple[str, ...]
    audit_digest: str
    audit_preview: str
    audit_steps: tuple[MemoryEvidenceRepairRecoveryCompletionAuditStep, ...]
    observed_state_supplied: bool
    would_verify_completion_receipt_recorded: bool
    would_verify_recovery_token_used: bool
    would_verify_recovery_lock_released: bool
    would_verify_recovery_steps_completed: bool
    would_verify_post_recovery_audit_clear: bool
    would_verify_no_secondary_memory_contamination: bool
    safety_note: str
    source_recovery_completion_receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "receipt_id": self.receipt_id,
            "executor_preview_id": self.executor_preview_id,
            "decision_id": self.decision_id,
            "execution_preview_id": self.execution_preview_id,
            "lock_id": self.lock_id,
            "token_id": self.token_id,
            "route": self.route,
            "operation_ids": list(self.operation_ids),
            "recovery_step_ids": list(self.recovery_step_ids),
            "future_mutation_step_ids": list(self.future_mutation_step_ids),
            "audit_step_ids": list(self.audit_step_ids),
            "audit_digest": self.audit_digest,
            "audit_preview": self.audit_preview,
            "audit_steps": [step.to_dict() for step in self.audit_steps],
            "observed_state_supplied": self.observed_state_supplied,
            "would_verify_completion_receipt_recorded": (
                self.would_verify_completion_receipt_recorded
            ),
            "would_verify_recovery_token_used": self.would_verify_recovery_token_used,
            "would_verify_recovery_lock_released": self.would_verify_recovery_lock_released,
            "would_verify_recovery_steps_completed": (
                self.would_verify_recovery_steps_completed
            ),
            "would_verify_post_recovery_audit_clear": (
                self.would_verify_post_recovery_audit_clear
            ),
            "would_verify_no_secondary_memory_contamination": (
                self.would_verify_no_secondary_memory_contamination
            ),
            "safety_note": self.safety_note,
            "source_recovery_completion_receipt": dict(
                self.source_recovery_completion_receipt
            ),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport:
    """Read-only recovery completion audit preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryCompletionAuditCheck, ...]
    previews: tuple[MemoryEvidenceRepairRecoveryCompletionAuditPreview, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_completion_receipt_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        by_audit_step_status: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        for preview in self.previews:
            for step in preview.audit_steps:
                by_audit_step_status[step.status] = (
                    by_audit_step_status.get(step.status, 0) + 1
                )
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "preview_count": len(self.previews),
            "audit_step_count": sum(len(preview.audit_steps) for preview in self.previews),
            "planned_audit_step_count": by_audit_step_status.get(
                RECOVERY_AUDIT_STEP_STATUS_PLANNED,
                0,
            ),
            "observed_pass_step_count": by_audit_step_status.get(
                RECOVERY_AUDIT_STEP_STATUS_PASS,
                0,
            ),
            "observed_fail_step_count": by_audit_step_status.get(
                RECOVERY_AUDIT_STEP_STATUS_FAIL,
                0,
            ),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_completion_audit_preview_available": bool(self.previews),
            "recovery_completion_audit_ready": (
                self.status == RECOVERY_COMPLETION_AUDIT_STATUS_READY
            ),
            "has_blocks": self.status == RECOVERY_COMPLETION_AUDIT_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
            "by_audit_step_status": by_audit_step_status,
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
            "source_recovery_completion_receipt_report": dict(
                self.source_recovery_completion_receipt_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_completion_audit_preview(
    *,
    recovery_completion_receipt: Mapping[str, Any] | None = None,
    observed_recovery_steps: Mapping[str, Any] | None = None,
    recorded_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    used_token_ids: Sequence[str] | None = None,
    released_lock_ids: Sequence[str] | None = None,
    post_recovery_audit_status: Mapping[str, Any] | None = None,
    contamination_status: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport:
    """Build a read-only recovery completion audit preview from a receipt."""

    recovery_completion_receipt = (
        recovery_completion_receipt
        if isinstance(recovery_completion_receipt, Mapping)
        else {}
    )
    report_status = _str(recovery_completion_receipt.get("status"))
    receipt = _extract_receipt(recovery_completion_receipt)
    observed = _observed_context(
        observed_recovery_steps=observed_recovery_steps,
        recorded_receipts=recorded_receipts,
        used_token_ids=used_token_ids,
        released_lock_ids=released_lock_ids,
        post_recovery_audit_status=post_recovery_audit_status,
        contamination_status=contamination_status,
    )

    if not receipt:
        if (
            not recovery_completion_receipt
            or report_status == RECOVERY_COMPLETION_RECEIPT_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(recovery_completion_receipt=recovery_completion_receipt)
        source_blocking = tuple(_list_of_str(recovery_completion_receipt.get("blocking_reasons")))
        source_actions = tuple(_list_of_str(recovery_completion_receipt.get("required_actions")))
        return _blocked_report(
            recovery_completion_receipt=recovery_completion_receipt,
            checks=(
                _fail(
                    CHECK_RECOVERY_COMPLETION_RECEIPT_READY,
                    "No recovery completion receipt was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions or ("produce_recovery_completion_receipt",),
                    {"recovery_completion_receipt_status": report_status},
                ),
            ),
        )

    audit_steps = tuple(_audit_steps_from_receipt(receipt, observed))
    checks = (
        _receipt_ready_check(recovery_completion_receipt, receipt),
        _receipt_integrity_check(receipt),
        _audit_requirements_check(receipt, audit_steps),
        _observed_recovery_completion_state_check(observed, audit_steps),
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
        return MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_COMPLETION_AUDIT_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_completion_receipt_report=dict(recovery_completion_receipt),
        )

    preview = _preview_from_receipt(
        receipt=receipt,
        audit_steps=audit_steps,
        observed_state_supplied=observed["supplied"],
    )
    return MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_COMPLETION_AUDIT_STATUS_READY,
        checks=checks,
        previews=(preview,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_completion_receipt_report=dict(recovery_completion_receipt),
    )


def empty_evidence_repair_recovery_completion_audit_preview() -> MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport:
    """Return an empty read-only recovery completion audit preview."""

    return _empty_report(recovery_completion_receipt={})


def _extract_receipt(recovery_completion_receipt: Mapping[str, Any]) -> dict[str, Any]:
    receipts = recovery_completion_receipt.get("receipts")
    if isinstance(receipts, list):
        for receipt in receipts:
            if isinstance(receipt, Mapping):
                return dict(receipt)
    if _str(recovery_completion_receipt.get("id")).startswith(
        "recovery-completion-receipt-"
    ):
        return dict(recovery_completion_receipt)
    return {}


def _receipt_ready_check(
    recovery_completion_receipt: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryCompletionAuditCheck:
    report_status = _str(recovery_completion_receipt.get("status"))
    receipt_status = _str(receipt.get("status"))
    if report_status and report_status != RECOVERY_COMPLETION_RECEIPT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_COMPLETION_RECEIPT_READY,
            "Recovery completion receipt report is not ready for audit planning.",
            tuple(_list_of_str(recovery_completion_receipt.get("required_actions")))
            or ("produce_ready_recovery_completion_receipt",),
            {"report_status": report_status, "receipt_status": receipt_status},
        )
    if receipt_status != RECOVERY_COMPLETION_RECEIPT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_COMPLETION_RECEIPT_READY,
            "Recovery completion receipt entry is not ready.",
            ("produce_ready_recovery_completion_receipt",),
            {"report_status": report_status, "receipt_status": receipt_status},
        )
    return _pass(
        CHECK_RECOVERY_COMPLETION_RECEIPT_READY,
        "Recovery completion receipt is ready for audit planning.",
        {"receipt_id": _str(receipt.get("id"))},
    )


def _receipt_integrity_check(
    receipt: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryCompletionAuditCheck:
    expected_digest = _expected_receipt_digest(receipt)
    actual_digest = _str(receipt.get("receipt_digest"))
    expected_id = (
        f"recovery-completion-receipt-{expected_digest[:16]}"
        if expected_digest
        else ""
    )
    missing_fields: list[str] = []
    for field_name in (
        "id",
        "token_id",
        "lock_id",
        "decision_id",
        "execution_preview_id",
        "executor_preview_id",
        "operation_ids",
        "recovery_step_ids",
        "future_mutation_step_ids",
        "executor_step_ids",
        "executor_preview_digest",
    ):
        value = receipt.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(receipt.get("id")) != expected_id
        or _str(receipt.get("receipt_type")) != RECOVERY_COMPLETION_RECEIPT_TYPE
        or _str(receipt.get("scope")) != RECOVERY_COMPLETION_RECEIPT_SCOPE
    ):
        return _fail(
            CHECK_RECOVERY_COMPLETION_RECEIPT_INTEGRITY,
            "Recovery completion receipt digest, id, type, scope, or required fields are invalid.",
            ("regenerate_recovery_completion_receipt",),
            {
                "expected_receipt_digest": expected_digest,
                "actual_receipt_digest": actual_digest,
                "expected_receipt_id": expected_id,
                "actual_receipt_id": _str(receipt.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_COMPLETION_RECEIPT_INTEGRITY,
        "Recovery completion receipt integrity checks pass.",
        {"receipt_id": expected_id, "receipt_digest": actual_digest},
    )


def _audit_requirements_check(
    receipt: Mapping[str, Any],
    audit_steps: Sequence[MemoryEvidenceRepairRecoveryCompletionAuditStep],
) -> MemoryEvidenceRepairRecoveryCompletionAuditCheck:
    categories = {step.category for step in audit_steps}
    required = {
        "completion_receipt",
        "token",
        "lock",
        "recovery_step",
        "post_recovery_audit",
        "contamination_guard",
    }
    missing = sorted(required - categories)
    if missing or not _list_of_str(receipt.get("future_mutation_step_ids")):
        return _fail(
            CHECK_RECOVERY_COMPLETION_AUDIT_REQUIREMENTS,
            "Recovery completion audit requirements are incomplete.",
            ("regenerate_recovery_completion_receipt_with_audit_requirements",),
            {
                "missing_categories": missing,
                "future_mutation_step_count": len(
                    _list_of_str(receipt.get("future_mutation_step_ids"))
                ),
            },
        )
    return _pass(
        CHECK_RECOVERY_COMPLETION_AUDIT_REQUIREMENTS,
        "Recovery completion audit covers receipt, token, lock, recovery steps, post-audit, and contamination guard checks.",
        {
            "audit_step_count": len(audit_steps),
            "future_mutation_step_count": len(
                _list_of_str(receipt.get("future_mutation_step_ids"))
            ),
        },
    )


def _observed_recovery_completion_state_check(
    observed: Mapping[str, Any],
    audit_steps: Sequence[MemoryEvidenceRepairRecoveryCompletionAuditStep],
) -> MemoryEvidenceRepairRecoveryCompletionAuditCheck:
    if not observed.get("supplied"):
        return _pass(
            CHECK_OBSERVED_RECOVERY_COMPLETION_STATE,
            "No observed recovery-completion state supplied; audit remains a planned preview.",
            {"observed_state_supplied": False},
        )
    failures = [
        step.to_dict()
        for step in audit_steps
        if step.status == RECOVERY_AUDIT_STEP_STATUS_FAIL
    ]
    if failures:
        return _fail(
            CHECK_OBSERVED_RECOVERY_COMPLETION_STATE,
            "Observed recovery-completion state does not satisfy the audit preview.",
            ("investigate_recovery_completion_audit_failures",),
            {"observed_state_supplied": True, "failures": failures},
        )
    return _pass(
        CHECK_OBSERVED_RECOVERY_COMPLETION_STATE,
        "Observed recovery-completion state satisfies the audit preview.",
        {"observed_state_supplied": True},
    )


def _preview_from_receipt(
    *,
    receipt: Mapping[str, Any],
    audit_steps: Sequence[MemoryEvidenceRepairRecoveryCompletionAuditStep],
    observed_state_supplied: bool,
) -> MemoryEvidenceRepairRecoveryCompletionAuditPreview:
    audit_step_ids = tuple(step.id for step in audit_steps)
    operation_ids = tuple(_list_of_str(receipt.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(receipt.get("recovery_step_ids")))
    future_mutation_step_ids = tuple(
        _list_of_str(receipt.get("future_mutation_step_ids"))
    )
    audit_seed = {
        "receipt_id": _str(receipt.get("id")),
        "executor_preview_id": _str(receipt.get("executor_preview_id")),
        "decision_id": _str(receipt.get("decision_id")),
        "execution_preview_id": _str(receipt.get("execution_preview_id")),
        "lock_id": _str(receipt.get("lock_id")),
        "token_id": _str(receipt.get("token_id")),
        "route": _str(receipt.get("route")),
        "operation_ids": operation_ids,
        "recovery_step_ids": recovery_step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "audit_step_ids": audit_step_ids,
        "observed_state_supplied": observed_state_supplied,
    }
    audit_digest = _digest(audit_seed)
    audit_id = f"recovery-completion-audit-{audit_digest[:16]}"
    return MemoryEvidenceRepairRecoveryCompletionAuditPreview(
        id=audit_id,
        status=RECOVERY_COMPLETION_AUDIT_STATUS_READY,
        receipt_id=_str(receipt.get("id")),
        executor_preview_id=_str(receipt.get("executor_preview_id")),
        decision_id=_str(receipt.get("decision_id")),
        execution_preview_id=_str(receipt.get("execution_preview_id")),
        lock_id=_str(receipt.get("lock_id")),
        token_id=_str(receipt.get("token_id")),
        route=_str(receipt.get("route")),
        operation_ids=operation_ids,
        recovery_step_ids=recovery_step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        audit_step_ids=audit_step_ids,
        audit_digest=audit_digest,
        audit_preview=f"{audit_id}:{audit_digest[:12]}",
        audit_steps=tuple(audit_steps),
        observed_state_supplied=observed_state_supplied,
        would_verify_completion_receipt_recorded=True,
        would_verify_recovery_token_used=True,
        would_verify_recovery_lock_released=True,
        would_verify_recovery_steps_completed=True,
        would_verify_post_recovery_audit_clear=True,
        would_verify_no_secondary_memory_contamination=True,
        safety_note=(
            "Read-only recovery completion audit preview. It plans or checks "
            "future recovery verification signals but does not read or write "
            "durable memory."
        ),
        source_recovery_completion_receipt=dict(receipt),
    )


def _audit_steps_from_receipt(
    receipt: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> list[MemoryEvidenceRepairRecoveryCompletionAuditStep]:
    steps: list[MemoryEvidenceRepairRecoveryCompletionAuditStep] = []
    steps.append(
        _audit_step(
            sequence=1,
            category="completion_receipt",
            assertion="recovery_completion_receipt_recorded",
            receipt=receipt,
            expected_signal={"receipt_id": _str(receipt.get("id"))},
            observed_signal={"recorded": _receipt_recorded(receipt, observed)},
            status=_boolean_status(_receipt_recorded(receipt, observed), observed["supplied"]),
            required_action_on_fail="record_or_recover_recovery_completion_receipt",
        )
    )
    steps.append(
        _audit_step(
            sequence=2,
            category="token",
            assertion="recovery_approval_token_marked_used",
            receipt=receipt,
            expected_signal={"token_id": _str(receipt.get("token_id"))},
            observed_signal={"used": _token_used(receipt, observed)},
            status=_boolean_status(_token_used(receipt, observed), observed["supplied"]),
            required_action_on_fail="mark_recovery_token_used_or_block_reuse",
        )
    )
    steps.append(
        _audit_step(
            sequence=3,
            category="lock",
            assertion="recovery_write_lock_released",
            receipt=receipt,
            expected_signal={"lock_id": _str(receipt.get("lock_id"))},
            observed_signal={"released": _lock_released(receipt, observed)},
            status=_boolean_status(_lock_released(receipt, observed), observed["supplied"]),
            required_action_on_fail="release_recovery_write_lock_or_mark_expired",
        )
    )
    sequence = 4
    observed_steps = _dict(observed.get("recovery_steps"))
    for recovery_step_id in _list_of_str(receipt.get("recovery_step_ids")):
        observed_signal = _dict(observed_steps.get(recovery_step_id))
        completed = _recovery_step_completed(observed_signal)
        steps.append(
            _audit_step(
                sequence=sequence,
                category="recovery_step",
                assertion="recovery_step_completed",
                receipt=receipt,
                recovery_step_id=recovery_step_id,
                expected_signal={"recovery_step_id": recovery_step_id, "completed": True},
                observed_signal=observed_signal,
                status=_boolean_status(completed, observed["supplied"]),
                required_action_on_fail="inspect_recovery_step_or_reopen_recovery_plan",
            )
        )
        sequence += 1
    post_status = _post_recovery_audit_clear(observed)
    steps.append(
        _audit_step(
            sequence=sequence,
            category="post_recovery_audit",
            assertion="post_recovery_audit_clear",
            receipt=receipt,
            expected_signal={"post_recovery_audit_clear": True},
            observed_signal=_dict(observed.get("post_recovery_audit_status")),
            status=_boolean_status(post_status, observed["supplied"]),
            required_action_on_fail="rerun_or_investigate_post_recovery_audit",
        )
    )
    sequence += 1
    contamination_clear = _contamination_clear(observed)
    steps.append(
        _audit_step(
            sequence=sequence,
            category="contamination_guard",
            assertion="no_secondary_memory_contamination",
            receipt=receipt,
            expected_signal={"secondary_memory_contamination": False},
            observed_signal=_dict(observed.get("contamination_status")),
            status=_boolean_status(contamination_clear, observed["supplied"]),
            required_action_on_fail="isolate_secondary_memory_contamination",
        )
    )
    return steps


def _audit_step(
    *,
    sequence: int,
    category: str,
    assertion: str,
    receipt: Mapping[str, Any],
    expected_signal: Mapping[str, Any],
    observed_signal: Mapping[str, Any],
    status: str,
    required_action_on_fail: str,
    recovery_step_id: str = "",
    operation_id: str = "",
) -> MemoryEvidenceRepairRecoveryCompletionAuditStep:
    return MemoryEvidenceRepairRecoveryCompletionAuditStep(
        id=f"recovery-completion-audit-step-{sequence}-{category}",
        sequence=sequence,
        category=category,
        assertion=assertion,
        operation_id=operation_id,
        recovery_step_id=recovery_step_id,
        receipt_id=_str(receipt.get("id")),
        token_id=_str(receipt.get("token_id")),
        lock_id=_str(receipt.get("lock_id")),
        expected_signal=dict(expected_signal),
        observed_signal=dict(observed_signal),
        status=status,
        required_action_on_fail=required_action_on_fail,
    )


def _observed_context(
    *,
    observed_recovery_steps: Mapping[str, Any] | None,
    recorded_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    used_token_ids: Sequence[str] | None,
    released_lock_ids: Sequence[str] | None,
    post_recovery_audit_status: Mapping[str, Any] | None,
    contamination_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    recovery_steps = _dict(observed_recovery_steps)
    receipt_ids = _receipt_ids(recorded_receipts)
    used_tokens = tuple(_list_of_str(used_token_ids))
    released_locks = tuple(_list_of_str(released_lock_ids))
    post_audit = _dict(post_recovery_audit_status)
    contamination = _dict(contamination_status)
    supplied = bool(
        recovery_steps
        or receipt_ids
        or used_tokens
        or released_locks
        or post_audit
        or contamination
    )
    return {
        "supplied": supplied,
        "recovery_steps": recovery_steps,
        "receipt_ids": receipt_ids,
        "used_token_ids": used_tokens,
        "released_lock_ids": released_locks,
        "post_recovery_audit_status": post_audit,
        "contamination_status": contamination,
    }


def _boolean_status(value: bool, observed_supplied: bool) -> str:
    if not observed_supplied:
        return RECOVERY_AUDIT_STEP_STATUS_PLANNED
    return RECOVERY_AUDIT_STEP_STATUS_PASS if value else RECOVERY_AUDIT_STEP_STATUS_FAIL


def _receipt_recorded(receipt: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return _str(receipt.get("id")) in set(observed.get("receipt_ids") or ())


def _token_used(receipt: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return _str(receipt.get("token_id")) in set(observed.get("used_token_ids") or ())


def _lock_released(receipt: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return _str(receipt.get("lock_id")) in set(observed.get("released_lock_ids") or ())


def _recovery_step_completed(observed_signal: Mapping[str, Any]) -> bool:
    if not observed_signal:
        return False
    status = _str(observed_signal.get("status")).lower()
    return observed_signal.get("completed") is True or status in {
        "complete",
        "completed",
        "success",
        "succeeded",
        "pass",
        "passed",
        "verified",
    }


def _post_recovery_audit_clear(observed: Mapping[str, Any]) -> bool:
    post_audit = _dict(observed.get("post_recovery_audit_status"))
    if not post_audit:
        return False
    status = _str(post_audit.get("status")).lower()
    return post_audit.get("clear") is True or status in {
        "clear",
        "clean",
        "pass",
        "passed",
        "success",
        "verified",
        "no_issues",
    }


def _contamination_clear(observed: Mapping[str, Any]) -> bool:
    contamination = _dict(observed.get("contamination_status"))
    if not contamination:
        return False
    status = _str(contamination.get("status")).lower()
    if contamination.get("contaminated") is False:
        return True
    return status in {"clear", "clean", "no_contamination", "verified", "pass"}


def _receipt_ids(recorded_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> tuple[str, ...]:
    ids: list[str] = []
    if isinstance(recorded_receipts, Mapping):
        ids.extend(_list_of_str(recorded_receipts.get("receipt_ids")))
        rows = recorded_receipts.get("receipts")
        if isinstance(rows, list):
            ids.extend(_receipt_ids(rows))
        if _str(recorded_receipts.get("id")):
            ids.append(_str(recorded_receipts.get("id")))
        if _str(recorded_receipts.get("receipt_id")):
            ids.append(_str(recorded_receipts.get("receipt_id")))
    elif isinstance(recorded_receipts, list):
        for receipt in recorded_receipts:
            if isinstance(receipt, Mapping):
                ids.append(_str(receipt.get("id")) or _str(receipt.get("receipt_id")))
    return tuple(_dedupe_strings(ids))


def _expected_receipt_digest(receipt: Mapping[str, Any]) -> str:
    source_preview = _dict(receipt.get("source_recovery_executor_preview"))
    seed = {
        "receipt_type": RECOVERY_COMPLETION_RECEIPT_TYPE,
        "scope": RECOVERY_COMPLETION_RECEIPT_SCOPE,
        "actor": _str(receipt.get("actor")) or "human",
        "recovery_reason": _str(receipt.get("recovery_reason")),
        "token_id": _str(receipt.get("token_id")),
        "lock_id": _str(receipt.get("lock_id")),
        "decision_id": _str(receipt.get("decision_id")),
        "execution_preview_id": _str(receipt.get("execution_preview_id")),
        "executor_preview_id": _str(receipt.get("executor_preview_id")),
        "route": _str(receipt.get("route")),
        "operation_ids": tuple(_list_of_str(receipt.get("operation_ids"))),
        "recovery_step_ids": tuple(_list_of_str(receipt.get("recovery_step_ids"))),
        "future_mutation_step_ids": tuple(
            _list_of_str(receipt.get("future_mutation_step_ids"))
        ),
        "executor_step_ids": tuple(_list_of_str(receipt.get("executor_step_ids"))),
        "executor_preview_digest": _str(receipt.get("executor_preview_digest")),
        "preview_status": _str(source_preview.get("status")),
    }
    return _digest(seed)


def _empty_report(
    *,
    recovery_completion_receipt: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport:
    return MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_COMPLETION_AUDIT_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_completion_receipt_report=dict(recovery_completion_receipt),
    )


def _blocked_report(
    *,
    recovery_completion_receipt: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryCompletionAuditCheck, ...],
) -> MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport:
    return MemoryEvidenceRepairRecoveryCompletionAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_COMPLETION_AUDIT_STATUS_BLOCKED,
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
        source_recovery_completion_receipt_report=dict(recovery_completion_receipt),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryCompletionAuditCheck:
    return MemoryEvidenceRepairRecoveryCompletionAuditCheck(
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
) -> MemoryEvidenceRepairRecoveryCompletionAuditCheck:
    return MemoryEvidenceRepairRecoveryCompletionAuditCheck(
        id=check_id,
        status=CHECK_STATUS_FAIL,
        reason=reason,
        required_actions=tuple(required_actions),
        details=dict(details),
    )


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
