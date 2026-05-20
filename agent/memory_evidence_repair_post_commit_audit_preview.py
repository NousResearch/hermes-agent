"""Read-only post-commit audit preview for memory evidence repair.

The post-commit audit preview turns an executor preview into an ordered
verification plan for after a future manual commit. It can also validate
optional observed post-commit state. It never writes to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_executor_preview import (
    EXECUTOR_PREVIEW_STATUS_BLOCKED,
    EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED,
    EXECUTOR_PREVIEW_STATUS_READY,
)
from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
)


POST_COMMIT_AUDIT_STATUS_READY = "post_commit_audit_preview_ready"
POST_COMMIT_AUDIT_STATUS_BLOCKED = "blocked"
POST_COMMIT_AUDIT_STATUS_NO_ACTION_NEEDED = "no_action_needed"

AUDIT_STEP_STATUS_PLANNED = "planned"
AUDIT_STEP_STATUS_PASS = "pass"
AUDIT_STEP_STATUS_FAIL = "fail"

CHECK_EXECUTOR_PREVIEW_READY = "executor_preview_ready"
CHECK_EXECUTOR_PREVIEW_INTEGRITY = "executor_preview_integrity"
CHECK_AUDIT_REQUIREMENTS = "audit_requirements"
CHECK_OBSERVED_POST_STATE = "observed_post_state"


@dataclass(frozen=True)
class MemoryEvidenceRepairPostCommitAuditCheck:
    """One read-only post-commit audit preview check."""

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
class MemoryEvidenceRepairPostCommitAuditStep:
    """One future post-commit audit step preview."""

    id: str
    sequence: int
    category: str
    assertion: str
    operation_id: str
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
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "expected_signal": dict(self.expected_signal),
            "observed_signal": dict(self.observed_signal),
            "status": self.status,
            "required_action_on_fail": self.required_action_on_fail,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairPostCommitAuditPreview:
    """One read-only post-commit audit preview."""

    id: str
    status: str
    executor_preview_id: str
    lock_id: str
    receipt_id: str
    token_id: str
    operation_ids: tuple[str, ...]
    patch_digests: tuple[str, ...]
    audit_step_ids: tuple[str, ...]
    audit_digest: str
    audit_preview: str
    audit_steps: tuple[MemoryEvidenceRepairPostCommitAuditStep, ...]
    observed_state_supplied: bool
    would_verify_memory_patches: bool
    would_verify_receipt_recorded: bool
    would_verify_token_used: bool
    would_verify_lock_released: bool
    would_verify_rollback_available: bool
    safety_note: str
    source_executor_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "executor_preview_id": self.executor_preview_id,
            "lock_id": self.lock_id,
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "operation_ids": list(self.operation_ids),
            "patch_digests": list(self.patch_digests),
            "audit_step_ids": list(self.audit_step_ids),
            "audit_digest": self.audit_digest,
            "audit_preview": self.audit_preview,
            "audit_steps": [step.to_dict() for step in self.audit_steps],
            "observed_state_supplied": self.observed_state_supplied,
            "would_verify_memory_patches": self.would_verify_memory_patches,
            "would_verify_receipt_recorded": self.would_verify_receipt_recorded,
            "would_verify_token_used": self.would_verify_token_used,
            "would_verify_lock_released": self.would_verify_lock_released,
            "would_verify_rollback_available": self.would_verify_rollback_available,
            "safety_note": self.safety_note,
            "source_executor_preview": dict(self.source_executor_preview),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairPostCommitAuditPreviewReport:
    """Read-only post-commit audit preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairPostCommitAuditCheck, ...]
    previews: tuple[MemoryEvidenceRepairPostCommitAuditPreview, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_executor_preview_report: dict[str, Any]

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
            "planned_audit_step_count": by_audit_step_status.get(AUDIT_STEP_STATUS_PLANNED, 0),
            "observed_pass_step_count": by_audit_step_status.get(AUDIT_STEP_STATUS_PASS, 0),
            "observed_fail_step_count": by_audit_step_status.get(AUDIT_STEP_STATUS_FAIL, 0),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "post_commit_audit_preview_available": bool(self.previews),
            "post_commit_audit_ready": self.status == POST_COMMIT_AUDIT_STATUS_READY,
            "has_blocks": self.status == POST_COMMIT_AUDIT_STATUS_BLOCKED,
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
            "source_executor_preview_report": dict(self.source_executor_preview_report),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_post_commit_audit_preview(
    *,
    executor_preview: Mapping[str, Any] | None = None,
    observed_memory_patches: Mapping[str, Any] | None = None,
    recorded_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    used_token_ids: Sequence[str] | None = None,
    released_lock_ids: Sequence[str] | None = None,
    rollback_status: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairPostCommitAuditPreviewReport:
    """Build a read-only post-commit audit preview from an executor preview."""

    executor_preview = executor_preview if isinstance(executor_preview, Mapping) else {}
    report_status = _str(executor_preview.get("status"))
    preview = _extract_preview(executor_preview)
    observed = _observed_context(
        observed_memory_patches=observed_memory_patches,
        recorded_receipts=recorded_receipts,
        used_token_ids=used_token_ids,
        released_lock_ids=released_lock_ids,
        rollback_status=rollback_status,
    )

    if not preview:
        if not executor_preview or report_status == EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED:
            return _empty_report(executor_preview=executor_preview)
        source_blocking = tuple(_list_of_str(executor_preview.get("blocking_reasons")))
        source_actions = tuple(_list_of_str(executor_preview.get("required_actions")))
        return _blocked_report(
            executor_preview=executor_preview,
            checks=(
                _fail(
                    CHECK_EXECUTOR_PREVIEW_READY,
                    "No executor preview was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions or ("produce_executor_preview",),
                    {"executor_preview_status": report_status},
                ),
            ),
        )

    audit_steps = tuple(_audit_steps_from_preview(preview, observed))
    checks = (
        _executor_preview_ready_check(executor_preview, preview),
        _executor_preview_integrity_check(preview),
        _audit_requirements_check(preview, audit_steps),
        _observed_post_state_check(observed, audit_steps),
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
        return MemoryEvidenceRepairPostCommitAuditPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=POST_COMMIT_AUDIT_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_executor_preview_report=dict(executor_preview),
        )

    post_preview = _preview_from_executor_preview(
        preview=preview,
        audit_steps=audit_steps,
        observed_state_supplied=observed["supplied"],
    )
    return MemoryEvidenceRepairPostCommitAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=POST_COMMIT_AUDIT_STATUS_READY,
        checks=checks,
        previews=(post_preview,),
        blocking_reasons=(),
        required_actions=(),
        source_executor_preview_report=dict(executor_preview),
    )


def empty_evidence_repair_post_commit_audit_preview() -> MemoryEvidenceRepairPostCommitAuditPreviewReport:
    """Return an empty read-only post-commit audit preview."""

    return _empty_report(executor_preview={})


def _extract_preview(executor_preview: Mapping[str, Any]) -> dict[str, Any]:
    previews = executor_preview.get("previews")
    if isinstance(previews, list):
        for preview in previews:
            if isinstance(preview, Mapping):
                return dict(preview)
    if _str(executor_preview.get("id")).startswith("executor-preview-"):
        return dict(executor_preview)
    return {}


def _executor_preview_ready_check(
    executor_preview: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairPostCommitAuditCheck:
    report_status = _str(executor_preview.get("status"))
    preview_status = _str(preview.get("status"))
    if report_status and report_status != EXECUTOR_PREVIEW_STATUS_READY:
        return _fail(
            CHECK_EXECUTOR_PREVIEW_READY,
            "Executor preview report is not ready for post-commit audit planning.",
            tuple(_list_of_str(executor_preview.get("required_actions")))
            or ("produce_ready_executor_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    if preview_status != EXECUTOR_PREVIEW_STATUS_READY:
        return _fail(
            CHECK_EXECUTOR_PREVIEW_READY,
            "Executor preview entry is not ready.",
            ("produce_ready_executor_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    return _pass(
        CHECK_EXECUTOR_PREVIEW_READY,
        "Executor preview is ready for post-commit audit planning.",
        {"executor_preview_id": _str(preview.get("id"))},
    )


def _executor_preview_integrity_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairPostCommitAuditCheck:
    expected_digest = _expected_executor_preview_digest(preview)
    actual_digest = _str(preview.get("preview_digest"))
    expected_id = f"executor-preview-{expected_digest[:16]}" if expected_digest else ""
    missing_fields: list[str] = []
    for field_name in ("lock_id", "receipt_id", "token_id", "operation_ids"):
        value = preview.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    if missing_fields or actual_digest != expected_digest or _str(preview.get("id")) != expected_id:
        return _fail(
            CHECK_EXECUTOR_PREVIEW_INTEGRITY,
            "Executor preview digest, id, or required fields are invalid.",
            ("regenerate_executor_preview",),
            {
                "expected_preview_digest": expected_digest,
                "actual_preview_digest": actual_digest,
                "expected_preview_id": expected_id,
                "actual_preview_id": _str(preview.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_EXECUTOR_PREVIEW_INTEGRITY,
        "Executor preview integrity checks pass.",
        {"executor_preview_id": expected_id, "preview_digest": actual_digest},
    )


def _audit_requirements_check(
    preview: Mapping[str, Any],
    audit_steps: Sequence[MemoryEvidenceRepairPostCommitAuditStep],
) -> MemoryEvidenceRepairPostCommitAuditCheck:
    categories = {step.category for step in audit_steps}
    required = {"memory_patch", "receipt", "token", "lock", "rollback"}
    missing = sorted(required - categories)
    write_steps = [
        step
        for step in _list(preview.get("steps"))
        if isinstance(step, Mapping) and step.get("future_would_write_memory") is True
    ]
    if missing or not write_steps:
        return _fail(
            CHECK_AUDIT_REQUIREMENTS,
            "Post-commit audit requirements are incomplete.",
            ("regenerate_executor_preview_with_audit_requirements",),
            {"missing_categories": missing, "future_write_step_count": len(write_steps)},
        )
    return _pass(
        CHECK_AUDIT_REQUIREMENTS,
        "Post-commit audit requirements cover memory patch, receipt, token, lock, and rollback checks.",
        {"audit_step_count": len(audit_steps), "future_write_step_count": len(write_steps)},
    )


def _observed_post_state_check(
    observed: Mapping[str, Any],
    audit_steps: Sequence[MemoryEvidenceRepairPostCommitAuditStep],
) -> MemoryEvidenceRepairPostCommitAuditCheck:
    if not observed.get("supplied"):
        return _pass(
            CHECK_OBSERVED_POST_STATE,
            "No observed post-commit state supplied; audit remains a planned preview.",
            {"observed_state_supplied": False},
        )
    failures = [step.to_dict() for step in audit_steps if step.status == AUDIT_STEP_STATUS_FAIL]
    if failures:
        return _fail(
            CHECK_OBSERVED_POST_STATE,
            "Observed post-commit state does not satisfy the audit preview.",
            ("investigate_post_commit_audit_failures",),
            {"observed_state_supplied": True, "failures": failures},
        )
    return _pass(
        CHECK_OBSERVED_POST_STATE,
        "Observed post-commit state satisfies the audit preview.",
        {"observed_state_supplied": True},
    )


def _preview_from_executor_preview(
    *,
    preview: Mapping[str, Any],
    audit_steps: Sequence[MemoryEvidenceRepairPostCommitAuditStep],
    observed_state_supplied: bool,
) -> MemoryEvidenceRepairPostCommitAuditPreview:
    audit_step_ids = tuple(step.id for step in audit_steps)
    operation_ids = tuple(_list_of_str(preview.get("operation_ids")))
    patch_digests = tuple(_list_of_str(preview.get("patch_digests")))
    audit_seed = {
        "executor_preview_id": _str(preview.get("id")),
        "lock_id": _str(preview.get("lock_id")),
        "receipt_id": _str(preview.get("receipt_id")),
        "token_id": _str(preview.get("token_id")),
        "operation_ids": operation_ids,
        "patch_digests": patch_digests,
        "audit_step_ids": audit_step_ids,
        "observed_state_supplied": observed_state_supplied,
    }
    audit_digest = _digest(audit_seed)
    audit_id = f"post-commit-audit-{audit_digest[:16]}"
    return MemoryEvidenceRepairPostCommitAuditPreview(
        id=audit_id,
        status=POST_COMMIT_AUDIT_STATUS_READY,
        executor_preview_id=_str(preview.get("id")),
        lock_id=_str(preview.get("lock_id")),
        receipt_id=_str(preview.get("receipt_id")),
        token_id=_str(preview.get("token_id")),
        operation_ids=operation_ids,
        patch_digests=patch_digests,
        audit_step_ids=audit_step_ids,
        audit_digest=audit_digest,
        audit_preview=f"{audit_id}:{audit_digest[:12]}",
        audit_steps=tuple(audit_steps),
        observed_state_supplied=observed_state_supplied,
        would_verify_memory_patches=True,
        would_verify_receipt_recorded=True,
        would_verify_token_used=True,
        would_verify_lock_released=True,
        would_verify_rollback_available=True,
        safety_note=(
            "Read-only post-commit audit preview. It plans or checks future "
            "verification signals but does not read or write durable memory."
        ),
        source_executor_preview=dict(preview),
    )


def _audit_steps_from_preview(
    preview: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> list[MemoryEvidenceRepairPostCommitAuditStep]:
    steps: list[MemoryEvidenceRepairPostCommitAuditStep] = []
    observed_patches = _dict(observed.get("memory_patches"))
    for sequence, write_step in enumerate(_write_steps(preview), start=1):
        operation_id = _str(write_step.get("operation_id"))
        expected = {
            "operation_id": operation_id,
            "patch_digest": _str(write_step.get("patch_digest")),
            "target": _dict(write_step.get("target")),
            "evidence_fields": _list_of_str(write_step.get("evidence_fields")),
        }
        observed_signal = _dict(observed_patches.get(operation_id))
        steps.append(
            _audit_step(
                sequence=sequence,
                category="memory_patch",
                assertion="memory_patch_applied_with_expected_digest",
                preview=preview,
                operation_id=operation_id,
                expected_signal=expected,
                observed_signal=observed_signal,
                status=_observed_patch_status(expected, observed_signal, observed["supplied"]),
                required_action_on_fail="inspect_memory_patch_and_rollback_if_needed",
            )
        )
    sequence = len(steps) + 1
    steps.append(
        _audit_step(
            sequence=sequence,
            category="receipt",
            assertion="commit_receipt_recorded",
            preview=preview,
            expected_signal={"receipt_id": _str(preview.get("receipt_id"))},
            observed_signal={"recorded": _receipt_recorded(preview, observed)},
            status=_boolean_status(_receipt_recorded(preview, observed), observed["supplied"]),
            required_action_on_fail="record_or_recover_commit_receipt",
        )
    )
    sequence += 1
    steps.append(
        _audit_step(
            sequence=sequence,
            category="token",
            assertion="approval_token_marked_used",
            preview=preview,
            expected_signal={"token_id": _str(preview.get("token_id"))},
            observed_signal={"used": _token_used(preview, observed)},
            status=_boolean_status(_token_used(preview, observed), observed["supplied"]),
            required_action_on_fail="mark_token_used_or_block_reuse",
        )
    )
    sequence += 1
    steps.append(
        _audit_step(
            sequence=sequence,
            category="lock",
            assertion="write_lock_released",
            preview=preview,
            expected_signal={"lock_id": _str(preview.get("lock_id"))},
            observed_signal={"released": _lock_released(preview, observed)},
            status=_boolean_status(_lock_released(preview, observed), observed["supplied"]),
            required_action_on_fail="release_write_lock_or_mark_lock_expired",
        )
    )
    sequence += 1
    rollback_ready = _rollback_available(observed)
    steps.append(
        _audit_step(
            sequence=sequence,
            category="rollback",
            assertion="rollback_or_snapshot_reference_available",
            preview=preview,
            expected_signal={"rollback_reference_required": True},
            observed_signal={"rollback_available": rollback_ready},
            status=_boolean_status(rollback_ready, observed["supplied"]),
            required_action_on_fail="verify_rollback_plan_or_pre_commit_snapshot",
        )
    )
    return steps


def _audit_step(
    *,
    sequence: int,
    category: str,
    assertion: str,
    preview: Mapping[str, Any],
    expected_signal: Mapping[str, Any],
    observed_signal: Mapping[str, Any],
    status: str,
    required_action_on_fail: str,
    operation_id: str = "",
) -> MemoryEvidenceRepairPostCommitAuditStep:
    return MemoryEvidenceRepairPostCommitAuditStep(
        id=f"post-commit-audit-step-{sequence}-{category}",
        sequence=sequence,
        category=category,
        assertion=assertion,
        operation_id=operation_id,
        receipt_id=_str(preview.get("receipt_id")),
        token_id=_str(preview.get("token_id")),
        lock_id=_str(preview.get("lock_id")),
        expected_signal=dict(expected_signal),
        observed_signal=dict(observed_signal),
        status=status,
        required_action_on_fail=required_action_on_fail,
    )


def _observed_context(
    *,
    observed_memory_patches: Mapping[str, Any] | None,
    recorded_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    used_token_ids: Sequence[str] | None,
    released_lock_ids: Sequence[str] | None,
    rollback_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    memory_patches = _dict(observed_memory_patches)
    receipt_ids = _receipt_ids(recorded_receipts)
    used_tokens = tuple(_list_of_str(used_token_ids))
    released_locks = tuple(_list_of_str(released_lock_ids))
    rollback = _dict(rollback_status)
    supplied = bool(memory_patches or receipt_ids or used_tokens or released_locks or rollback)
    return {
        "supplied": supplied,
        "memory_patches": memory_patches,
        "receipt_ids": receipt_ids,
        "used_token_ids": used_tokens,
        "released_lock_ids": released_locks,
        "rollback_status": rollback,
    }


def _observed_patch_status(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    observed_supplied: bool,
) -> str:
    if not observed_supplied:
        return AUDIT_STEP_STATUS_PLANNED
    if (
        observed.get("applied") is True
        and _str(observed.get("patch_digest")) == _str(expected.get("patch_digest"))
    ):
        return AUDIT_STEP_STATUS_PASS
    return AUDIT_STEP_STATUS_FAIL


def _boolean_status(value: bool, observed_supplied: bool) -> str:
    if not observed_supplied:
        return AUDIT_STEP_STATUS_PLANNED
    return AUDIT_STEP_STATUS_PASS if value else AUDIT_STEP_STATUS_FAIL


def _receipt_recorded(preview: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return _str(preview.get("receipt_id")) in set(observed.get("receipt_ids") or ())


def _token_used(preview: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return _str(preview.get("token_id")) in set(observed.get("used_token_ids") or ())


def _lock_released(preview: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return _str(preview.get("lock_id")) in set(observed.get("released_lock_ids") or ())


def _rollback_available(observed: Mapping[str, Any]) -> bool:
    rollback = _dict(observed.get("rollback_status"))
    status = _str(rollback.get("status"))
    if not rollback:
        return False
    return status in {
        "ready",
        "available",
        "verified",
        "rollback_ready",
        "snapshot_available",
    }


def _write_steps(preview: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        step
        for step in _list(preview.get("steps"))
        if isinstance(step, Mapping) and step.get("future_would_write_memory") is True
    ]


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


def _expected_executor_preview_digest(preview: Mapping[str, Any]) -> str:
    seed = {
        "lock_id": _str(preview.get("lock_id")),
        "receipt_id": _str(preview.get("receipt_id")),
        "token_id": _str(preview.get("token_id")),
        "operation_ids": tuple(_list_of_str(preview.get("operation_ids"))),
        "candidate_ids": tuple(_list_of_str(preview.get("candidate_ids"))),
        "patch_digests": tuple(_list_of_str(preview.get("patch_digests"))),
        "step_ids": tuple(_list_of_str(preview.get("step_ids"))),
        "execution_mode": "manual_ordered_commit",
        "failure_policy": "stop_on_first_write_failure_then_release_lock",
    }
    return _digest(seed)


def _empty_report(
    *,
    executor_preview: Mapping[str, Any],
) -> MemoryEvidenceRepairPostCommitAuditPreviewReport:
    return MemoryEvidenceRepairPostCommitAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=POST_COMMIT_AUDIT_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_executor_preview_report=dict(executor_preview),
    )


def _blocked_report(
    *,
    executor_preview: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairPostCommitAuditCheck, ...],
) -> MemoryEvidenceRepairPostCommitAuditPreviewReport:
    return MemoryEvidenceRepairPostCommitAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=POST_COMMIT_AUDIT_STATUS_BLOCKED,
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
        source_executor_preview_report=dict(executor_preview),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairPostCommitAuditCheck:
    return MemoryEvidenceRepairPostCommitAuditCheck(
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
) -> MemoryEvidenceRepairPostCommitAuditCheck:
    return MemoryEvidenceRepairPostCommitAuditCheck(
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
