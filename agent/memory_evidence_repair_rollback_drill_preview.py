"""Read-only rollback drill preview for memory evidence repair audits.

The rollback drill preview turns a post-commit audit preview or observed
post-commit audit failure into a rehearsal plan. It never rolls back durable
memory and never mutates receipts, approval tokens, or write locks.
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
from agent.memory_evidence_repair_post_commit_audit_preview import (
    AUDIT_STEP_STATUS_FAIL,
    POST_COMMIT_AUDIT_STATUS_BLOCKED,
    POST_COMMIT_AUDIT_STATUS_NO_ACTION_NEEDED,
    POST_COMMIT_AUDIT_STATUS_READY,
)


ROLLBACK_DRILL_STATUS_READY = "rollback_drill_preview_ready"
ROLLBACK_DRILL_STATUS_BLOCKED = "blocked"
ROLLBACK_DRILL_STATUS_NO_ACTION_NEEDED = "no_action_needed"

ROLLBACK_DRILL_TRIGGER_AUDIT_FAILURE = "post_commit_audit_failure"
ROLLBACK_DRILL_TRIGGER_PREPAREDNESS = "post_commit_audit_preparedness"

DRILL_STEP_STATUS_PLANNED = "planned"

CHECK_POST_COMMIT_AUDIT_AVAILABLE = "post_commit_audit_available"
CHECK_AUDIT_FAILURE_CONTEXT = "audit_failure_context"
CHECK_ROLLBACK_DRILL_SOURCES = "rollback_drill_sources"


@dataclass(frozen=True)
class MemoryEvidenceRepairRollbackDrillCheck:
    """One read-only rollback drill readiness check."""

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
class MemoryEvidenceRepairRollbackDrillStep:
    """One future rollback drill step preview."""

    id: str
    sequence: int
    phase: str
    action: str
    description: str
    status: str
    operation_id: str
    audit_step_id: str
    audit_category: str
    receipt_id: str
    token_id: str
    lock_id: str
    patch_digest: str
    rollback_source: str
    expected_result: str
    required_operator_action: str
    source_audit_step: dict[str, Any]
    future_would_restore_memory: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "phase": self.phase,
            "action": self.action,
            "description": self.description,
            "status": self.status,
            "operation_id": self.operation_id,
            "audit_step_id": self.audit_step_id,
            "audit_category": self.audit_category,
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "patch_digest": self.patch_digest,
            "rollback_source": self.rollback_source,
            "expected_result": self.expected_result,
            "required_operator_action": self.required_operator_action,
            "source_audit_step": dict(self.source_audit_step),
            "future_would_restore_memory": self.future_would_restore_memory,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRollbackDrillPreview:
    """One read-only rollback drill preview."""

    id: str
    status: str
    trigger: str
    rollback_mode: str
    post_commit_audit_id: str
    executor_preview_id: str
    receipt_id: str
    token_id: str
    lock_id: str
    operation_ids: tuple[str, ...]
    patch_digests: tuple[str, ...]
    failed_audit_step_ids: tuple[str, ...]
    drill_step_ids: tuple[str, ...]
    drill_digest: str
    drill_preview: str
    steps: tuple[MemoryEvidenceRepairRollbackDrillStep, ...]
    would_preserve_failed_audit_evidence: bool
    would_isolate_failed_memory: bool
    would_restore_snapshots: bool
    would_reconcile_receipt_token_lock: bool
    would_rerun_post_commit_audit: bool
    safety_note: str
    source_post_commit_audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "trigger": self.trigger,
            "rollback_mode": self.rollback_mode,
            "post_commit_audit_id": self.post_commit_audit_id,
            "executor_preview_id": self.executor_preview_id,
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "operation_ids": list(self.operation_ids),
            "patch_digests": list(self.patch_digests),
            "failed_audit_step_ids": list(self.failed_audit_step_ids),
            "drill_step_ids": list(self.drill_step_ids),
            "drill_digest": self.drill_digest,
            "drill_preview": self.drill_preview,
            "steps": [step.to_dict() for step in self.steps],
            "would_preserve_failed_audit_evidence": self.would_preserve_failed_audit_evidence,
            "would_isolate_failed_memory": self.would_isolate_failed_memory,
            "would_restore_snapshots": self.would_restore_snapshots,
            "would_reconcile_receipt_token_lock": self.would_reconcile_receipt_token_lock,
            "would_rerun_post_commit_audit": self.would_rerun_post_commit_audit,
            "safety_note": self.safety_note,
            "source_post_commit_audit": dict(self.source_post_commit_audit),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRollbackDrillPreviewReport:
    """Read-only rollback drill preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRollbackDrillCheck, ...]
    previews: tuple[MemoryEvidenceRepairRollbackDrillPreview, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_post_commit_audit_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        by_trigger: dict[str, int] = {}
        future_restore_count = 0
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        for preview in self.previews:
            by_trigger[preview.trigger] = by_trigger.get(preview.trigger, 0) + 1
            future_restore_count += sum(
                1 for step in preview.steps if step.future_would_restore_memory
            )
        drill_step_count = sum(len(preview.steps) for preview in self.previews)
        failed_step_count = sum(
            len(preview.failed_audit_step_ids) for preview in self.previews
        )
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "preview_count": len(self.previews),
            "drill_step_count": drill_step_count,
            "failed_audit_step_count": failed_step_count,
            "future_restore_step_count": future_restore_count,
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "rollback_drill_preview_available": bool(self.previews),
            "rollback_drill_ready": self.status == ROLLBACK_DRILL_STATUS_READY,
            "has_blocks": self.status == ROLLBACK_DRILL_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
            "by_trigger": by_trigger,
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
            "source_post_commit_audit_report": dict(self.source_post_commit_audit_report),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_rollback_drill_preview(
    *,
    post_commit_audit: Mapping[str, Any] | None = None,
    rollback_plan: Mapping[str, Any] | None = None,
    snapshot_plan: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRollbackDrillPreviewReport:
    """Build a read-only rollback drill from a post-commit audit report."""

    post_commit_audit = post_commit_audit if isinstance(post_commit_audit, Mapping) else {}
    rollback_plan = rollback_plan if isinstance(rollback_plan, Mapping) else {}
    snapshot_plan = snapshot_plan if isinstance(snapshot_plan, Mapping) else {}
    audit_status = _str(post_commit_audit.get("status"))
    audit_preview = _extract_audit_preview(post_commit_audit)
    failed_steps = tuple(_failed_audit_steps(post_commit_audit, audit_preview))
    executor_preview = _extract_executor_preview(post_commit_audit, audit_preview)

    if not post_commit_audit or audit_status == POST_COMMIT_AUDIT_STATUS_NO_ACTION_NEEDED:
        return _empty_report(post_commit_audit=post_commit_audit)

    if not audit_preview and not failed_steps:
        return _blocked_report(
            post_commit_audit=post_commit_audit,
            checks=(
                _fail(
                    CHECK_POST_COMMIT_AUDIT_AVAILABLE,
                    "No usable post-commit audit preview or observed audit failure was supplied.",
                    tuple(_list_of_str(post_commit_audit.get("required_actions")))
                    or ("produce_post_commit_audit_preview",),
                    {"post_commit_audit_status": audit_status},
                ),
            ),
        )

    if (
        audit_status
        and audit_status not in {POST_COMMIT_AUDIT_STATUS_READY, POST_COMMIT_AUDIT_STATUS_BLOCKED}
        and not audit_preview
    ):
        return _blocked_report(
            post_commit_audit=post_commit_audit,
            checks=(
                _fail(
                    CHECK_POST_COMMIT_AUDIT_AVAILABLE,
                    "Post-commit audit status is not ready for rollback drill planning.",
                    ("produce_ready_or_failed_post_commit_audit",),
                    {"post_commit_audit_status": audit_status},
                ),
            ),
        )

    checks = (
        _post_commit_audit_available_check(
            post_commit_audit=post_commit_audit,
            audit_preview=audit_preview,
            failed_steps=failed_steps,
        ),
        _audit_failure_context_check(failed_steps=failed_steps, audit_preview=audit_preview),
        _rollback_drill_sources_check(
            rollback_plan=rollback_plan,
            snapshot_plan=snapshot_plan,
            audit_preview=audit_preview,
            executor_preview=executor_preview,
        ),
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
        return MemoryEvidenceRepairRollbackDrillPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=ROLLBACK_DRILL_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_post_commit_audit_report=dict(post_commit_audit),
        )

    preview = _preview_from_audit(
        post_commit_audit=post_commit_audit,
        audit_preview=audit_preview,
        failed_steps=failed_steps,
        executor_preview=executor_preview,
        rollback_plan=rollback_plan,
        snapshot_plan=snapshot_plan,
    )
    return MemoryEvidenceRepairRollbackDrillPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=ROLLBACK_DRILL_STATUS_READY,
        checks=checks,
        previews=(preview,),
        blocking_reasons=(),
        required_actions=(),
        source_post_commit_audit_report=dict(post_commit_audit),
    )


def empty_evidence_repair_rollback_drill_preview() -> MemoryEvidenceRepairRollbackDrillPreviewReport:
    """Return an empty read-only rollback drill preview."""

    return _empty_report(post_commit_audit={})


def _post_commit_audit_available_check(
    *,
    post_commit_audit: Mapping[str, Any],
    audit_preview: Mapping[str, Any],
    failed_steps: Sequence[Mapping[str, Any]],
) -> MemoryEvidenceRepairRollbackDrillCheck:
    audit_status = _str(post_commit_audit.get("status"))
    if audit_preview or failed_steps:
        return _pass(
            CHECK_POST_COMMIT_AUDIT_AVAILABLE,
            "Post-commit audit context is available for rollback drill planning.",
            {
                "post_commit_audit_status": audit_status,
                "audit_preview_id": _str(audit_preview.get("id")),
                "failed_audit_step_count": len(failed_steps),
            },
        )
    return _fail(
        CHECK_POST_COMMIT_AUDIT_AVAILABLE,
        "No post-commit audit preview or failure context is available.",
        ("produce_post_commit_audit_preview",),
        {"post_commit_audit_status": audit_status},
    )


def _audit_failure_context_check(
    *,
    failed_steps: Sequence[Mapping[str, Any]],
    audit_preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRollbackDrillCheck:
    if failed_steps:
        return _pass(
            CHECK_AUDIT_FAILURE_CONTEXT,
            "Observed audit failures can drive a rollback drill.",
            {"failed_audit_step_ids": [_str(step.get("id")) for step in failed_steps]},
        )
    return _pass(
        CHECK_AUDIT_FAILURE_CONTEXT,
        "No observed audit failure supplied; rollback drill remains a preparedness rehearsal.",
        {
            "audit_preview_id": _str(audit_preview.get("id")),
            "trigger": ROLLBACK_DRILL_TRIGGER_PREPAREDNESS,
        },
    )


def _rollback_drill_sources_check(
    *,
    rollback_plan: Mapping[str, Any],
    snapshot_plan: Mapping[str, Any],
    audit_preview: Mapping[str, Any],
    executor_preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRollbackDrillCheck:
    sources = _rollback_sources(
        rollback_plan=rollback_plan,
        snapshot_plan=snapshot_plan,
        audit_preview=audit_preview,
        executor_preview=executor_preview,
    )
    return _pass(
        CHECK_ROLLBACK_DRILL_SOURCES,
        "Rollback drill source references are captured for future human verification.",
        {"sources": sources},
    )


def _preview_from_audit(
    *,
    post_commit_audit: Mapping[str, Any],
    audit_preview: Mapping[str, Any],
    failed_steps: Sequence[Mapping[str, Any]],
    executor_preview: Mapping[str, Any],
    rollback_plan: Mapping[str, Any],
    snapshot_plan: Mapping[str, Any],
) -> MemoryEvidenceRepairRollbackDrillPreview:
    trigger = (
        ROLLBACK_DRILL_TRIGGER_AUDIT_FAILURE
        if failed_steps
        else ROLLBACK_DRILL_TRIGGER_PREPAREDNESS
    )
    operation_ids = tuple(
        _operation_ids(
            audit_preview=audit_preview,
            executor_preview=executor_preview,
            failed_steps=failed_steps,
        )
    )
    patch_digests = tuple(
        _patch_digests(
            audit_preview=audit_preview,
            executor_preview=executor_preview,
            operation_ids=operation_ids,
        )
    )
    receipt_id = _first_nonempty(
        _str(audit_preview.get("receipt_id")),
        _str(executor_preview.get("receipt_id")),
        _first_from_steps(failed_steps, "receipt_id"),
    )
    token_id = _first_nonempty(
        _str(audit_preview.get("token_id")),
        _str(executor_preview.get("token_id")),
        _first_from_steps(failed_steps, "token_id"),
    )
    lock_id = _first_nonempty(
        _str(audit_preview.get("lock_id")),
        _str(executor_preview.get("lock_id")),
        _first_from_steps(failed_steps, "lock_id"),
    )
    steps = tuple(
        _drill_steps(
            trigger=trigger,
            operation_ids=operation_ids,
            patch_digests=patch_digests,
            failed_steps=failed_steps,
            receipt_id=receipt_id,
            token_id=token_id,
            lock_id=lock_id,
            rollback_plan=rollback_plan,
            snapshot_plan=snapshot_plan,
            executor_preview=executor_preview,
        )
    )
    drill_step_ids = tuple(step.id for step in steps)
    failed_step_ids = tuple(
        _dedupe_strings(_str(step.get("id")) for step in failed_steps)
    )
    drill_seed = {
        "trigger": trigger,
        "post_commit_audit_id": _str(audit_preview.get("id")),
        "executor_preview_id": _str(executor_preview.get("id")),
        "receipt_id": receipt_id,
        "token_id": token_id,
        "lock_id": lock_id,
        "operation_ids": operation_ids,
        "patch_digests": patch_digests,
        "failed_audit_step_ids": failed_step_ids,
        "drill_step_ids": drill_step_ids,
    }
    drill_digest = _digest(drill_seed)
    drill_id = f"rollback-drill-{drill_digest[:16]}"
    return MemoryEvidenceRepairRollbackDrillPreview(
        id=drill_id,
        status=ROLLBACK_DRILL_STATUS_READY,
        trigger=trigger,
        rollback_mode="failure_response" if failed_steps else "preparedness_rehearsal",
        post_commit_audit_id=_str(audit_preview.get("id")),
        executor_preview_id=_str(executor_preview.get("id")),
        receipt_id=receipt_id,
        token_id=token_id,
        lock_id=lock_id,
        operation_ids=operation_ids,
        patch_digests=patch_digests,
        failed_audit_step_ids=failed_step_ids,
        drill_step_ids=drill_step_ids,
        drill_digest=drill_digest,
        drill_preview=f"{drill_id}:{drill_digest[:12]}",
        steps=steps,
        would_preserve_failed_audit_evidence=True,
        would_isolate_failed_memory=True,
        would_restore_snapshots=True,
        would_reconcile_receipt_token_lock=True,
        would_rerun_post_commit_audit=True,
        safety_note=(
            "Read-only rollback drill preview. It rehearses future isolation, "
            "snapshot restoration, reconciliation, and re-audit steps but does "
            "not mutate durable memory."
        ),
        source_post_commit_audit=dict(post_commit_audit),
    )


def _drill_steps(
    *,
    trigger: str,
    operation_ids: Sequence[str],
    patch_digests: Sequence[str],
    failed_steps: Sequence[Mapping[str, Any]],
    receipt_id: str,
    token_id: str,
    lock_id: str,
    rollback_plan: Mapping[str, Any],
    snapshot_plan: Mapping[str, Any],
    executor_preview: Mapping[str, Any],
) -> list[MemoryEvidenceRepairRollbackDrillStep]:
    steps: list[MemoryEvidenceRepairRollbackDrillStep] = []
    failure_mode = trigger == ROLLBACK_DRILL_TRIGGER_AUDIT_FAILURE
    source_step = failed_steps[0] if failed_steps else {}
    steps.append(
        _step(
            sequence=1,
            phase="preserve",
            action=(
                "preserve_failed_audit_context"
                if failure_mode
                else "prepare_rollback_context"
            ),
            description=(
                "Preserve the failed post-commit audit report, executor preview, and observed signals."
                if failure_mode
                else "Prepare the audit, executor, receipt, token, and lock context for a future rollback rehearsal."
            ),
            operation_id="",
            source_step=source_step,
            receipt_id=receipt_id,
            token_id=token_id,
            lock_id=lock_id,
            expected_result="rollback_drill_context_preserved",
            required_operator_action="archive_audit_report_executor_preview_and_observed_signals",
        )
    )
    steps.append(
        _step(
            sequence=2,
            phase="isolation",
            action=(
                "isolate_impacted_memory_records"
                if failure_mode
                else "define_impacted_operation_isolation"
            ),
            description=(
                "Identify and isolate memory records touched by failed audit steps before any manual rollback."
                if failure_mode
                else "Define which memory records would be isolated if the post-commit audit later fails."
            ),
            operation_id=",".join(operation_ids),
            source_step=source_step,
            receipt_id=receipt_id,
            token_id=token_id,
            lock_id=lock_id,
            expected_result="impacted_operations_isolated_for_review",
            required_operator_action="mark_impacted_operations_read_only_until_review_completes",
        )
    )
    next_sequence = 3
    restore_targets = list(operation_ids) or [""]
    for index, operation_id in enumerate(restore_targets):
        patch_digest = patch_digests[index] if index < len(patch_digests) else ""
        steps.append(
            _step(
                sequence=next_sequence,
                phase="restore",
                action=(
                    "restore_from_pre_commit_snapshot_or_rollback_plan"
                    if failure_mode
                    else "verify_pre_commit_snapshot_or_rollback_plan"
                ),
                description=(
                    "Restore the impacted operation from its pre-commit snapshot or approved rollback plan."
                    if failure_mode
                    else "Verify a pre-commit snapshot or rollback plan exists for the operation before any future write."
                ),
                operation_id=operation_id,
                source_step=_failed_step_for_operation(failed_steps, operation_id),
                receipt_id=receipt_id,
                token_id=token_id,
                lock_id=lock_id,
                patch_digest=patch_digest,
                rollback_source=_rollback_source_for_operation(
                    operation_id=operation_id,
                    patch_digest=patch_digest,
                    rollback_plan=rollback_plan,
                    snapshot_plan=snapshot_plan,
                    executor_preview=executor_preview,
                ),
                expected_result="pre_commit_state_restored_or_verified",
                required_operator_action="verify_snapshot_digest_then_restore_manually",
                future_would_restore_memory=failure_mode,
            )
        )
        next_sequence += 1
    steps.append(
        _step(
            sequence=next_sequence,
            phase="reconcile",
            action="reconcile_commit_receipt_token_and_lock",
            description="Reconcile receipt, approval-token, and write-lock state after rollback handling.",
            operation_id="",
            source_step=source_step,
            receipt_id=receipt_id,
            token_id=token_id,
            lock_id=lock_id,
            expected_result="receipt_token_lock_state_consistent",
            required_operator_action="verify_receipt_token_and_lock_state_before_unlocking_future_writes",
        )
    )
    next_sequence += 1
    steps.append(
        _step(
            sequence=next_sequence,
            phase="audit",
            action="rerun_post_commit_audit",
            description="Rerun the post-commit audit after rollback or preparedness checks complete.",
            operation_id="",
            source_step=source_step,
            receipt_id=receipt_id,
            token_id=token_id,
            lock_id=lock_id,
            expected_result="post_commit_audit_passes_after_rollback_drill",
            required_operator_action="rerun_memory_evidence_repair_post_commit_audit_preview",
        )
    )
    return steps


def _step(
    *,
    sequence: int,
    phase: str,
    action: str,
    description: str,
    operation_id: str,
    source_step: Mapping[str, Any],
    receipt_id: str,
    token_id: str,
    lock_id: str,
    expected_result: str,
    required_operator_action: str,
    patch_digest: str = "",
    rollback_source: str = "",
    future_would_restore_memory: bool = False,
) -> MemoryEvidenceRepairRollbackDrillStep:
    audit_step_id = _str(source_step.get("id"))
    audit_category = _str(source_step.get("category"))
    if not patch_digest:
        patch_digest = _patch_digest_from_audit_step(source_step)
    if not rollback_source:
        rollback_source = "pre_commit_snapshot_or_rollback_plan_required"
    return MemoryEvidenceRepairRollbackDrillStep(
        id=f"rollback-drill-step-{sequence}-{action}",
        sequence=sequence,
        phase=phase,
        action=action,
        description=description,
        status=DRILL_STEP_STATUS_PLANNED,
        operation_id=operation_id,
        audit_step_id=audit_step_id,
        audit_category=audit_category,
        receipt_id=receipt_id or _str(source_step.get("receipt_id")),
        token_id=token_id or _str(source_step.get("token_id")),
        lock_id=lock_id or _str(source_step.get("lock_id")),
        patch_digest=patch_digest,
        rollback_source=rollback_source,
        expected_result=expected_result,
        required_operator_action=required_operator_action,
        source_audit_step=dict(source_step),
        future_would_restore_memory=future_would_restore_memory,
    )


def _extract_audit_preview(post_commit_audit: Mapping[str, Any]) -> dict[str, Any]:
    previews = post_commit_audit.get("previews")
    if isinstance(previews, list):
        for preview in previews:
            if isinstance(preview, Mapping):
                return dict(preview)
    if _str(post_commit_audit.get("id")).startswith("post-commit-audit-"):
        return dict(post_commit_audit)
    return {}


def _failed_audit_steps(
    post_commit_audit: Mapping[str, Any],
    audit_preview: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    steps: list[Mapping[str, Any]] = []
    for step in _list(audit_preview.get("audit_steps")):
        if isinstance(step, Mapping) and _str(step.get("status")) == AUDIT_STEP_STATUS_FAIL:
            steps.append(dict(step))
    for check in _list(post_commit_audit.get("checks")):
        if not isinstance(check, Mapping):
            continue
        details = _dict(check.get("details"))
        for failure in _list(details.get("failures")):
            if isinstance(failure, Mapping):
                steps.append(dict(failure))
    return _dedupe_steps(steps)


def _extract_executor_preview(
    post_commit_audit: Mapping[str, Any],
    audit_preview: Mapping[str, Any],
) -> dict[str, Any]:
    source = audit_preview.get("source_executor_preview")
    if isinstance(source, Mapping):
        return dict(source)
    source_report = post_commit_audit.get("source_executor_preview_report")
    if isinstance(source_report, Mapping):
        previews = source_report.get("previews")
        if isinstance(previews, list):
            for preview in previews:
                if isinstance(preview, Mapping):
                    return dict(preview)
        if _str(source_report.get("id")).startswith("executor-preview-"):
            return dict(source_report)
    return {}


def _operation_ids(
    *,
    audit_preview: Mapping[str, Any],
    executor_preview: Mapping[str, Any],
    failed_steps: Sequence[Mapping[str, Any]],
) -> list[str]:
    ids: list[str] = []
    ids.extend(_list_of_str(audit_preview.get("operation_ids")))
    ids.extend(_list_of_str(executor_preview.get("operation_ids")))
    ids.extend(_str(step.get("operation_id")) for step in failed_steps)
    if not ids and failed_steps:
        ids.extend(_list_of_str(executor_preview.get("operation_ids")))
    return _dedupe_strings(ids)


def _patch_digests(
    *,
    audit_preview: Mapping[str, Any],
    executor_preview: Mapping[str, Any],
    operation_ids: Sequence[str],
) -> list[str]:
    digests = _list_of_str(audit_preview.get("patch_digests"))
    if digests:
        return digests
    digests = _list_of_str(executor_preview.get("patch_digests"))
    if digests:
        return digests
    steps = [
        step
        for step in _list(executor_preview.get("steps"))
        if isinstance(step, Mapping) and step.get("future_would_write_memory") is True
    ]
    by_operation = {_str(step.get("operation_id")): _str(step.get("patch_digest")) for step in steps}
    return [by_operation.get(operation_id, "") for operation_id in operation_ids]


def _rollback_sources(
    *,
    rollback_plan: Mapping[str, Any],
    snapshot_plan: Mapping[str, Any],
    audit_preview: Mapping[str, Any],
    executor_preview: Mapping[str, Any],
) -> list[str]:
    sources: list[str] = []
    if rollback_plan:
        sources.append("rollback_plan")
    if snapshot_plan:
        sources.append("snapshot_plan")
    if audit_preview:
        sources.append("post_commit_audit_preview")
    if executor_preview:
        sources.append("executor_preview")
    if not sources:
        sources.append("operator_supplied_pre_commit_snapshot_required")
    return _dedupe_strings(sources)


def _rollback_source_for_operation(
    *,
    operation_id: str,
    patch_digest: str,
    rollback_plan: Mapping[str, Any],
    snapshot_plan: Mapping[str, Any],
    executor_preview: Mapping[str, Any],
) -> str:
    if _matches_plan(rollback_plan, operation_id=operation_id, patch_digest=patch_digest):
        return "provided_rollback_plan"
    if _matches_plan(snapshot_plan, operation_id=operation_id, patch_digest=patch_digest):
        return "provided_snapshot_plan"
    if executor_preview:
        return "executor_preview_rollback_hint"
    return "pre_commit_snapshot_or_rollback_plan_required"


def _matches_plan(plan: Mapping[str, Any], *, operation_id: str, patch_digest: str) -> bool:
    if not plan:
        return False
    rows = []
    for key in ("steps", "plans", "snapshots", "previews"):
        rows.extend(item for item in _list(plan.get(key)) if isinstance(item, Mapping))
    if not rows and plan:
        rows.append(plan)
    for row in rows:
        if operation_id and operation_id in {
            _str(row.get("operation_id")),
            _str(row.get("id")),
            _str(row.get("source_operation_id")),
            _str(row.get("ledger_entry_id")),
        }:
            return True
        if patch_digest and patch_digest == _str(row.get("patch_digest")):
            return True
    return False


def _failed_step_for_operation(
    failed_steps: Sequence[Mapping[str, Any]],
    operation_id: str,
) -> Mapping[str, Any]:
    for step in failed_steps:
        if _str(step.get("operation_id")) == operation_id:
            return step
    return failed_steps[0] if failed_steps else {}


def _patch_digest_from_audit_step(step: Mapping[str, Any]) -> str:
    expected = _dict(step.get("expected_signal"))
    return _str(expected.get("patch_digest"))


def _first_from_steps(steps: Sequence[Mapping[str, Any]], field_name: str) -> str:
    for step in steps:
        value = _str(step.get(field_name))
        if value:
            return value
    return ""


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _empty_report(
    *,
    post_commit_audit: Mapping[str, Any],
) -> MemoryEvidenceRepairRollbackDrillPreviewReport:
    return MemoryEvidenceRepairRollbackDrillPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=ROLLBACK_DRILL_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_post_commit_audit_report=dict(post_commit_audit),
    )


def _blocked_report(
    *,
    post_commit_audit: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRollbackDrillCheck, ...],
) -> MemoryEvidenceRepairRollbackDrillPreviewReport:
    return MemoryEvidenceRepairRollbackDrillPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=ROLLBACK_DRILL_STATUS_BLOCKED,
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
        source_post_commit_audit_report=dict(post_commit_audit),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRollbackDrillCheck:
    return MemoryEvidenceRepairRollbackDrillCheck(
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
) -> MemoryEvidenceRepairRollbackDrillCheck:
    return MemoryEvidenceRepairRollbackDrillCheck(
        id=check_id,
        status=CHECK_STATUS_FAIL,
        reason=reason,
        required_actions=tuple(required_actions),
        details=dict(details),
    )


def _dedupe_steps(steps: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    result: list[Mapping[str, Any]] = []
    for step in steps:
        key = _str(step.get("id")) or _digest(step)
        if key in seen:
            continue
        seen.add(key)
        result.append(step)
    return result


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
