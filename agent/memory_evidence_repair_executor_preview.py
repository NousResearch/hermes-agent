"""Read-only manual commit executor preview for memory evidence repair.

The executor preview turns a verified write-lock gate into an ordered execution
plan for a later manual memory repair commit. It never writes to durable memory
stores and never acquires or releases a real lock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_commit_receipt import RECEIPT_SCOPE
from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
)
from agent.memory_evidence_repair_write_lock_gate import (
    LOCK_DRAFT_STATUS_READY,
    WRITE_LOCK_STATUS_BLOCKED,
    WRITE_LOCK_STATUS_NO_ACTION_NEEDED,
    WRITE_LOCK_STATUS_READY,
    WRITE_LOCK_TYPE,
)


EXECUTOR_PREVIEW_STATUS_READY = "executor_preview_ready_for_manual_commit"
EXECUTOR_PREVIEW_STATUS_BLOCKED = "blocked"
EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED = "no_action_needed"

CHECK_WRITE_LOCK_READY = "write_lock_ready"
CHECK_WRITE_LOCK_INTEGRITY = "write_lock_integrity"
CHECK_WRITE_LOCK_EXPIRY = "write_lock_expiry"
CHECK_EXECUTION_OPERATIONS = "execution_operations"


@dataclass(frozen=True)
class MemoryEvidenceRepairExecutorPreviewCheck:
    """One read-only executor preview readiness check."""

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
class MemoryEvidenceRepairExecutorStep:
    """One future executor step preview."""

    id: str
    sequence: int
    phase: str
    action: str
    description: str
    operation_id: str
    candidate_id: str
    ledger_entry_id: str
    patch_digest: str
    target: dict[str, Any]
    evidence_fields: tuple[str, ...]
    source_operation: dict[str, Any]
    future_would_write_memory: bool
    stop_on_failure: bool
    rollback_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "phase": self.phase,
            "action": self.action,
            "description": self.description,
            "operation_id": self.operation_id,
            "candidate_id": self.candidate_id,
            "ledger_entry_id": self.ledger_entry_id,
            "patch_digest": self.patch_digest,
            "target": dict(self.target),
            "evidence_fields": list(self.evidence_fields),
            "source_operation": dict(self.source_operation),
            "future_would_write_memory": self.future_would_write_memory,
            "stop_on_failure": self.stop_on_failure,
            "rollback_hint": self.rollback_hint,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairExecutorPreview:
    """One read-only executor preview."""

    id: str
    status: str
    lock_id: str
    receipt_id: str
    token_id: str
    operation_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    patch_digests: tuple[str, ...]
    step_ids: tuple[str, ...]
    preview_digest: str
    preview_preview: str
    execution_mode: str
    failure_policy: str
    steps: tuple[MemoryEvidenceRepairExecutorStep, ...]
    would_apply_memory_patches: bool
    would_record_receipt: bool
    would_mark_token_used: bool
    would_release_write_lock: bool
    safety_note: str
    source_lock: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "lock_id": self.lock_id,
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "operation_ids": list(self.operation_ids),
            "candidate_ids": list(self.candidate_ids),
            "patch_digests": list(self.patch_digests),
            "step_ids": list(self.step_ids),
            "preview_digest": self.preview_digest,
            "preview_preview": self.preview_preview,
            "execution_mode": self.execution_mode,
            "failure_policy": self.failure_policy,
            "steps": [step.to_dict() for step in self.steps],
            "would_apply_memory_patches": self.would_apply_memory_patches,
            "would_record_receipt": self.would_record_receipt,
            "would_mark_token_used": self.would_mark_token_used,
            "would_release_write_lock": self.would_release_write_lock,
            "safety_note": self.safety_note,
            "source_lock": dict(self.source_lock),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairExecutorPreviewReport:
    """Read-only manual commit executor preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairExecutorPreviewCheck, ...]
    previews: tuple[MemoryEvidenceRepairExecutorPreview, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_write_lock_gate: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        step_count = sum(len(preview.steps) for preview in self.previews)
        future_write_step_count = sum(
            1
            for preview in self.previews
            for step in preview.steps
            if step.future_would_write_memory
        )
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "preview_count": len(self.previews),
            "step_count": step_count,
            "future_write_step_count": future_write_step_count,
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "executor_preview_available": bool(self.previews),
            "manual_executor_preview_ready": self.status == EXECUTOR_PREVIEW_STATUS_READY,
            "has_blocks": self.status == EXECUTOR_PREVIEW_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
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
            "source_write_lock_gate": dict(self.source_write_lock_gate),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_executor_preview(
    *,
    write_lock_gate: Mapping[str, Any] | None = None,
    current_time: str | datetime | None = None,
) -> MemoryEvidenceRepairExecutorPreviewReport:
    """Build a read-only executor preview from a write-lock gate."""

    write_lock_gate = write_lock_gate if isinstance(write_lock_gate, Mapping) else {}
    gate_status = _str(write_lock_gate.get("status"))
    lock = _extract_lock(write_lock_gate)
    now = _parse_datetime(current_time) or datetime.now(timezone.utc)

    if not lock:
        if not write_lock_gate or gate_status == WRITE_LOCK_STATUS_NO_ACTION_NEEDED:
            return _empty_report(write_lock_gate=write_lock_gate)
        source_blocking = tuple(_list_of_str(write_lock_gate.get("blocking_reasons")))
        source_actions = tuple(_list_of_str(write_lock_gate.get("required_actions")))
        return _blocked_report(
            write_lock_gate=write_lock_gate,
            checks=(
                _fail(
                    CHECK_WRITE_LOCK_READY,
                    "No write-lock draft was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions or ("produce_write_lock_gate",),
                    {"write_lock_gate_status": gate_status},
                ),
            ),
        )

    operations = tuple(_operations_from_lock(lock))
    checks = (
        _write_lock_ready_check(write_lock_gate, lock),
        _write_lock_integrity_check(lock),
        _write_lock_expiry_check(lock, now),
        _execution_operations_check(lock, operations),
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
        return MemoryEvidenceRepairExecutorPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=EXECUTOR_PREVIEW_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_write_lock_gate=dict(write_lock_gate),
        )

    preview = _preview_from_lock(lock=lock, operations=operations)
    return MemoryEvidenceRepairExecutorPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=EXECUTOR_PREVIEW_STATUS_READY,
        checks=checks,
        previews=(preview,),
        blocking_reasons=(),
        required_actions=(),
        source_write_lock_gate=dict(write_lock_gate),
    )


def empty_evidence_repair_executor_preview() -> MemoryEvidenceRepairExecutorPreviewReport:
    """Return an empty read-only manual commit executor preview."""

    return _empty_report(write_lock_gate={})


def _extract_lock(write_lock_gate: Mapping[str, Any]) -> dict[str, Any]:
    locks = write_lock_gate.get("locks")
    if isinstance(locks, list):
        for lock in locks:
            if isinstance(lock, Mapping):
                return dict(lock)
    if _str(write_lock_gate.get("id")).startswith("write-lock-"):
        return dict(write_lock_gate)
    return {}


def _write_lock_ready_check(
    write_lock_gate: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> MemoryEvidenceRepairExecutorPreviewCheck:
    gate_status = _str(write_lock_gate.get("status"))
    lock_status = _str(lock.get("status"))
    if gate_status and gate_status != WRITE_LOCK_STATUS_READY:
        return _fail(
            CHECK_WRITE_LOCK_READY,
            "Write-lock gate is not ready for manual commit execution.",
            tuple(_list_of_str(write_lock_gate.get("required_actions")))
            or ("produce_ready_write_lock_gate",),
            {"write_lock_gate_status": gate_status, "lock_status": lock_status},
        )
    if lock_status != LOCK_DRAFT_STATUS_READY:
        return _fail(
            CHECK_WRITE_LOCK_READY,
            "Write-lock draft is not ready.",
            ("produce_ready_write_lock_gate",),
            {"write_lock_gate_status": gate_status, "lock_status": lock_status},
        )
    return _pass(
        CHECK_WRITE_LOCK_READY,
        "Write-lock gate and draft are ready.",
        {"lock_id": _str(lock.get("id"))},
    )


def _write_lock_integrity_check(
    lock: Mapping[str, Any],
) -> MemoryEvidenceRepairExecutorPreviewCheck:
    expected_digest = _expected_lock_digest(lock)
    actual_digest = _str(lock.get("lock_digest"))
    expected_id = f"write-lock-{expected_digest[:16]}" if expected_digest else ""
    missing_fields: list[str] = []
    for field_name in ("receipt_id", "token_id", "operation_ids", "patch_digests"):
        value = lock.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(lock.get("id")) != expected_id
        or _str(lock.get("lock_type")) != WRITE_LOCK_TYPE
        or _str(lock.get("scope")) != RECEIPT_SCOPE
    ):
        return _fail(
            CHECK_WRITE_LOCK_INTEGRITY,
            "Write-lock digest, id, type, scope, or required fields are invalid.",
            ("regenerate_write_lock_gate",),
            {
                "expected_lock_digest": expected_digest,
                "actual_lock_digest": actual_digest,
                "expected_lock_id": expected_id,
                "actual_lock_id": _str(lock.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_WRITE_LOCK_INTEGRITY,
        "Write-lock integrity checks pass.",
        {"lock_id": expected_id, "lock_digest": actual_digest},
    )


def _write_lock_expiry_check(
    lock: Mapping[str, Any],
    now: datetime,
) -> MemoryEvidenceRepairExecutorPreviewCheck:
    expires_at = _parse_datetime(lock.get("expires_at"))
    if expires_at is None:
        return _fail(
            CHECK_WRITE_LOCK_EXPIRY,
            "Write-lock expiry is missing or invalid.",
            ("regenerate_write_lock_gate",),
            {"expires_at": _str(lock.get("expires_at"))},
        )
    if expires_at <= now:
        return _fail(
            CHECK_WRITE_LOCK_EXPIRY,
            "Write-lock draft has expired.",
            ("regenerate_write_lock_gate",),
            {"expires_at": expires_at.isoformat(), "current_time": now.isoformat()},
        )
    return _pass(
        CHECK_WRITE_LOCK_EXPIRY,
        "Write-lock draft is within its expiry window.",
        {"expires_at": expires_at.isoformat(), "current_time": now.isoformat()},
    )


def _execution_operations_check(
    lock: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
) -> MemoryEvidenceRepairExecutorPreviewCheck:
    lock_operation_ids = _list_of_str(lock.get("operation_ids"))
    operation_ids = [_str(operation.get("id")) for operation in operations]
    lock_patch_digests = _list_of_str(lock.get("patch_digests"))
    operation_patch_digests = [_str(operation.get("patch_digest")) for operation in operations]
    if not operations:
        return _fail(
            CHECK_EXECUTION_OPERATIONS,
            "No executable dry-run operations were found in the write-lock source receipt.",
            ("regenerate_commit_receipt_with_source_operations",),
            {"lock_operation_ids": lock_operation_ids},
        )
    if lock_operation_ids != operation_ids or lock_patch_digests != operation_patch_digests:
        return _fail(
            CHECK_EXECUTION_OPERATIONS,
            "Write-lock operation ids or patch digests do not match source operations.",
            ("regenerate_write_lock_gate",),
            {
                "lock_operation_ids": lock_operation_ids,
                "operation_ids": operation_ids,
                "lock_patch_digests": lock_patch_digests,
                "operation_patch_digests": operation_patch_digests,
            },
        )
    return _pass(
        CHECK_EXECUTION_OPERATIONS,
        "Executable operations match the write-lock draft.",
        {"operation_count": len(operations), "operation_ids": operation_ids},
    )


def _preview_from_lock(
    *,
    lock: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
) -> MemoryEvidenceRepairExecutorPreview:
    steps = tuple(_steps_from_lock(lock=lock, operations=operations))
    step_ids = tuple(step.id for step in steps)
    operation_ids = tuple(_list_of_str(lock.get("operation_ids")))
    candidate_ids = tuple(_list_of_str(lock.get("candidate_ids")))
    patch_digests = tuple(_list_of_str(lock.get("patch_digests")))
    preview_seed = {
        "lock_id": _str(lock.get("id")),
        "receipt_id": _str(lock.get("receipt_id")),
        "token_id": _str(lock.get("token_id")),
        "operation_ids": operation_ids,
        "candidate_ids": candidate_ids,
        "patch_digests": patch_digests,
        "step_ids": step_ids,
        "execution_mode": "manual_ordered_commit",
        "failure_policy": "stop_on_first_write_failure_then_release_lock",
    }
    preview_digest = _digest(preview_seed)
    preview_id = f"executor-preview-{preview_digest[:16]}"
    return MemoryEvidenceRepairExecutorPreview(
        id=preview_id,
        status=EXECUTOR_PREVIEW_STATUS_READY,
        lock_id=_str(lock.get("id")),
        receipt_id=_str(lock.get("receipt_id")),
        token_id=_str(lock.get("token_id")),
        operation_ids=operation_ids,
        candidate_ids=candidate_ids,
        patch_digests=patch_digests,
        step_ids=step_ids,
        preview_digest=preview_digest,
        preview_preview=f"{preview_id}:{preview_digest[:12]}",
        execution_mode="manual_ordered_commit",
        failure_policy="stop_on_first_write_failure_then_release_lock",
        steps=steps,
        would_apply_memory_patches=True,
        would_record_receipt=True,
        would_mark_token_used=True,
        would_release_write_lock=True,
        safety_note=(
            "Read-only executor preview. It describes a future ordered manual "
            "commit and does not write memory, acquire locks, or mark tokens used."
        ),
        source_lock=dict(lock),
    )


def _steps_from_lock(
    *,
    lock: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
) -> list[MemoryEvidenceRepairExecutorStep]:
    steps: list[MemoryEvidenceRepairExecutorStep] = []
    steps.append(
        _step(
            sequence=1,
            phase="preflight",
            action="verify_write_lock_and_receipt",
            description="Verify write-lock, commit receipt, token, and operation digests.",
            lock=lock,
        )
    )
    sequence = 2
    for operation in operations:
        steps.append(
            _operation_step(
                sequence=sequence,
                operation=operation,
            )
        )
        sequence += 1
    steps.append(
        _step(
            sequence=sequence,
            phase="audit",
            action="record_commit_receipt",
            description="Record the manual commit receipt after all memory patch steps succeed.",
            lock=lock,
            future_would_write_memory=False,
            rollback_hint="skip_receipt_recording_if_any_write_step_fails",
        )
    )
    sequence += 1
    steps.append(
        _step(
            sequence=sequence,
            phase="audit",
            action="mark_approval_token_used",
            description="Mark the approval token as used only after receipt recording is ready.",
            lock=lock,
            future_would_write_memory=False,
            rollback_hint="do_not_mark_token_used_if_receipt_recording_fails",
        )
    )
    sequence += 1
    steps.append(
        _step(
            sequence=sequence,
            phase="cleanup",
            action="release_write_lock",
            description="Release the write lock regardless of success or failure.",
            lock=lock,
            future_would_write_memory=False,
            stop_on_failure=False,
            rollback_hint="release_lock_in_finally_block",
        )
    )
    return steps


def _operation_step(
    *,
    sequence: int,
    operation: Mapping[str, Any],
) -> MemoryEvidenceRepairExecutorStep:
    operation_id = _str(operation.get("id"))
    return MemoryEvidenceRepairExecutorStep(
        id=f"executor-step-{sequence}-{operation_id or 'operation'}",
        sequence=sequence,
        phase="write",
        action="apply_evidence_metadata_patch",
        description="Apply the approved evidence metadata patch for one dry-run operation.",
        operation_id=operation_id,
        candidate_id=_str(operation.get("candidate_id")),
        ledger_entry_id=_str(operation.get("ledger_entry_id")),
        patch_digest=_str(operation.get("patch_digest")),
        target=_dict(operation.get("target")),
        evidence_fields=tuple(_list_of_str(operation.get("evidence_fields"))),
        source_operation=dict(operation),
        future_would_write_memory=True,
        stop_on_failure=True,
        rollback_hint="stop_executor_and_use_pre_commit_snapshot_or_rollback_plan",
    )


def _step(
    *,
    sequence: int,
    phase: str,
    action: str,
    description: str,
    lock: Mapping[str, Any],
    future_would_write_memory: bool = False,
    stop_on_failure: bool = True,
    rollback_hint: str = "stop_executor_before_memory_write",
) -> MemoryEvidenceRepairExecutorStep:
    return MemoryEvidenceRepairExecutorStep(
        id=f"executor-step-{sequence}-{action}",
        sequence=sequence,
        phase=phase,
        action=action,
        description=description,
        operation_id="",
        candidate_id="",
        ledger_entry_id="",
        patch_digest="",
        target={
            "lock_id": _str(lock.get("id")),
            "receipt_id": _str(lock.get("receipt_id")),
            "token_id": _str(lock.get("token_id")),
        },
        evidence_fields=(),
        source_operation={},
        future_would_write_memory=future_would_write_memory,
        stop_on_failure=stop_on_failure,
        rollback_hint=rollback_hint,
    )


def _operations_from_lock(lock: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    receipt = _dict(lock.get("source_receipt"))
    source_gate = _dict(receipt.get("source_gate"))
    dry_run = _dict(source_gate.get("source_dry_run"))
    operations = [
        operation
        for operation in _list(dry_run.get("operations"))
        if isinstance(operation, Mapping)
    ]
    if operations:
        return operations
    operation_ids = _list_of_str(lock.get("operation_ids"))
    candidate_ids = _list_of_str(lock.get("candidate_ids"))
    patch_digests = _list_of_str(lock.get("patch_digests"))
    return [
        {
            "id": operation_id,
            "candidate_id": candidate_ids[index] if index < len(candidate_ids) else "",
            "patch_digest": patch_digests[index] if index < len(patch_digests) else "",
            "target": {},
            "evidence_fields": [],
        }
        for index, operation_id in enumerate(operation_ids)
    ]


def _expected_lock_digest(lock: Mapping[str, Any]) -> str:
    seed = {
        "lock_type": WRITE_LOCK_TYPE,
        "scope": RECEIPT_SCOPE,
        "owner": _str(lock.get("owner")) or "human",
        "receipt_id": _str(lock.get("receipt_id")),
        "token_id": _str(lock.get("token_id")),
        "operation_ids": tuple(_list_of_str(lock.get("operation_ids"))),
        "candidate_ids": tuple(_list_of_str(lock.get("candidate_ids"))),
        "patch_digests": tuple(_list_of_str(lock.get("patch_digests"))),
        "expires_in_minutes": _positive_int(lock.get("expires_in_minutes")) or 10,
    }
    return _digest(seed)


def _empty_report(
    *,
    write_lock_gate: Mapping[str, Any],
) -> MemoryEvidenceRepairExecutorPreviewReport:
    return MemoryEvidenceRepairExecutorPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_write_lock_gate=dict(write_lock_gate),
    )


def _blocked_report(
    *,
    write_lock_gate: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairExecutorPreviewCheck, ...],
) -> MemoryEvidenceRepairExecutorPreviewReport:
    return MemoryEvidenceRepairExecutorPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=EXECUTOR_PREVIEW_STATUS_BLOCKED,
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
        source_write_lock_gate=dict(write_lock_gate),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairExecutorPreviewCheck:
    return MemoryEvidenceRepairExecutorPreviewCheck(
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
) -> MemoryEvidenceRepairExecutorPreviewCheck:
    return MemoryEvidenceRepairExecutorPreviewCheck(
        id=check_id,
        status=CHECK_STATUS_FAIL,
        reason=reason,
        required_actions=tuple(required_actions),
        details=dict(details),
    )


def _parse_datetime(value: str | datetime | Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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
