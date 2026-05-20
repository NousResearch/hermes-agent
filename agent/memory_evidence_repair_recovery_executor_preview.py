"""Read-only recovery executor preview for memory evidence repair.

The recovery executor preview turns a verified recovery write-lock gate into
an ordered manual recovery execution plan. It never executes rollback, never
mutates durable memory, and never acquires, releases, or records a real lock.
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
from agent.memory_evidence_repair_recovery_execution_preview import (
    RECOVERY_EXECUTION_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_write_lock_gate import (
    DEFAULT_RECOVERY_LOCK_TTL_MINUTES,
    RECOVERY_LOCK_DRAFT_STATUS_READY,
    RECOVERY_WRITE_LOCK_SCOPE,
    RECOVERY_WRITE_LOCK_STATUS_NO_ACTION_NEEDED,
    RECOVERY_WRITE_LOCK_STATUS_READY,
    RECOVERY_WRITE_LOCK_TYPE,
)


RECOVERY_EXECUTOR_PREVIEW_STATUS_READY = "recovery_executor_preview_ready_for_manual_recovery"
RECOVERY_EXECUTOR_PREVIEW_STATUS_BLOCKED = "blocked"
RECOVERY_EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED = "no_action_needed"

CHECK_RECOVERY_WRITE_LOCK_READY = "recovery_write_lock_ready"
CHECK_RECOVERY_WRITE_LOCK_INTEGRITY = "recovery_write_lock_integrity"
CHECK_RECOVERY_WRITE_LOCK_EXPIRY = "recovery_write_lock_expiry"
CHECK_RECOVERY_EXECUTION_STEPS = "recovery_execution_steps"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryExecutorPreviewCheck:
    """One read-only recovery executor readiness check."""

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
class MemoryEvidenceRepairRecoveryExecutorStep:
    """One future manual recovery executor step preview."""

    id: str
    sequence: int
    phase: str
    action: str
    description: str
    recovery_execution_step_id: str
    operation_id: str
    decision_id: str
    execution_preview_id: str
    receipt_id: str
    token_id: str
    lock_id: str
    route: str
    required_precondition: str
    expected_result: str
    source_recovery_execution_step: dict[str, Any]
    future_would_mutate_memory: bool
    stop_on_failure: bool
    rollback_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "phase": self.phase,
            "action": self.action,
            "description": self.description,
            "recovery_execution_step_id": self.recovery_execution_step_id,
            "operation_id": self.operation_id,
            "decision_id": self.decision_id,
            "execution_preview_id": self.execution_preview_id,
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "route": self.route,
            "required_precondition": self.required_precondition,
            "expected_result": self.expected_result,
            "source_recovery_execution_step": dict(self.source_recovery_execution_step),
            "future_would_mutate_memory": self.future_would_mutate_memory,
            "stop_on_failure": self.stop_on_failure,
            "rollback_hint": self.rollback_hint,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryExecutorPreview:
    """One read-only recovery executor preview."""

    id: str
    status: str
    lock_id: str
    token_id: str
    decision_id: str
    execution_preview_id: str
    route: str
    operation_ids: tuple[str, ...]
    step_ids: tuple[str, ...]
    future_mutation_step_ids: tuple[str, ...]
    executor_step_ids: tuple[str, ...]
    preview_digest: str
    preview_preview: str
    execution_mode: str
    failure_policy: str
    steps: tuple[MemoryEvidenceRepairRecoveryExecutorStep, ...]
    would_execute_manual_recovery: bool
    would_apply_memory_recovery: bool
    would_mark_recovery_token_used: bool
    would_release_recovery_write_lock: bool
    would_rerun_post_commit_audit: bool
    safety_note: str
    source_lock: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "lock_id": self.lock_id,
            "token_id": self.token_id,
            "decision_id": self.decision_id,
            "execution_preview_id": self.execution_preview_id,
            "route": self.route,
            "operation_ids": list(self.operation_ids),
            "step_ids": list(self.step_ids),
            "future_mutation_step_ids": list(self.future_mutation_step_ids),
            "executor_step_ids": list(self.executor_step_ids),
            "preview_digest": self.preview_digest,
            "preview_preview": self.preview_preview,
            "execution_mode": self.execution_mode,
            "failure_policy": self.failure_policy,
            "steps": [step.to_dict() for step in self.steps],
            "would_execute_manual_recovery": self.would_execute_manual_recovery,
            "would_apply_memory_recovery": self.would_apply_memory_recovery,
            "would_mark_recovery_token_used": self.would_mark_recovery_token_used,
            "would_release_recovery_write_lock": self.would_release_recovery_write_lock,
            "would_rerun_post_commit_audit": self.would_rerun_post_commit_audit,
            "safety_note": self.safety_note,
            "source_lock": dict(self.source_lock),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryExecutorPreviewReport:
    """Read-only recovery executor preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryExecutorPreviewCheck, ...]
    previews: tuple[MemoryEvidenceRepairRecoveryExecutorPreview, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_write_lock_gate: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        step_count = sum(len(preview.steps) for preview in self.previews)
        future_mutation_step_count = sum(
            1
            for preview in self.previews
            for step in preview.steps
            if step.future_would_mutate_memory
        )
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "preview_count": len(self.previews),
            "step_count": step_count,
            "future_mutation_step_count": future_mutation_step_count,
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_executor_preview_available": bool(self.previews),
            "manual_recovery_executor_preview_ready": (
                self.status == RECOVERY_EXECUTOR_PREVIEW_STATUS_READY
            ),
            "would_apply_memory_recovery_count": sum(
                1 for preview in self.previews if preview.would_apply_memory_recovery
            ),
            "has_blocks": self.status == RECOVERY_EXECUTOR_PREVIEW_STATUS_BLOCKED,
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
            "source_recovery_write_lock_gate": dict(self.source_recovery_write_lock_gate),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_executor_preview(
    *,
    recovery_write_lock_gate: Mapping[str, Any] | None = None,
    current_time: str | datetime | None = None,
) -> MemoryEvidenceRepairRecoveryExecutorPreviewReport:
    """Build a read-only recovery executor preview from a recovery write lock."""

    recovery_write_lock_gate = (
        recovery_write_lock_gate if isinstance(recovery_write_lock_gate, Mapping) else {}
    )
    gate_status = _str(recovery_write_lock_gate.get("status"))
    lock = _extract_lock(recovery_write_lock_gate)
    now = _parse_datetime(current_time) or datetime.now(timezone.utc)

    if not lock:
        if not recovery_write_lock_gate or gate_status == RECOVERY_WRITE_LOCK_STATUS_NO_ACTION_NEEDED:
            return _empty_report(recovery_write_lock_gate=recovery_write_lock_gate)
        source_blocking = tuple(_list_of_str(recovery_write_lock_gate.get("blocking_reasons")))
        source_actions = tuple(_list_of_str(recovery_write_lock_gate.get("required_actions")))
        return _blocked_report(
            recovery_write_lock_gate=recovery_write_lock_gate,
            checks=(
                _fail(
                    CHECK_RECOVERY_WRITE_LOCK_READY,
                    "No recovery write-lock draft was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions or ("produce_recovery_write_lock_gate",),
                    {"recovery_write_lock_gate_status": gate_status},
                ),
            ),
        )

    execution = _execution_from_lock(lock)
    checks = (
        _recovery_write_lock_ready_check(recovery_write_lock_gate, lock),
        _recovery_write_lock_integrity_check(lock),
        _recovery_write_lock_expiry_check(lock, now),
        _recovery_execution_steps_check(lock, execution),
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
        return MemoryEvidenceRepairRecoveryExecutorPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_EXECUTOR_PREVIEW_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_write_lock_gate=dict(recovery_write_lock_gate),
        )

    preview = _preview_from_lock(lock=lock, execution=execution)
    return MemoryEvidenceRepairRecoveryExecutorPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_EXECUTOR_PREVIEW_STATUS_READY,
        checks=checks,
        previews=(preview,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_write_lock_gate=dict(recovery_write_lock_gate),
    )


def empty_evidence_repair_recovery_executor_preview() -> MemoryEvidenceRepairRecoveryExecutorPreviewReport:
    """Return an empty read-only recovery executor preview report."""

    return _empty_report(recovery_write_lock_gate={})


def _extract_lock(recovery_write_lock_gate: Mapping[str, Any]) -> dict[str, Any]:
    locks = recovery_write_lock_gate.get("locks")
    if isinstance(locks, list):
        for lock in locks:
            if isinstance(lock, Mapping):
                return dict(lock)
    if _str(recovery_write_lock_gate.get("id")).startswith("recovery-write-lock-"):
        return dict(recovery_write_lock_gate)
    return {}


def _recovery_write_lock_ready_check(
    recovery_write_lock_gate: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutorPreviewCheck:
    gate_status = _str(recovery_write_lock_gate.get("status"))
    lock_status = _str(lock.get("status"))
    if gate_status and gate_status != RECOVERY_WRITE_LOCK_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_WRITE_LOCK_READY,
            "Recovery write-lock gate is not ready for manual recovery execution.",
            tuple(_list_of_str(recovery_write_lock_gate.get("required_actions")))
            or ("produce_ready_recovery_write_lock_gate",),
            {"recovery_write_lock_gate_status": gate_status, "lock_status": lock_status},
        )
    if lock_status != RECOVERY_LOCK_DRAFT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_WRITE_LOCK_READY,
            "Recovery write-lock draft is not ready.",
            ("produce_ready_recovery_write_lock_gate",),
            {"recovery_write_lock_gate_status": gate_status, "lock_status": lock_status},
        )
    return _pass(
        CHECK_RECOVERY_WRITE_LOCK_READY,
        "Recovery write-lock gate and draft are ready.",
        {"lock_id": _str(lock.get("id"))},
    )


def _recovery_write_lock_integrity_check(
    lock: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutorPreviewCheck:
    expected_digest = _expected_lock_digest(lock)
    actual_digest = _str(lock.get("lock_digest"))
    expected_id = f"recovery-write-lock-{expected_digest[:16]}" if expected_digest else ""
    missing_fields: list[str] = []
    for field_name in (
        "token_id",
        "decision_id",
        "execution_preview_id",
        "operation_ids",
        "step_ids",
        "future_mutation_step_ids",
    ):
        value = lock.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(lock.get("id")) != expected_id
        or _str(lock.get("lock_type")) != RECOVERY_WRITE_LOCK_TYPE
        or _str(lock.get("scope")) != RECOVERY_WRITE_LOCK_SCOPE
    ):
        return _fail(
            CHECK_RECOVERY_WRITE_LOCK_INTEGRITY,
            "Recovery write-lock digest, id, type, scope, or required fields are invalid.",
            ("regenerate_recovery_write_lock_gate",),
            {
                "expected_lock_digest": expected_digest,
                "actual_lock_digest": actual_digest,
                "expected_lock_id": expected_id,
                "actual_lock_id": _str(lock.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_WRITE_LOCK_INTEGRITY,
        "Recovery write-lock integrity checks pass.",
        {"lock_id": expected_id, "lock_digest": actual_digest},
    )


def _recovery_write_lock_expiry_check(
    lock: Mapping[str, Any],
    now: datetime,
) -> MemoryEvidenceRepairRecoveryExecutorPreviewCheck:
    expires_at = _parse_datetime(lock.get("expires_at"))
    if expires_at is None:
        return _fail(
            CHECK_RECOVERY_WRITE_LOCK_EXPIRY,
            "Recovery write-lock expiry is missing or invalid.",
            ("regenerate_recovery_write_lock_gate",),
            {"expires_at": _str(lock.get("expires_at"))},
        )
    if expires_at <= now:
        return _fail(
            CHECK_RECOVERY_WRITE_LOCK_EXPIRY,
            "Recovery write-lock draft has expired.",
            ("regenerate_recovery_write_lock_gate",),
            {"expires_at": expires_at.isoformat(), "current_time": now.isoformat()},
        )
    return _pass(
        CHECK_RECOVERY_WRITE_LOCK_EXPIRY,
        "Recovery write-lock draft is within its expiry window.",
        {"expires_at": expires_at.isoformat(), "current_time": now.isoformat()},
    )


def _recovery_execution_steps_check(
    lock: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutorPreviewCheck:
    steps = _execution_steps(execution)
    execution_step_ids = [_str(step.get("id")) for step in steps]
    execution_future_mutation_step_ids = _future_mutation_step_ids(execution)
    lock_step_ids = _list_of_str(lock.get("step_ids"))
    lock_future_mutation_step_ids = _list_of_str(lock.get("future_mutation_step_ids"))
    missing_fields: list[str] = []
    if not execution:
        missing_fields.append("source_recovery_execution_preview")
    if not steps:
        missing_fields.append("source_recovery_execution_preview.steps")
    if (
        missing_fields
        or _str(execution.get("status")) != RECOVERY_EXECUTION_STATUS_READY
        or _str(execution.get("id")) != _str(lock.get("execution_preview_id"))
        or _str(execution.get("decision_id")) != _str(lock.get("decision_id"))
        or _list_of_str(execution.get("operation_ids")) != _list_of_str(lock.get("operation_ids"))
        or execution_step_ids != lock_step_ids
        or execution_future_mutation_step_ids != lock_future_mutation_step_ids
    ):
        return _fail(
            CHECK_RECOVERY_EXECUTION_STEPS,
            "Recovery execution preview does not match the recovery write-lock draft.",
            ("regenerate_recovery_write_lock_gate",),
            {
                "missing_fields": missing_fields,
                "execution_preview_id": _str(execution.get("id")),
                "lock_execution_preview_id": _str(lock.get("execution_preview_id")),
                "execution_step_ids": execution_step_ids,
                "lock_step_ids": lock_step_ids,
                "execution_future_mutation_step_ids": execution_future_mutation_step_ids,
                "lock_future_mutation_step_ids": lock_future_mutation_step_ids,
            },
        )
    return _pass(
        CHECK_RECOVERY_EXECUTION_STEPS,
        "Recovery execution steps match the recovery write-lock draft.",
        {
            "execution_preview_id": _str(execution.get("id")),
            "step_count": len(steps),
            "future_mutation_step_count": len(execution_future_mutation_step_ids),
        },
    )


def _preview_from_lock(
    *,
    lock: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutorPreview:
    steps = tuple(_steps_from_lock(lock=lock, execution=execution))
    executor_step_ids = tuple(step.id for step in steps)
    step_ids = tuple(_list_of_str(lock.get("step_ids")))
    operation_ids = tuple(_list_of_str(lock.get("operation_ids")))
    future_mutation_step_ids = tuple(_list_of_str(lock.get("future_mutation_step_ids")))
    preview_seed = {
        "lock_id": _str(lock.get("id")),
        "token_id": _str(lock.get("token_id")),
        "decision_id": _str(lock.get("decision_id")),
        "execution_preview_id": _str(lock.get("execution_preview_id")),
        "operation_ids": operation_ids,
        "step_ids": step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "executor_step_ids": executor_step_ids,
        "execution_mode": "manual_ordered_recovery",
        "failure_policy": "stop_on_first_recovery_failure_then_release_lock",
    }
    preview_digest = _digest(preview_seed)
    preview_id = f"recovery-executor-preview-{preview_digest[:16]}"
    return MemoryEvidenceRepairRecoveryExecutorPreview(
        id=preview_id,
        status=RECOVERY_EXECUTOR_PREVIEW_STATUS_READY,
        lock_id=_str(lock.get("id")),
        token_id=_str(lock.get("token_id")),
        decision_id=_str(lock.get("decision_id")),
        execution_preview_id=_str(lock.get("execution_preview_id")),
        route=_str(execution.get("route")),
        operation_ids=operation_ids,
        step_ids=step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        executor_step_ids=executor_step_ids,
        preview_digest=preview_digest,
        preview_preview=f"{preview_id}:{preview_digest[:12]}",
        execution_mode="manual_ordered_recovery",
        failure_policy="stop_on_first_recovery_failure_then_release_lock",
        steps=steps,
        would_execute_manual_recovery=True,
        would_apply_memory_recovery=bool(future_mutation_step_ids),
        would_mark_recovery_token_used=True,
        would_release_recovery_write_lock=True,
        would_rerun_post_commit_audit=any(
            step.action.startswith("rerun_post_commit_audit") for step in steps
        ),
        safety_note=(
            "Read-only recovery executor preview. It describes a future ordered "
            "manual recovery and does not execute rollback, write memory, mark "
            "tokens used, or release a real lock."
        ),
        source_lock=dict(lock),
    )


def _steps_from_lock(
    *,
    lock: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> list[MemoryEvidenceRepairRecoveryExecutorStep]:
    steps: list[MemoryEvidenceRepairRecoveryExecutorStep] = [
        _control_step(
            sequence=1,
            phase="preflight",
            action="verify_recovery_write_lock_token_and_execution",
            description="Verify recovery write-lock, token gate, execution preview, and step ids.",
            lock=lock,
            execution=execution,
        )
    ]
    sequence = 2
    for source_step in _execution_steps(execution):
        steps.append(
            _source_step(
                sequence=sequence,
                source_step=source_step,
                lock=lock,
                execution=execution,
            )
        )
        sequence += 1
    steps.append(
        _control_step(
            sequence=sequence,
            phase="audit",
            action="mark_recovery_token_used",
            description="Mark the recovery approval token as used only after manual recovery succeeds.",
            lock=lock,
            execution=execution,
            rollback_hint="do_not_mark_recovery_token_used_if_recovery_steps_fail",
        )
    )
    sequence += 1
    steps.append(
        _control_step(
            sequence=sequence,
            phase="cleanup",
            action="release_recovery_write_lock",
            description="Release the recovery write lock regardless of success or failure.",
            lock=lock,
            execution=execution,
            stop_on_failure=False,
            rollback_hint="release_recovery_lock_in_finally_block",
        )
    )
    return steps


def _source_step(
    *,
    sequence: int,
    source_step: Mapping[str, Any],
    lock: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutorStep:
    action = _str(source_step.get("action")) or "recovery_step"
    future_mutation = source_step.get("future_would_mutate_memory") is True
    return MemoryEvidenceRepairRecoveryExecutorStep(
        id=f"recovery-executor-step-{sequence}-{action}",
        sequence=sequence,
        phase=_str(source_step.get("phase")) or "recovery",
        action=action,
        description=_str(source_step.get("description")) or "Preview the future manual recovery step.",
        recovery_execution_step_id=_str(source_step.get("id")),
        operation_id=_str(source_step.get("operation_id")),
        decision_id=_str(source_step.get("source_decision_id")) or _str(lock.get("decision_id")),
        execution_preview_id=_str(execution.get("id")),
        receipt_id=_str(source_step.get("receipt_id")),
        token_id=_str(lock.get("token_id")),
        lock_id=_str(lock.get("id")),
        route=_str(source_step.get("route")) or _str(execution.get("route")),
        required_precondition=_str(source_step.get("required_precondition")),
        expected_result=_str(source_step.get("expected_result")),
        source_recovery_execution_step=dict(source_step),
        future_would_mutate_memory=future_mutation,
        stop_on_failure=True,
        rollback_hint=_rollback_hint(action, future_mutation),
    )


def _control_step(
    *,
    sequence: int,
    phase: str,
    action: str,
    description: str,
    lock: Mapping[str, Any],
    execution: Mapping[str, Any],
    future_would_mutate_memory: bool = False,
    stop_on_failure: bool = True,
    rollback_hint: str = "stop_executor_before_recovery_mutation",
) -> MemoryEvidenceRepairRecoveryExecutorStep:
    return MemoryEvidenceRepairRecoveryExecutorStep(
        id=f"recovery-executor-step-{sequence}-{action}",
        sequence=sequence,
        phase=phase,
        action=action,
        description=description,
        recovery_execution_step_id="",
        operation_id="",
        decision_id=_str(lock.get("decision_id")),
        execution_preview_id=_str(execution.get("id")),
        receipt_id=_str(execution.get("receipt_id")),
        token_id=_str(lock.get("token_id")),
        lock_id=_str(lock.get("id")),
        route=_str(execution.get("route")),
        required_precondition="verified_recovery_write_lock",
        expected_result=f"{action}_complete",
        source_recovery_execution_step={},
        future_would_mutate_memory=future_would_mutate_memory,
        stop_on_failure=stop_on_failure,
        rollback_hint=rollback_hint,
    )


def _rollback_hint(action: str, future_mutation: bool) -> str:
    if future_mutation:
        return "stop_recovery_executor_and_use_verified_snapshot_or_rollback_plan"
    if action.startswith("rerun_post_commit_audit"):
        return "preserve_recovery_context_if_audit_still_fails"
    return "stop_recovery_executor_before_next_manual_step"


def _execution_from_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    token_gate = _dict(lock.get("source_recovery_token_gate"))
    execution = _dict(token_gate.get("source_recovery_execution_preview"))
    return execution


def _execution_steps(execution: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        step
        for step in _list(execution.get("steps"))
        if isinstance(step, Mapping)
    ]


def _future_mutation_step_ids(execution: Mapping[str, Any]) -> list[str]:
    return [
        _str(step.get("id"))
        for step in _execution_steps(execution)
        if step.get("future_would_mutate_memory") is True
    ]


def _expected_lock_digest(lock: Mapping[str, Any]) -> str:
    seed = {
        "lock_type": RECOVERY_WRITE_LOCK_TYPE,
        "scope": RECOVERY_WRITE_LOCK_SCOPE,
        "owner": _str(lock.get("owner")) or "human",
        "token_id": _str(lock.get("token_id")),
        "decision_id": _str(lock.get("decision_id")),
        "execution_preview_id": _str(lock.get("execution_preview_id")),
        "operation_ids": tuple(_list_of_str(lock.get("operation_ids"))),
        "step_ids": tuple(_list_of_str(lock.get("step_ids"))),
        "future_mutation_step_ids": tuple(_list_of_str(lock.get("future_mutation_step_ids"))),
        "expires_in_minutes": (
            _positive_int(lock.get("expires_in_minutes"))
            or DEFAULT_RECOVERY_LOCK_TTL_MINUTES
        ),
    }
    return _digest(seed)


def _empty_report(
    *,
    recovery_write_lock_gate: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutorPreviewReport:
    return MemoryEvidenceRepairRecoveryExecutorPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_write_lock_gate=dict(recovery_write_lock_gate),
    )


def _blocked_report(
    *,
    recovery_write_lock_gate: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryExecutorPreviewCheck, ...],
) -> MemoryEvidenceRepairRecoveryExecutorPreviewReport:
    return MemoryEvidenceRepairRecoveryExecutorPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_EXECUTOR_PREVIEW_STATUS_BLOCKED,
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
        source_recovery_write_lock_gate=dict(recovery_write_lock_gate),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryExecutorPreviewCheck:
    return MemoryEvidenceRepairRecoveryExecutorPreviewCheck(
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
) -> MemoryEvidenceRepairRecoveryExecutorPreviewCheck:
    return MemoryEvidenceRepairRecoveryExecutorPreviewCheck(
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
