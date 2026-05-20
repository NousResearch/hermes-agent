"""Read-only write-lock gate for memory evidence repair recovery.

The recovery write-lock gate verifies a recovery approval token gate, checks
supplied active locks and used-token ids, then drafts a future recovery write
lock. It never writes to durable memory stores and never acquires a real lock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
)
from agent.memory_evidence_repair_recovery_approval_token_gate import (
    RECOVERY_TOKEN_GATE_STATUS_BLOCKED,
    RECOVERY_TOKEN_GATE_STATUS_NO_ACTION_NEEDED,
    RECOVERY_TOKEN_GATE_STATUS_VERIFIED,
)
from agent.memory_evidence_repair_recovery_execution_preview import (
    RECOVERY_EXECUTION_STATUS_READY,
)


RECOVERY_WRITE_LOCK_TYPE = "memory_evidence_repair_recovery_write_lock"
RECOVERY_WRITE_LOCK_SCOPE = "manual_memory_evidence_repair_recovery"
RECOVERY_WRITE_LOCK_STATUS_READY = "recovery_write_lock_ready_for_manual_recovery"
RECOVERY_WRITE_LOCK_STATUS_BLOCKED = "blocked"
RECOVERY_WRITE_LOCK_STATUS_NO_ACTION_NEEDED = "no_action_needed"
RECOVERY_LOCK_DRAFT_STATUS_READY = "recovery_write_lock_draft_ready"
DEFAULT_RECOVERY_LOCK_TTL_MINUTES = 10

CHECK_RECOVERY_TOKEN_GATE_READY = "recovery_token_gate_ready"
CHECK_RECOVERY_TOKEN_GATE_INTEGRITY = "recovery_token_gate_integrity"
CHECK_RECOVERY_TOKEN_REUSE = "recovery_token_reuse"
CHECK_RECOVERY_ACTIVE_LOCKS = "recovery_active_locks"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryWriteLockCheck:
    """One read-only recovery write-lock readiness check."""

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
class MemoryEvidenceRepairRecoveryWriteLockDraft:
    """One read-only future recovery write-lock draft."""

    id: str
    lock_type: str
    status: str
    scope: str
    owner: str
    token_id: str
    decision_id: str
    execution_preview_id: str
    operation_ids: tuple[str, ...]
    step_ids: tuple[str, ...]
    future_mutation_step_ids: tuple[str, ...]
    lock_digest: str
    lock_preview: str
    issued_at: str
    expires_at: str
    expires_in_minutes: int
    would_acquire_write_lock: bool
    would_release_after_recovery: bool
    safety_note: str
    source_recovery_token_gate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lock_type": self.lock_type,
            "status": self.status,
            "scope": self.scope,
            "owner": self.owner,
            "token_id": self.token_id,
            "decision_id": self.decision_id,
            "execution_preview_id": self.execution_preview_id,
            "operation_ids": list(self.operation_ids),
            "step_ids": list(self.step_ids),
            "future_mutation_step_ids": list(self.future_mutation_step_ids),
            "lock_digest": self.lock_digest,
            "lock_preview": self.lock_preview,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "expires_in_minutes": self.expires_in_minutes,
            "would_acquire_write_lock": self.would_acquire_write_lock,
            "would_release_after_recovery": self.would_release_after_recovery,
            "safety_note": self.safety_note,
            "source_recovery_token_gate": dict(self.source_recovery_token_gate),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryWriteLockGateReport:
    """Read-only recovery write-lock gate report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryWriteLockCheck, ...]
    locks: tuple[MemoryEvidenceRepairRecoveryWriteLockDraft, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    active_lock_conflicts: tuple[dict[str, Any], ...]
    source_recovery_token_gate: dict[str, Any]

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
            "lock_count": len(self.locks),
            "active_lock_conflict_count": len(self.active_lock_conflicts),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_write_lock_available": bool(self.locks),
            "recovery_executor_preview_allowed": self.status == RECOVERY_WRITE_LOCK_STATUS_READY,
            "would_acquire_write_lock_count": sum(
                1 for lock in self.locks if lock.would_acquire_write_lock
            ),
            "has_blocks": self.status == RECOVERY_WRITE_LOCK_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "locks": [lock.to_dict() for lock in self.locks],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "active_lock_conflicts": [dict(conflict) for conflict in self.active_lock_conflicts],
            "source_recovery_token_gate": dict(self.source_recovery_token_gate),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_write_lock_gate(
    *,
    recovery_token_gate: Mapping[str, Any] | None = None,
    active_locks: Sequence[Mapping[str, Any]] | None = None,
    already_used_token_ids: Sequence[str] | None = None,
    lock_owner: str = "human",
    lock_ttl_minutes: int = DEFAULT_RECOVERY_LOCK_TTL_MINUTES,
    current_time: str | datetime | None = None,
) -> MemoryEvidenceRepairRecoveryWriteLockGateReport:
    """Build a read-only recovery write-lock gate from a verified token gate."""

    recovery_token_gate = recovery_token_gate if isinstance(recovery_token_gate, Mapping) else {}
    gate_status = _str(recovery_token_gate.get("status"))
    now = _parse_datetime(current_time) or datetime.now(timezone.utc)

    if not recovery_token_gate or gate_status == RECOVERY_TOKEN_GATE_STATUS_NO_ACTION_NEEDED:
        return _empty_report(recovery_token_gate=recovery_token_gate)

    if gate_status == RECOVERY_TOKEN_GATE_STATUS_BLOCKED:
        return _blocked_report(
            recovery_token_gate=recovery_token_gate,
            checks=(
                _fail(
                    CHECK_RECOVERY_TOKEN_GATE_READY,
                    "; ".join(_list_of_str(recovery_token_gate.get("blocking_reasons")))
                    or "Recovery approval token gate is blocked.",
                    tuple(_list_of_str(recovery_token_gate.get("required_actions")))
                    or ("produce_verified_recovery_approval_token_gate",),
                    {"recovery_token_gate_status": gate_status},
                ),
            ),
            conflicts=(),
        )

    checks: list[MemoryEvidenceRepairRecoveryWriteLockCheck] = [
        _token_gate_ready_check(recovery_token_gate),
        _token_gate_integrity_check(recovery_token_gate),
        _token_reuse_check(recovery_token_gate, already_used_token_ids),
    ]
    conflicts = tuple(_active_lock_conflicts(recovery_token_gate, active_locks, now))
    checks.append(_active_lock_check(conflicts))

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
        return MemoryEvidenceRepairRecoveryWriteLockGateReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_WRITE_LOCK_STATUS_BLOCKED,
            checks=tuple(checks),
            locks=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            active_lock_conflicts=conflicts,
            source_recovery_token_gate=dict(recovery_token_gate),
        )

    lock = _lock_from_gate(
        recovery_token_gate=recovery_token_gate,
        lock_owner=lock_owner,
        lock_ttl_minutes=lock_ttl_minutes,
        issued_at=now,
    )
    return MemoryEvidenceRepairRecoveryWriteLockGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_WRITE_LOCK_STATUS_READY,
        checks=tuple(checks),
        locks=(lock,),
        blocking_reasons=(),
        required_actions=(),
        active_lock_conflicts=(),
        source_recovery_token_gate=dict(recovery_token_gate),
    )


def empty_evidence_repair_recovery_write_lock_gate() -> MemoryEvidenceRepairRecoveryWriteLockGateReport:
    """Return an empty read-only recovery write-lock gate report."""

    return _empty_report(recovery_token_gate={})


def _token_gate_ready_check(
    recovery_token_gate: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryWriteLockCheck:
    status = _str(recovery_token_gate.get("status"))
    if status != RECOVERY_TOKEN_GATE_STATUS_VERIFIED:
        return _fail(
            CHECK_RECOVERY_TOKEN_GATE_READY,
            "Recovery approval token gate is not verified for manual recovery.",
            tuple(_list_of_str(recovery_token_gate.get("required_actions")))
            or ("produce_verified_recovery_approval_token_gate",),
            {"recovery_token_gate_status": status},
        )
    return _pass(
        CHECK_RECOVERY_TOKEN_GATE_READY,
        "Recovery approval token gate is verified for manual recovery.",
        {"token_id": _str(recovery_token_gate.get("token_id"))},
    )


def _token_gate_integrity_check(
    recovery_token_gate: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryWriteLockCheck:
    token = _dict(recovery_token_gate.get("source_token"))
    execution = _dict(recovery_token_gate.get("source_recovery_execution_preview"))
    future_mutation_step_ids = _list_of_str(
        recovery_token_gate.get("verified_future_mutation_step_ids")
    )
    missing_fields: list[str] = []
    for field_name in ("token_id", "verified_operation_ids", "verified_step_ids"):
        value = recovery_token_gate.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    if not future_mutation_step_ids:
        missing_fields.append("verified_future_mutation_step_ids")
    if not _str(execution.get("id")):
        missing_fields.append("source_recovery_execution_preview.id")
    if not _str(execution.get("decision_id")):
        missing_fields.append("source_recovery_execution_preview.decision_id")
    if (
        missing_fields
        or _str(token.get("id")) != _str(recovery_token_gate.get("token_id"))
        or _str(execution.get("status")) != RECOVERY_EXECUTION_STATUS_READY
        or _list_of_str(execution.get("operation_ids"))
        != _list_of_str(recovery_token_gate.get("verified_operation_ids"))
        or _list_of_str(execution.get("step_ids"))
        != _list_of_str(recovery_token_gate.get("verified_step_ids"))
        or _future_mutation_step_ids(execution) != future_mutation_step_ids
    ):
        return _fail(
            CHECK_RECOVERY_TOKEN_GATE_INTEGRITY,
            "Recovery token gate source token, execution preview, or verified ids are invalid.",
            ("regenerate_recovery_approval_token_gate",),
            {
                "missing_fields": missing_fields,
                "token_id": _str(recovery_token_gate.get("token_id")),
                "source_token_id": _str(token.get("id")),
                "execution_preview_status": _str(execution.get("status")),
            },
        )
    return _pass(
        CHECK_RECOVERY_TOKEN_GATE_INTEGRITY,
        "Recovery token gate integrity checks pass.",
        {
            "token_id": _str(recovery_token_gate.get("token_id")),
            "future_mutation_step_count": len(future_mutation_step_ids),
        },
    )


def _token_reuse_check(
    recovery_token_gate: Mapping[str, Any],
    already_used_token_ids: Sequence[str] | None,
) -> MemoryEvidenceRepairRecoveryWriteLockCheck:
    token_id = _str(recovery_token_gate.get("token_id"))
    used = {_str(item) for item in already_used_token_ids or [] if _str(item)}
    if token_id and token_id in used:
        return _fail(
            CHECK_RECOVERY_TOKEN_REUSE,
            "Recovery approval token is already present in the external used-token ledger.",
            ("regenerate_recovery_human_approval_token",),
            {"token_id": token_id, "already_used_token_count": len(used)},
        )
    return _pass(
        CHECK_RECOVERY_TOKEN_REUSE,
        "Recovery approval token is not present in the external used-token ledger.",
        {"token_id": token_id, "already_used_token_count": len(used)},
    )


def _active_lock_check(
    conflicts: Sequence[Mapping[str, Any]],
) -> MemoryEvidenceRepairRecoveryWriteLockCheck:
    if conflicts:
        return _fail(
            CHECK_RECOVERY_ACTIVE_LOCKS,
            "Active recovery write locks conflict with the planned memory evidence repair recovery.",
            ("wait_for_recovery_write_lock_release", "retry_recovery_write_lock_gate"),
            {"conflict_count": len(conflicts), "conflicts": list(conflicts)},
        )
    return _pass(
        CHECK_RECOVERY_ACTIVE_LOCKS,
        "No active recovery write-lock conflicts were supplied.",
        {"conflict_count": 0},
    )


def _lock_from_gate(
    *,
    recovery_token_gate: Mapping[str, Any],
    lock_owner: str,
    lock_ttl_minutes: int,
    issued_at: datetime,
) -> MemoryEvidenceRepairRecoveryWriteLockDraft:
    ttl = _positive_int(lock_ttl_minutes) or DEFAULT_RECOVERY_LOCK_TTL_MINUTES
    expires_at = issued_at + timedelta(minutes=ttl)
    execution = _dict(recovery_token_gate.get("source_recovery_execution_preview"))
    operation_ids = tuple(_list_of_str(recovery_token_gate.get("verified_operation_ids")))
    step_ids = tuple(_list_of_str(recovery_token_gate.get("verified_step_ids")))
    future_mutation_step_ids = tuple(
        _list_of_str(recovery_token_gate.get("verified_future_mutation_step_ids"))
    )
    seed = {
        "lock_type": RECOVERY_WRITE_LOCK_TYPE,
        "scope": RECOVERY_WRITE_LOCK_SCOPE,
        "owner": _str(lock_owner) or "human",
        "token_id": _str(recovery_token_gate.get("token_id")),
        "decision_id": _str(execution.get("decision_id")),
        "execution_preview_id": _str(execution.get("id")),
        "operation_ids": operation_ids,
        "step_ids": step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "expires_in_minutes": ttl,
    }
    lock_digest = _digest(seed)
    lock_id = f"recovery-write-lock-{lock_digest[:16]}"
    return MemoryEvidenceRepairRecoveryWriteLockDraft(
        id=lock_id,
        lock_type=RECOVERY_WRITE_LOCK_TYPE,
        status=RECOVERY_LOCK_DRAFT_STATUS_READY,
        scope=RECOVERY_WRITE_LOCK_SCOPE,
        owner=_str(lock_owner) or "human",
        token_id=_str(recovery_token_gate.get("token_id")),
        decision_id=_str(execution.get("decision_id")),
        execution_preview_id=_str(execution.get("id")),
        operation_ids=operation_ids,
        step_ids=step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        lock_digest=lock_digest,
        lock_preview=f"{lock_id}:{lock_digest[:12]}",
        issued_at=issued_at.isoformat(),
        expires_at=expires_at.isoformat(),
        expires_in_minutes=ttl,
        would_acquire_write_lock=True,
        would_release_after_recovery=True,
        safety_note=(
            "Read-only recovery write-lock draft. This does not acquire a real "
            "lock and does not mutate durable memory."
        ),
        source_recovery_token_gate=dict(recovery_token_gate),
    )


def _active_lock_conflicts(
    recovery_token_gate: Mapping[str, Any],
    active_locks: Sequence[Mapping[str, Any]] | None,
    now: datetime,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for lock in active_locks or []:
        if not isinstance(lock, Mapping) or not _is_active_lock(lock, now):
            continue
        if _lock_conflicts_with_gate(lock, recovery_token_gate):
            conflicts.append(
                {
                    "lock_id": _str(lock.get("id")),
                    "owner": _str(lock.get("owner")),
                    "scope": _str(lock.get("scope")),
                    "expires_at": _str(lock.get("expires_at")),
                    "reason": _conflict_reason(lock, recovery_token_gate),
                }
            )
    return conflicts


def _is_active_lock(lock: Mapping[str, Any], now: datetime) -> bool:
    status = _str(lock.get("status")).lower()
    if status in {"released", "expired", "cancelled", "canceled", "inactive"}:
        return False
    expires_at = _parse_datetime(lock.get("expires_at"))
    if expires_at is not None and expires_at <= now:
        return False
    return status in {"", "active", "recovery_write_lock_active", RECOVERY_LOCK_DRAFT_STATUS_READY}


def _lock_conflicts_with_gate(
    lock: Mapping[str, Any],
    recovery_token_gate: Mapping[str, Any],
) -> bool:
    lock_scope = _str(lock.get("scope"))
    if lock_scope and lock_scope != RECOVERY_WRITE_LOCK_SCOPE:
        return False
    execution = _dict(recovery_token_gate.get("source_recovery_execution_preview"))
    if _str(lock.get("token_id")) and _str(lock.get("token_id")) == _str(recovery_token_gate.get("token_id")):
        return True
    if _str(lock.get("decision_id")) and _str(lock.get("decision_id")) == _str(execution.get("decision_id")):
        return True
    if _str(lock.get("execution_preview_id")) and _str(lock.get("execution_preview_id")) == _str(execution.get("id")):
        return True
    if _intersects(_list_of_str(lock.get("operation_ids")), _list_of_str(recovery_token_gate.get("verified_operation_ids"))):
        return True
    if _intersects(_list_of_str(lock.get("step_ids")), _list_of_str(recovery_token_gate.get("verified_step_ids"))):
        return True
    if _intersects(_list_of_str(lock.get("future_mutation_step_ids")), _list_of_str(recovery_token_gate.get("verified_future_mutation_step_ids"))):
        return True
    identifiers = (
        _str(lock.get("token_id")),
        _str(lock.get("decision_id")),
        _str(lock.get("execution_preview_id")),
        *_list_of_str(lock.get("operation_ids")),
        *_list_of_str(lock.get("step_ids")),
        *_list_of_str(lock.get("future_mutation_step_ids")),
    )
    return not any(identifiers)


def _conflict_reason(lock: Mapping[str, Any], recovery_token_gate: Mapping[str, Any]) -> str:
    execution = _dict(recovery_token_gate.get("source_recovery_execution_preview"))
    if _str(lock.get("token_id")) == _str(recovery_token_gate.get("token_id")):
        return "same_token_id"
    if _str(lock.get("decision_id")) == _str(execution.get("decision_id")):
        return "same_decision_id"
    if _str(lock.get("execution_preview_id")) == _str(execution.get("id")):
        return "same_execution_preview_id"
    if _intersects(_list_of_str(lock.get("operation_ids")), _list_of_str(recovery_token_gate.get("verified_operation_ids"))):
        return "overlapping_operation_ids"
    if _intersects(_list_of_str(lock.get("step_ids")), _list_of_str(recovery_token_gate.get("verified_step_ids"))):
        return "overlapping_step_ids"
    if _intersects(_list_of_str(lock.get("future_mutation_step_ids")), _list_of_str(recovery_token_gate.get("verified_future_mutation_step_ids"))):
        return "overlapping_future_mutation_step_ids"
    return "global_scope_lock"


def _future_mutation_step_ids(recovery_execution: Mapping[str, Any]) -> list[str]:
    return [
        _str(step.get("id"))
        for step in _list(recovery_execution.get("steps"))
        if isinstance(step, Mapping) and step.get("future_would_mutate_memory") is True
    ]


def _empty_report(
    *,
    recovery_token_gate: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryWriteLockGateReport:
    return MemoryEvidenceRepairRecoveryWriteLockGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_WRITE_LOCK_STATUS_NO_ACTION_NEEDED,
        checks=(),
        locks=(),
        blocking_reasons=(),
        required_actions=(),
        active_lock_conflicts=(),
        source_recovery_token_gate=dict(recovery_token_gate),
    )


def _blocked_report(
    *,
    recovery_token_gate: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryWriteLockCheck, ...],
    conflicts: Sequence[Mapping[str, Any]],
) -> MemoryEvidenceRepairRecoveryWriteLockGateReport:
    return MemoryEvidenceRepairRecoveryWriteLockGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_WRITE_LOCK_STATUS_BLOCKED,
        checks=checks,
        locks=(),
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
        active_lock_conflicts=tuple(dict(conflict) for conflict in conflicts),
        source_recovery_token_gate=dict(recovery_token_gate),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryWriteLockCheck:
    return MemoryEvidenceRepairRecoveryWriteLockCheck(
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
) -> MemoryEvidenceRepairRecoveryWriteLockCheck:
    return MemoryEvidenceRepairRecoveryWriteLockCheck(
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


def _intersects(left: Sequence[str], right: Sequence[str]) -> bool:
    return bool(set(left).intersection(set(right)))


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
