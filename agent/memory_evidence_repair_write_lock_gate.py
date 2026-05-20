"""Read-only write-lock gate for memory evidence repair manual commits.

The write-lock gate verifies a commit receipt, checks supplied active locks and
used-token ids, then drafts a future write lock. It never writes to durable
memory stores and never acquires a real lock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_commit_receipt import (
    RECEIPT_SCOPE,
    RECEIPT_STATUS_BLOCKED,
    RECEIPT_STATUS_NO_ACTION_NEEDED,
    RECEIPT_STATUS_READY,
    RECEIPT_TYPE,
)
from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
)


WRITE_LOCK_TYPE = "memory_evidence_repair_write_lock"
WRITE_LOCK_STATUS_READY = "write_lock_ready_for_manual_commit"
WRITE_LOCK_STATUS_BLOCKED = "blocked"
WRITE_LOCK_STATUS_NO_ACTION_NEEDED = "no_action_needed"
LOCK_DRAFT_STATUS_READY = "write_lock_draft_ready"
DEFAULT_LOCK_TTL_MINUTES = 10

CHECK_RECEIPT_READY = "receipt_ready"
CHECK_RECEIPT_INTEGRITY = "receipt_integrity"
CHECK_TOKEN_REUSE = "token_reuse"
CHECK_ACTIVE_LOCKS = "active_locks"


@dataclass(frozen=True)
class MemoryEvidenceRepairWriteLockCheck:
    """One read-only write-lock readiness check."""

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
class MemoryEvidenceRepairWriteLockDraft:
    """One read-only future write-lock draft."""

    id: str
    lock_type: str
    status: str
    scope: str
    owner: str
    receipt_id: str
    token_id: str
    operation_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    patch_digests: tuple[str, ...]
    lock_digest: str
    lock_preview: str
    issued_at: str
    expires_at: str
    expires_in_minutes: int
    would_acquire_write_lock: bool
    would_release_after_commit: bool
    safety_note: str
    source_receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lock_type": self.lock_type,
            "status": self.status,
            "scope": self.scope,
            "owner": self.owner,
            "receipt_id": self.receipt_id,
            "token_id": self.token_id,
            "operation_ids": list(self.operation_ids),
            "candidate_ids": list(self.candidate_ids),
            "patch_digests": list(self.patch_digests),
            "lock_digest": self.lock_digest,
            "lock_preview": self.lock_preview,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "expires_in_minutes": self.expires_in_minutes,
            "would_acquire_write_lock": self.would_acquire_write_lock,
            "would_release_after_commit": self.would_release_after_commit,
            "safety_note": self.safety_note,
            "source_receipt": dict(self.source_receipt),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairWriteLockGateReport:
    """Read-only write-lock gate report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairWriteLockCheck, ...]
    locks: tuple[MemoryEvidenceRepairWriteLockDraft, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    active_lock_conflicts: tuple[dict[str, Any], ...]
    source_receipt: dict[str, Any]

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
            "write_lock_available": bool(self.locks),
            "executor_dry_run_allowed": self.status == WRITE_LOCK_STATUS_READY,
            "would_acquire_write_lock_count": sum(
                1 for lock in self.locks if lock.would_acquire_write_lock
            ),
            "has_blocks": self.status == WRITE_LOCK_STATUS_BLOCKED,
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
            "source_receipt": dict(self.source_receipt),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_write_lock_gate(
    *,
    commit_receipt: Mapping[str, Any] | None = None,
    active_locks: Sequence[Mapping[str, Any]] | None = None,
    already_used_token_ids: Sequence[str] | None = None,
    lock_owner: str = "human",
    lock_ttl_minutes: int = DEFAULT_LOCK_TTL_MINUTES,
    current_time: str | datetime | None = None,
) -> MemoryEvidenceRepairWriteLockGateReport:
    """Build a read-only write-lock gate from a commit receipt report."""

    commit_receipt = commit_receipt if isinstance(commit_receipt, Mapping) else {}
    receipt = _extract_receipt(commit_receipt)
    receipt_status = _str(commit_receipt.get("status")) or _str(receipt.get("status"))
    now = _parse_datetime(current_time) or datetime.now(timezone.utc)

    if not receipt:
        if not commit_receipt or receipt_status == RECEIPT_STATUS_NO_ACTION_NEEDED:
            return _empty_report(commit_receipt=commit_receipt)
        source_blocking = tuple(_list_of_str(commit_receipt.get("blocking_reasons")))
        source_actions = tuple(_list_of_str(commit_receipt.get("required_actions")))
        return _blocked_report(
            commit_receipt=commit_receipt,
            checks=(
                _fail(
                    CHECK_RECEIPT_READY,
                    "No commit receipt was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions or ("produce_commit_receipt",),
                    {"receipt_status": receipt_status},
                ),
            ),
            conflicts=(),
        )

    checks: list[MemoryEvidenceRepairWriteLockCheck] = [
        _receipt_ready_check(commit_receipt, receipt),
        _receipt_integrity_check(receipt),
        _token_reuse_check(receipt, already_used_token_ids),
    ]
    conflicts = tuple(_active_lock_conflicts(receipt, active_locks, now))
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
        return MemoryEvidenceRepairWriteLockGateReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=WRITE_LOCK_STATUS_BLOCKED,
            checks=tuple(checks),
            locks=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            active_lock_conflicts=conflicts,
            source_receipt=dict(commit_receipt),
        )

    lock = _lock_from_receipt(
        receipt=receipt,
        lock_owner=lock_owner,
        lock_ttl_minutes=lock_ttl_minutes,
        issued_at=now,
    )
    return MemoryEvidenceRepairWriteLockGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=WRITE_LOCK_STATUS_READY,
        checks=tuple(checks),
        locks=(lock,),
        blocking_reasons=(),
        required_actions=(),
        active_lock_conflicts=(),
        source_receipt=dict(commit_receipt),
    )


def empty_evidence_repair_write_lock_gate() -> MemoryEvidenceRepairWriteLockGateReport:
    """Return an empty read-only write-lock gate report."""

    return _empty_report(commit_receipt={})


def _extract_receipt(commit_receipt: Mapping[str, Any]) -> dict[str, Any]:
    receipts = commit_receipt.get("receipts")
    if isinstance(receipts, list):
        for receipt in receipts:
            if isinstance(receipt, Mapping):
                return dict(receipt)
    if _str(commit_receipt.get("id")).startswith("commit-receipt-"):
        return dict(commit_receipt)
    return {}


def _receipt_ready_check(
    commit_receipt: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> MemoryEvidenceRepairWriteLockCheck:
    report_status = _str(commit_receipt.get("status"))
    receipt_status = _str(receipt.get("status"))
    if report_status and report_status != RECEIPT_STATUS_READY:
        return _fail(
            CHECK_RECEIPT_READY,
            "Commit receipt report is not ready for manual commit.",
            tuple(_list_of_str(commit_receipt.get("required_actions")))
            or ("produce_ready_commit_receipt",),
            {"report_status": report_status, "receipt_status": receipt_status},
        )
    if receipt_status != RECEIPT_STATUS_READY:
        return _fail(
            CHECK_RECEIPT_READY,
            "Commit receipt entry is not ready for manual commit.",
            ("produce_ready_commit_receipt",),
            {"report_status": report_status, "receipt_status": receipt_status},
        )
    return _pass(
        CHECK_RECEIPT_READY,
        "Commit receipt is ready for manual commit.",
        {"receipt_id": _str(receipt.get("id"))},
    )


def _receipt_integrity_check(
    receipt: Mapping[str, Any],
) -> MemoryEvidenceRepairWriteLockCheck:
    expected_digest = _expected_receipt_digest(receipt)
    actual_digest = _str(receipt.get("receipt_digest"))
    expected_id = f"commit-receipt-{expected_digest[:16]}" if expected_digest else ""
    missing_fields: list[str] = []
    for field_name in ("token_id", "operation_ids", "patch_digests"):
        value = receipt.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(receipt.get("id")) != expected_id
        or _str(receipt.get("receipt_type")) != RECEIPT_TYPE
        or _str(receipt.get("scope")) != RECEIPT_SCOPE
    ):
        return _fail(
            CHECK_RECEIPT_INTEGRITY,
            "Commit receipt digest, id, type, scope, or required fields are invalid.",
            ("regenerate_commit_receipt",),
            {
                "expected_receipt_digest": expected_digest,
                "actual_receipt_digest": actual_digest,
                "expected_receipt_id": expected_id,
                "actual_receipt_id": _str(receipt.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECEIPT_INTEGRITY,
        "Commit receipt integrity checks pass.",
        {"receipt_id": expected_id, "receipt_digest": actual_digest},
    )


def _token_reuse_check(
    receipt: Mapping[str, Any],
    already_used_token_ids: Sequence[str] | None,
) -> MemoryEvidenceRepairWriteLockCheck:
    token_id = _str(receipt.get("token_id"))
    used = {_str(item) for item in already_used_token_ids or [] if _str(item)}
    if token_id and token_id in used:
        return _fail(
            CHECK_TOKEN_REUSE,
            "Approval token is already present in the external used-token ledger.",
            ("regenerate_human_approval_token",),
            {"token_id": token_id, "already_used_token_count": len(used)},
        )
    return _pass(
        CHECK_TOKEN_REUSE,
        "Approval token is not present in the external used-token ledger.",
        {"token_id": token_id, "already_used_token_count": len(used)},
    )


def _active_lock_check(
    conflicts: Sequence[Mapping[str, Any]],
) -> MemoryEvidenceRepairWriteLockCheck:
    if conflicts:
        return _fail(
            CHECK_ACTIVE_LOCKS,
            "Active write locks conflict with the planned memory evidence repair commit.",
            ("wait_for_write_lock_release", "retry_write_lock_gate"),
            {"conflict_count": len(conflicts), "conflicts": list(conflicts)},
        )
    return _pass(
        CHECK_ACTIVE_LOCKS,
        "No active write-lock conflicts were supplied.",
        {"conflict_count": 0},
    )


def _lock_from_receipt(
    *,
    receipt: Mapping[str, Any],
    lock_owner: str,
    lock_ttl_minutes: int,
    issued_at: datetime,
) -> MemoryEvidenceRepairWriteLockDraft:
    ttl = _positive_int(lock_ttl_minutes) or DEFAULT_LOCK_TTL_MINUTES
    expires_at = issued_at + timedelta(minutes=ttl)
    operation_ids = tuple(_list_of_str(receipt.get("operation_ids")))
    candidate_ids = tuple(_list_of_str(receipt.get("candidate_ids")))
    patch_digests = tuple(_list_of_str(receipt.get("patch_digests")))
    seed = {
        "lock_type": WRITE_LOCK_TYPE,
        "scope": RECEIPT_SCOPE,
        "owner": _str(lock_owner) or "human",
        "receipt_id": _str(receipt.get("id")),
        "token_id": _str(receipt.get("token_id")),
        "operation_ids": operation_ids,
        "candidate_ids": candidate_ids,
        "patch_digests": patch_digests,
        "expires_in_minutes": ttl,
    }
    lock_digest = _digest(seed)
    lock_id = f"write-lock-{lock_digest[:16]}"
    return MemoryEvidenceRepairWriteLockDraft(
        id=lock_id,
        lock_type=WRITE_LOCK_TYPE,
        status=LOCK_DRAFT_STATUS_READY,
        scope=RECEIPT_SCOPE,
        owner=_str(lock_owner) or "human",
        receipt_id=_str(receipt.get("id")),
        token_id=_str(receipt.get("token_id")),
        operation_ids=operation_ids,
        candidate_ids=candidate_ids,
        patch_digests=patch_digests,
        lock_digest=lock_digest,
        lock_preview=f"{lock_id}:{lock_digest[:12]}",
        issued_at=issued_at.isoformat(),
        expires_at=expires_at.isoformat(),
        expires_in_minutes=ttl,
        would_acquire_write_lock=True,
        would_release_after_commit=True,
        safety_note=(
            "Read-only write-lock draft. This does not acquire a real lock and "
            "does not mutate durable memory."
        ),
        source_receipt=dict(receipt),
    )


def _active_lock_conflicts(
    receipt: Mapping[str, Any],
    active_locks: Sequence[Mapping[str, Any]] | None,
    now: datetime,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for lock in active_locks or []:
        if not isinstance(lock, Mapping) or not _is_active_lock(lock, now):
            continue
        if _lock_conflicts_with_receipt(lock, receipt):
            conflicts.append(
                {
                    "lock_id": _str(lock.get("id")),
                    "owner": _str(lock.get("owner")),
                    "scope": _str(lock.get("scope")),
                    "expires_at": _str(lock.get("expires_at")),
                    "reason": _conflict_reason(lock, receipt),
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
    return status in {"", "active", "write_lock_active", LOCK_DRAFT_STATUS_READY}


def _lock_conflicts_with_receipt(lock: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
    lock_scope = _str(lock.get("scope"))
    if lock_scope and lock_scope != RECEIPT_SCOPE:
        return False
    if _str(lock.get("token_id")) and _str(lock.get("token_id")) == _str(receipt.get("token_id")):
        return True
    if _str(lock.get("receipt_id")) and _str(lock.get("receipt_id")) == _str(receipt.get("id")):
        return True
    if _intersects(_list_of_str(lock.get("operation_ids")), _list_of_str(receipt.get("operation_ids"))):
        return True
    if _intersects(_list_of_str(lock.get("candidate_ids")), _list_of_str(receipt.get("candidate_ids"))):
        return True
    if _intersects(_list_of_str(lock.get("patch_digests")), _list_of_str(receipt.get("patch_digests"))):
        return True
    identifiers = (
        _str(lock.get("token_id")),
        _str(lock.get("receipt_id")),
        *_list_of_str(lock.get("operation_ids")),
        *_list_of_str(lock.get("candidate_ids")),
        *_list_of_str(lock.get("patch_digests")),
    )
    return not any(identifiers)


def _conflict_reason(lock: Mapping[str, Any], receipt: Mapping[str, Any]) -> str:
    if _str(lock.get("token_id")) == _str(receipt.get("token_id")):
        return "same_token_id"
    if _str(lock.get("receipt_id")) == _str(receipt.get("id")):
        return "same_receipt_id"
    if _intersects(_list_of_str(lock.get("operation_ids")), _list_of_str(receipt.get("operation_ids"))):
        return "overlapping_operation_ids"
    if _intersects(_list_of_str(lock.get("candidate_ids")), _list_of_str(receipt.get("candidate_ids"))):
        return "overlapping_candidate_ids"
    if _intersects(_list_of_str(lock.get("patch_digests")), _list_of_str(receipt.get("patch_digests"))):
        return "overlapping_patch_digests"
    return "global_scope_lock"


def _expected_receipt_digest(receipt: Mapping[str, Any]) -> str:
    seed = {
        "receipt_type": RECEIPT_TYPE,
        "scope": RECEIPT_SCOPE,
        "actor": _str(receipt.get("actor")) or "human",
        "commit_reason": _str(receipt.get("commit_reason")),
        "token_id": _str(receipt.get("token_id")),
        "token_digest": _str(receipt.get("token_digest")),
        "dry_run_digest": _str(receipt.get("dry_run_digest")),
        "gate_status": _str(receipt.get("gate_status")),
        "operation_ids": _list_of_str(receipt.get("operation_ids")),
        "ledger_entry_ids": _list_of_str(receipt.get("ledger_entry_ids")),
        "candidate_ids": _list_of_str(receipt.get("candidate_ids")),
        "patch_digests": _list_of_str(receipt.get("patch_digests")),
    }
    return _digest(seed)


def _empty_report(*, commit_receipt: Mapping[str, Any]) -> MemoryEvidenceRepairWriteLockGateReport:
    return MemoryEvidenceRepairWriteLockGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=WRITE_LOCK_STATUS_NO_ACTION_NEEDED,
        checks=(),
        locks=(),
        blocking_reasons=(),
        required_actions=(),
        active_lock_conflicts=(),
        source_receipt=dict(commit_receipt),
    )


def _blocked_report(
    *,
    commit_receipt: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairWriteLockCheck, ...],
    conflicts: Sequence[Mapping[str, Any]],
) -> MemoryEvidenceRepairWriteLockGateReport:
    return MemoryEvidenceRepairWriteLockGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=WRITE_LOCK_STATUS_BLOCKED,
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
        source_receipt=dict(commit_receipt),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairWriteLockCheck:
    return MemoryEvidenceRepairWriteLockCheck(
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
) -> MemoryEvidenceRepairWriteLockCheck:
    return MemoryEvidenceRepairWriteLockCheck(
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
