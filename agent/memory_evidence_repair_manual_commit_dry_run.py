"""Read-only manual commit dry-run for memory evidence repair.

The dry-run layer summarizes whether a future manual evidence repair write is
ready after commit gate, ledger, rollback, and snapshot checks. It never writes
to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT = "ready_for_manual_commit"
DRY_RUN_STATUS_BLOCKED = "blocked"
DRY_RUN_STATUS_NO_ACTION_NEEDED = "no_action_needed"

CHECK_STATUS_PASS = "pass"
CHECK_STATUS_FAIL = "fail"
CHECK_STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class MemoryEvidenceRepairDryRunCheck:
    """One readiness check in the manual commit dry-run."""

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
class MemoryEvidenceRepairDryRunOperation:
    """One future manual commit operation preview."""

    id: str
    ledger_entry_id: str
    candidate_id: str
    preview_id: str
    provider: str
    repair_action: str
    patch_digest: str
    target: dict[str, Any]
    evidence_fields: tuple[str, ...]
    source_entry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ledger_entry_id": self.ledger_entry_id,
            "candidate_id": self.candidate_id,
            "preview_id": self.preview_id,
            "provider": self.provider,
            "repair_action": self.repair_action,
            "patch_digest": self.patch_digest,
            "target": dict(self.target),
            "evidence_fields": list(self.evidence_fields),
            "source_entry": dict(self.source_entry),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairManualCommitDryRun:
    """Read-only manual commit dry-run report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairDryRunCheck, ...]
    operations: tuple[MemoryEvidenceRepairDryRunOperation, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "operation_count": len(self.operations),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "by_check_status": by_check_status,
            "manual_commit_allowed": self.status == DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT,
            "has_blocks": self.status == DRY_RUN_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "operations": [operation.to_dict() for operation in self.operations],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_manual_commit_dry_run(
    *,
    snapshot_plan: Mapping[str, Any] | None = None,
    commit_gate: Mapping[str, Any] | None = None,
    ledger: Mapping[str, Any] | None = None,
    rollback_plan: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairManualCommitDryRun:
    """Build a read-only final dry-run before any manual repair write."""

    snapshot_plan = snapshot_plan if isinstance(snapshot_plan, Mapping) else {}
    commit_gate = commit_gate if isinstance(commit_gate, Mapping) else {}
    ledger = ledger if isinstance(ledger, Mapping) else {}
    rollback_plan = rollback_plan if isinstance(rollback_plan, Mapping) else {}

    checks = (
        _commit_gate_check(commit_gate),
        _ledger_check(ledger),
        _rollback_check(rollback_plan),
        _snapshot_check(snapshot_plan),
    )
    operations = tuple(_operations_from_ledger(ledger))
    blocking_reasons = tuple(
        _dedupe_strings(
            reason
            for check in checks
            if check.status == CHECK_STATUS_FAIL
            for reason in [check.reason]
            if reason
        )
    )
    required_actions = tuple(
        _dedupe_strings(
            action
            for check in checks
            if check.status == CHECK_STATUS_FAIL
            for action in check.required_actions
        )
    )
    status = _dry_run_status(checks=checks, operations=operations)
    return MemoryEvidenceRepairManualCommitDryRun(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        checks=checks,
        operations=operations,
        blocking_reasons=blocking_reasons,
        required_actions=required_actions,
    )


def empty_evidence_repair_manual_commit_dry_run() -> MemoryEvidenceRepairManualCommitDryRun:
    """Return an empty read-only manual commit dry-run."""

    return MemoryEvidenceRepairManualCommitDryRun(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=DRY_RUN_STATUS_NO_ACTION_NEEDED,
        checks=(),
        operations=(),
        blocking_reasons=(),
        required_actions=(),
    )


def _commit_gate_check(commit_gate: Mapping[str, Any]) -> MemoryEvidenceRepairDryRunCheck:
    summary = _dict(commit_gate.get("summary"))
    decision_count = _int(summary.get("decision_count"))
    if decision_count == 0:
        return _check(
            "commit_gate",
            CHECK_STATUS_SKIPPED,
            "No commit gate decisions were supplied.",
            (),
            summary,
        )
    blocked_count = _int(summary.get("blocked_count"))
    needs_confirmation_count = _int(summary.get("needs_confirmation_count"))
    allow_count = _int(summary.get("allow_count"))
    if blocked_count or needs_confirmation_count or allow_count == 0:
        actions: list[str] = []
        if blocked_count:
            actions.append("resolve_commit_gate_blocks")
        if needs_confirmation_count:
            actions.append("obtain_explicit_user_confirmation")
        if allow_count == 0:
            actions.append("produce_allowed_commit_gate_decision")
        return _check(
            "commit_gate",
            CHECK_STATUS_FAIL,
            "Commit gate is not allowing manual commit.",
            tuple(actions),
            summary,
        )
    return _check(
        "commit_gate",
        CHECK_STATUS_PASS,
        "Commit gate allows manual commit.",
        (),
        summary,
    )


def _ledger_check(ledger: Mapping[str, Any]) -> MemoryEvidenceRepairDryRunCheck:
    summary = _dict(ledger.get("summary"))
    entry_count = _int(summary.get("entry_count"))
    if entry_count == 0:
        return _check(
            "commit_ledger",
            CHECK_STATUS_SKIPPED,
            "No commit ledger entries were supplied.",
            (),
            summary,
        )
    allow_count = _int(summary.get("allow_count"))
    blocked_count = _int(summary.get("blocked_count"))
    followup_count = _int(summary.get("followup_count"))
    if blocked_count or followup_count or allow_count == 0:
        actions: list[str] = []
        if blocked_count:
            actions.append("resolve_blocked_ledger_entries")
        if followup_count:
            actions.append("complete_ledger_followups")
        if allow_count == 0:
            actions.append("produce_allowed_ledger_entry")
        return _check(
            "commit_ledger",
            CHECK_STATUS_FAIL,
            "Commit ledger is not clean for manual commit.",
            tuple(actions),
            summary,
        )
    return _check(
        "commit_ledger",
        CHECK_STATUS_PASS,
        "Commit ledger contains allowed entries with no follow-up.",
        (),
        summary,
    )


def _rollback_check(rollback_plan: Mapping[str, Any]) -> MemoryEvidenceRepairDryRunCheck:
    summary = _dict(rollback_plan.get("summary"))
    step_count = _int(summary.get("rollback_step_count"))
    if step_count == 0:
        return _check(
            "rollback_plan",
            CHECK_STATUS_SKIPPED,
            "No rollback steps were supplied.",
            (),
            summary,
        )
    snapshot_required_count = _int(summary.get("snapshot_required_count"))
    ready_count = _int(summary.get("ready_count"))
    if snapshot_required_count or ready_count == 0:
        actions: list[str] = []
        if snapshot_required_count:
            actions.append("capture_pre_commit_snapshots")
        if ready_count == 0:
            actions.append("produce_ready_rollback_plan")
        return _check(
            "rollback_plan",
            CHECK_STATUS_FAIL,
            "Rollback plan is not ready for manual rollback.",
            tuple(actions),
            summary,
        )
    return _check(
        "rollback_plan",
        CHECK_STATUS_PASS,
        "Rollback plan is ready.",
        (),
        summary,
    )


def _snapshot_check(snapshot_plan: Mapping[str, Any]) -> MemoryEvidenceRepairDryRunCheck:
    summary = _dict(snapshot_plan.get("summary"))
    request_count = _int(summary.get("snapshot_request_count"))
    if request_count == 0:
        return _check(
            "snapshot_plan",
            CHECK_STATUS_SKIPPED,
            "No snapshot requests were supplied.",
            (),
            summary,
        )
    capture_required_count = _int(summary.get("capture_required_count"))
    blocking_count = _int(summary.get("blocking_count"))
    if capture_required_count or blocking_count:
        return _check(
            "snapshot_plan",
            CHECK_STATUS_FAIL,
            "Pre-commit snapshots are still required.",
            ("capture_pre_commit_snapshots",),
            summary,
        )
    return _check(
        "snapshot_plan",
        CHECK_STATUS_PASS,
        "Pre-commit snapshots are available or not needed.",
        (),
        summary,
    )


def _operations_from_ledger(
    ledger: Mapping[str, Any],
) -> list[MemoryEvidenceRepairDryRunOperation]:
    operations: list[MemoryEvidenceRepairDryRunOperation] = []
    for entry in _list(ledger.get("entries")):
        if not isinstance(entry, Mapping):
            continue
        if not bool(entry.get("manual_commit_allowed")):
            continue
        entry_id = _str(entry.get("id"))
        patch_digest = _str(entry.get("patch_digest"))
        operations.append(
            MemoryEvidenceRepairDryRunOperation(
                id=_stable_operation_id(entry_id=entry_id, patch_digest=patch_digest),
                ledger_entry_id=entry_id,
                candidate_id=_str(entry.get("candidate_id")),
                preview_id=_str(entry.get("preview_id")),
                provider=_str(entry.get("provider")) or "memory",
                repair_action=_str(entry.get("repair_action")) or "attach_provenance",
                patch_digest=patch_digest,
                target=_dict(entry.get("target")),
                evidence_fields=tuple(_str(field) for field in _list(entry.get("evidence_fields")) if _str(field)),
                source_entry=dict(entry),
            )
        )
    return _dedupe_operations(operations)


def _dry_run_status(
    *,
    checks: tuple[MemoryEvidenceRepairDryRunCheck, ...],
    operations: tuple[MemoryEvidenceRepairDryRunOperation, ...],
) -> str:
    if any(check.status == CHECK_STATUS_FAIL for check in checks):
        return DRY_RUN_STATUS_BLOCKED
    if not operations:
        return DRY_RUN_STATUS_NO_ACTION_NEEDED
    return DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT


def _check(
    check_id: str,
    status: str,
    reason: str,
    required_actions: tuple[str, ...],
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairDryRunCheck:
    return MemoryEvidenceRepairDryRunCheck(
        id=check_id,
        status=status,
        reason=reason,
        required_actions=required_actions,
        details=dict(details),
    )


def _stable_operation_id(*, entry_id: str, patch_digest: str) -> str:
    raw = json.dumps(
        {"entry_id": entry_id, "patch_digest": patch_digest},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"dryrun-op-{digest}"


def _dedupe_operations(
    operations: list[MemoryEvidenceRepairDryRunOperation],
) -> list[MemoryEvidenceRepairDryRunOperation]:
    seen: set[str] = set()
    deduped: list[MemoryEvidenceRepairDryRunOperation] = []
    for operation in operations:
        if operation.id in seen:
            continue
        seen.add(operation.id)
        deduped.append(operation)
    return deduped


def _dedupe_strings(values) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _str(value: Any) -> str:
    return str(value or "").strip()
