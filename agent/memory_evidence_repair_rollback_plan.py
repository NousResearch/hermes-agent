"""Read-only rollback plans for memory evidence repair ledger entries.

The rollback layer drafts how to undo a future manual evidence repair write.
It never writes to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_commit_ledger import OUTCOME_ALLOWED


ROLLBACK_ACTION_RESTORE_EVIDENCE_METADATA = "restore_evidence_metadata"
ROLLBACK_ACTION_NOOP = "no_rollback_required"
ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK = "ready_for_manual_rollback"
ROLLBACK_STATUS_SNAPSHOT_REQUIRED = "snapshot_required"
ROLLBACK_STATUS_NO_ACTION_NEEDED = "no_action_needed"
SNAPSHOT_REQUIRED_VALUE = "<requires_pre_commit_snapshot>"


@dataclass(frozen=True)
class MemoryEvidenceRepairRollbackStep:
    """One read-only rollback step draft."""

    id: str
    ledger_entry_id: str
    sequence: int
    status: str
    rollback_action: str
    outcome: str
    provider: str
    repair_action: str
    candidate_id: str
    preview_id: str
    patch_digest: str
    evidence_fields: tuple[str, ...]
    target: dict[str, Any]
    inverse_patch_preview: dict[str, Any]
    required_preconditions: tuple[str, ...]
    verification_steps: tuple[str, ...]
    reason: str
    safety_note: str
    source_entry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ledger_entry_id": self.ledger_entry_id,
            "sequence": self.sequence,
            "status": self.status,
            "rollback_action": self.rollback_action,
            "outcome": self.outcome,
            "provider": self.provider,
            "repair_action": self.repair_action,
            "candidate_id": self.candidate_id,
            "preview_id": self.preview_id,
            "patch_digest": self.patch_digest,
            "evidence_fields": list(self.evidence_fields),
            "target": dict(self.target),
            "inverse_patch_preview": dict(self.inverse_patch_preview),
            "required_preconditions": list(self.required_preconditions),
            "verification_steps": list(self.verification_steps),
            "reason": self.reason,
            "safety_note": self.safety_note,
            "source_entry": dict(self.source_entry),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRollbackPlan:
    """Read-only rollback plan draft."""

    generated_at: str
    steps: tuple[MemoryEvidenceRepairRollbackStep, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        by_repair_action: dict[str, int] = {}
        ready_count = 0
        snapshot_required_count = 0
        no_action_count = 0
        for step in self.steps:
            by_status[step.status] = by_status.get(step.status, 0) + 1
            by_repair_action[step.repair_action] = (
                by_repair_action.get(step.repair_action, 0) + 1
            )
            if step.provider:
                by_provider[step.provider] = by_provider.get(step.provider, 0) + 1
            if step.status == ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK:
                ready_count += 1
            elif step.status == ROLLBACK_STATUS_SNAPSHOT_REQUIRED:
                snapshot_required_count += 1
            elif step.status == ROLLBACK_STATUS_NO_ACTION_NEEDED:
                no_action_count += 1
        return {
            "rollback_step_count": len(self.steps),
            "ready_count": ready_count,
            "snapshot_required_count": snapshot_required_count,
            "no_action_count": no_action_count,
            "by_status": by_status,
            "by_provider": by_provider,
            "by_repair_action": by_repair_action,
            "requires_snapshot": snapshot_required_count > 0,
            "has_ready_rollbacks": ready_count > 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_rollback_plan(
    *,
    ledger: Mapping[str, Any] | None = None,
    entries: Sequence[Mapping[str, Any]] | None = None,
    pre_commit_snapshots: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRollbackPlan:
    """Build a read-only rollback plan from commit ledger entries."""

    ledger = ledger if isinstance(ledger, Mapping) else {}
    pre_commit_snapshots = (
        pre_commit_snapshots if isinstance(pre_commit_snapshots, Mapping) else {}
    )
    rows = _entry_rows(ledger=ledger, entries=entries)
    steps: list[MemoryEvidenceRepairRollbackStep] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        step = _step_from_entry(
            row,
            sequence=index,
            pre_commit_snapshots=pre_commit_snapshots,
        )
        if step is not None:
            steps.append(step)

    return MemoryEvidenceRepairRollbackPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        steps=tuple(_dedupe_steps(steps)),
    )


def empty_evidence_repair_rollback_plan() -> MemoryEvidenceRepairRollbackPlan:
    """Return an empty read-only rollback plan."""

    return MemoryEvidenceRepairRollbackPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        steps=(),
    )


def _step_from_entry(
    entry: Mapping[str, Any],
    *,
    sequence: int,
    pre_commit_snapshots: Mapping[str, Any],
) -> MemoryEvidenceRepairRollbackStep | None:
    entry_id = _str(entry.get("id"))
    outcome = _str(entry.get("outcome"))
    candidate_id = _str(entry.get("candidate_id"))
    preview_id = _str(entry.get("preview_id"))
    if not entry_id and not outcome and not candidate_id and not preview_id:
        return None

    evidence_fields = _evidence_fields(entry)
    snapshot = _snapshot_for_entry(entry, pre_commit_snapshots, evidence_fields)
    allowed = _is_allowed(entry)
    status = _status_for_entry(allowed=allowed, snapshot=snapshot)
    rollback_action = (
        ROLLBACK_ACTION_RESTORE_EVIDENCE_METADATA
        if allowed
        else ROLLBACK_ACTION_NOOP
    )
    inverse_patch = _inverse_patch_preview(
        entry=entry,
        allowed=allowed,
        snapshot=snapshot,
    )
    return MemoryEvidenceRepairRollbackStep(
        id=_stable_step_id(entry_id=entry_id, patch_digest=_str(entry.get("patch_digest"))),
        ledger_entry_id=entry_id,
        sequence=sequence,
        status=status,
        rollback_action=rollback_action,
        outcome=outcome,
        provider=_str(entry.get("provider")) or "memory",
        repair_action=_str(entry.get("repair_action")) or "attach_provenance",
        candidate_id=candidate_id,
        preview_id=preview_id,
        patch_digest=_str(entry.get("patch_digest")),
        evidence_fields=evidence_fields,
        target=_dict(entry.get("target")),
        inverse_patch_preview=inverse_patch,
        required_preconditions=_required_preconditions(status),
        verification_steps=_verification_steps(status),
        reason=_reason(status, outcome),
        safety_note=(
            "Read-only rollback plan. This step describes a future manual "
            "rollback path and does not mutate durable memory."
        ),
        source_entry=dict(entry),
    )


def _entry_rows(
    *,
    ledger: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]] | None,
) -> list[Any]:
    if entries is not None:
        return list(entries)
    if "outcome" in ledger:
        return [ledger]
    value = ledger.get("entries")
    return value if isinstance(value, list) else []


def _is_allowed(entry: Mapping[str, Any]) -> bool:
    return bool(entry.get("manual_commit_allowed")) or _str(entry.get("outcome")) == OUTCOME_ALLOWED


def _status_for_entry(*, allowed: bool, snapshot: dict[str, Any] | None) -> str:
    if not allowed:
        return ROLLBACK_STATUS_NO_ACTION_NEEDED
    if snapshot is None:
        return ROLLBACK_STATUS_SNAPSHOT_REQUIRED
    return ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK


def _inverse_patch_preview(
    *,
    entry: Mapping[str, Any],
    allowed: bool,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    if not allowed:
        return {
            "operation": ROLLBACK_ACTION_NOOP,
            "reason": "Ledger entry was not allowed for manual commit.",
        }
    return {
        "operation": ROLLBACK_ACTION_RESTORE_EVIDENCE_METADATA,
        "target": _dict(entry.get("target")),
        "set": {
            "evidence": snapshot if snapshot is not None else SNAPSHOT_REQUIRED_VALUE,
        },
        "verify_patch_digest": _str(entry.get("patch_digest")),
        "source_ledger_entry_id": _str(entry.get("id")),
    }


def _required_preconditions(status: str) -> tuple[str, ...]:
    if status == ROLLBACK_STATUS_NO_ACTION_NEEDED:
        return ()
    preconditions = ["explicit_user_confirmation", "verify_patch_digest_matches_ledger"]
    if status == ROLLBACK_STATUS_SNAPSHOT_REQUIRED:
        preconditions.append("capture_pre_commit_snapshot_before_apply")
    if status == ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK:
        preconditions.append("manual_rollback_only")
    return tuple(preconditions)


def _verification_steps(status: str) -> tuple[str, ...]:
    if status == ROLLBACK_STATUS_NO_ACTION_NEEDED:
        return ("verify_no_manual_commit_occurred",)
    steps = [
        "verify_target_matches_ledger_entry",
        "verify_patch_digest_matches_ledger",
        "manual_review_before_rollback",
    ]
    if status == ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK:
        steps.append("restore_previous_evidence_metadata")
    else:
        steps.append("capture_pre_commit_snapshot_before_apply")
    return tuple(steps)


def _reason(status: str, outcome: str) -> str:
    if status == ROLLBACK_STATUS_NO_ACTION_NEEDED:
        return f"Ledger outcome {outcome or 'unknown'} does not require rollback."
    if status == ROLLBACK_STATUS_SNAPSHOT_REQUIRED:
        return "Allowed entry needs a pre-commit evidence snapshot before rollback can be ready."
    return "Allowed entry has a pre-commit evidence snapshot and can be manually rolled back."


def _snapshot_for_entry(
    entry: Mapping[str, Any],
    pre_commit_snapshots: Mapping[str, Any],
    evidence_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    keys = (
        _str(entry.get("id")),
        _str(entry.get("candidate_id")),
        _str(entry.get("preview_id")),
        _str(entry.get("decision_id")),
        _str(entry.get("patch_digest")),
    )
    for key in keys:
        if key and key in pre_commit_snapshots:
            return _snapshot_evidence(pre_commit_snapshots.get(key))
    if evidence_fields and any(field in pre_commit_snapshots for field in evidence_fields):
        return {field: pre_commit_snapshots.get(field) for field in evidence_fields}
    return None


def _snapshot_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    evidence = value.get("evidence")
    if isinstance(evidence, Mapping):
        return dict(evidence)
    return dict(value)


def _evidence_fields(entry: Mapping[str, Any]) -> tuple[str, ...]:
    fields = entry.get("evidence_fields")
    if isinstance(fields, list):
        parsed = tuple(_str(field) for field in fields if _str(field))
        if parsed:
            return parsed
    source_decision = entry.get("source_decision")
    if isinstance(source_decision, Mapping):
        evidence_patch = source_decision.get("evidence_patch")
        if isinstance(evidence_patch, Mapping):
            return tuple(str(field) for field in evidence_patch.keys())
    return ()


def _stable_step_id(*, entry_id: str, patch_digest: str) -> str:
    raw = json.dumps(
        {"entry_id": entry_id, "patch_digest": patch_digest},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"rollback-{digest}"


def _dedupe_steps(
    steps: list[MemoryEvidenceRepairRollbackStep],
) -> list[MemoryEvidenceRepairRollbackStep]:
    seen: set[str] = set()
    deduped: list[MemoryEvidenceRepairRollbackStep] = []
    for step in steps:
        if step.id in seen:
            continue
        seen.add(step.id)
        deduped.append(step)
    return deduped


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _str(value: Any) -> str:
    return str(value or "").strip()
