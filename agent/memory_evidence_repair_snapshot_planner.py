"""Read-only pre-commit snapshot planner for memory evidence repair.

The snapshot planner identifies which existing evidence fields must be captured
before a future manual repair write. It never writes to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_rollback_plan import (
    ROLLBACK_ACTION_NOOP,
    ROLLBACK_STATUS_NO_ACTION_NEEDED,
    ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK,
    ROLLBACK_STATUS_SNAPSHOT_REQUIRED,
    build_evidence_repair_rollback_plan,
)


SNAPSHOT_ACTION_CAPTURE_EVIDENCE_METADATA = "capture_evidence_metadata"
SNAPSHOT_ACTION_VERIFY_EXISTING_SNAPSHOT = "verify_existing_snapshot"
SNAPSHOT_ACTION_NOOP = "no_snapshot_required"
SNAPSHOT_STATUS_CAPTURE_REQUIRED = "capture_required"
SNAPSHOT_STATUS_ALREADY_AVAILABLE = "already_available"
SNAPSHOT_STATUS_NO_ACTION_NEEDED = "no_action_needed"


@dataclass(frozen=True)
class MemoryEvidenceRepairSnapshotRequest:
    """One read-only pre-commit snapshot request."""

    id: str
    sequence: int
    status: str
    snapshot_action: str
    ledger_entry_id: str
    rollback_step_id: str
    candidate_id: str
    preview_id: str
    provider: str
    repair_action: str
    patch_digest: str
    target: dict[str, Any]
    evidence_fields: tuple[str, ...]
    snapshot_key_candidates: tuple[str, ...]
    snapshot_payload_preview: dict[str, Any]
    required_before: str
    blocks_manual_commit: bool
    reason: str
    verification_steps: tuple[str, ...]
    safety_note: str
    source_step: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "status": self.status,
            "snapshot_action": self.snapshot_action,
            "ledger_entry_id": self.ledger_entry_id,
            "rollback_step_id": self.rollback_step_id,
            "candidate_id": self.candidate_id,
            "preview_id": self.preview_id,
            "provider": self.provider,
            "repair_action": self.repair_action,
            "patch_digest": self.patch_digest,
            "target": dict(self.target),
            "evidence_fields": list(self.evidence_fields),
            "snapshot_key_candidates": list(self.snapshot_key_candidates),
            "snapshot_payload_preview": dict(self.snapshot_payload_preview),
            "required_before": self.required_before,
            "blocks_manual_commit": self.blocks_manual_commit,
            "reason": self.reason,
            "verification_steps": list(self.verification_steps),
            "safety_note": self.safety_note,
            "source_step": dict(self.source_step),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairSnapshotPlan:
    """Read-only pre-commit snapshot plan."""

    generated_at: str
    requests: tuple[MemoryEvidenceRepairSnapshotRequest, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        by_repair_action: dict[str, int] = {}
        capture_required_count = 0
        already_available_count = 0
        no_action_count = 0
        blocking_count = 0
        for request in self.requests:
            by_status[request.status] = by_status.get(request.status, 0) + 1
            by_repair_action[request.repair_action] = (
                by_repair_action.get(request.repair_action, 0) + 1
            )
            if request.provider:
                by_provider[request.provider] = by_provider.get(request.provider, 0) + 1
            if request.status == SNAPSHOT_STATUS_CAPTURE_REQUIRED:
                capture_required_count += 1
            elif request.status == SNAPSHOT_STATUS_ALREADY_AVAILABLE:
                already_available_count += 1
            elif request.status == SNAPSHOT_STATUS_NO_ACTION_NEEDED:
                no_action_count += 1
            if request.blocks_manual_commit:
                blocking_count += 1
        return {
            "snapshot_request_count": len(self.requests),
            "capture_required_count": capture_required_count,
            "already_available_count": already_available_count,
            "no_action_count": no_action_count,
            "blocking_count": blocking_count,
            "by_status": by_status,
            "by_provider": by_provider,
            "by_repair_action": by_repair_action,
            "requires_capture": capture_required_count > 0,
            "blocks_manual_commit": blocking_count > 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "requests": [request.to_dict() for request in self.requests],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_snapshot_plan(
    *,
    rollback_plan: Mapping[str, Any] | None = None,
    steps: Sequence[Mapping[str, Any]] | None = None,
    ledger: Mapping[str, Any] | None = None,
    entries: Sequence[Mapping[str, Any]] | None = None,
    existing_snapshots: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairSnapshotPlan:
    """Build a read-only pre-commit snapshot plan."""

    rollback_plan = rollback_plan if isinstance(rollback_plan, Mapping) else {}
    existing_snapshots = (
        existing_snapshots if isinstance(existing_snapshots, Mapping) else {}
    )
    rows = _step_rows(rollback_plan=rollback_plan, steps=steps)
    if not rows and (ledger or entries is not None):
        rows = build_evidence_repair_rollback_plan(
            ledger=ledger if isinstance(ledger, Mapping) else {},
            entries=entries,
            pre_commit_snapshots=existing_snapshots,
        ).to_dict().get("steps", [])

    requests: list[MemoryEvidenceRepairSnapshotRequest] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        request = _request_from_step(row, sequence=index)
        if request is not None:
            requests.append(request)

    return MemoryEvidenceRepairSnapshotPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        requests=tuple(_dedupe_requests(requests)),
    )


def empty_evidence_repair_snapshot_plan() -> MemoryEvidenceRepairSnapshotPlan:
    """Return an empty read-only pre-commit snapshot plan."""

    return MemoryEvidenceRepairSnapshotPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        requests=(),
    )


def _request_from_step(
    step: Mapping[str, Any],
    *,
    sequence: int,
) -> MemoryEvidenceRepairSnapshotRequest | None:
    step_id = _str(step.get("id"))
    ledger_entry_id = _str(step.get("ledger_entry_id"))
    candidate_id = _str(step.get("candidate_id"))
    preview_id = _str(step.get("preview_id"))
    status = _str(step.get("status"))
    if not step_id and not ledger_entry_id and not candidate_id and not status:
        return None

    snapshot_status = _snapshot_status(step)
    evidence_fields = _evidence_fields(step)
    key_candidates = _snapshot_key_candidates(step)
    request_id = _stable_request_id(
        ledger_entry_id=ledger_entry_id,
        rollback_step_id=step_id,
        patch_digest=_str(step.get("patch_digest")),
        evidence_fields=evidence_fields,
    )
    return MemoryEvidenceRepairSnapshotRequest(
        id=request_id,
        sequence=sequence,
        status=snapshot_status,
        snapshot_action=_snapshot_action(snapshot_status),
        ledger_entry_id=ledger_entry_id,
        rollback_step_id=step_id,
        candidate_id=candidate_id,
        preview_id=preview_id,
        provider=_str(step.get("provider")) or "memory",
        repair_action=_str(step.get("repair_action")) or "attach_provenance",
        patch_digest=_str(step.get("patch_digest")),
        target=_dict(step.get("target")),
        evidence_fields=evidence_fields,
        snapshot_key_candidates=key_candidates,
        snapshot_payload_preview=_snapshot_payload_preview(
            step=step,
            snapshot_status=snapshot_status,
            evidence_fields=evidence_fields,
            key_candidates=key_candidates,
        ),
        required_before="manual_commit",
        blocks_manual_commit=snapshot_status == SNAPSHOT_STATUS_CAPTURE_REQUIRED,
        reason=_reason(snapshot_status),
        verification_steps=_verification_steps(snapshot_status),
        safety_note=(
            "Read-only snapshot plan. Capture requests describe evidence that "
            "must be saved before a future manual write; this plan writes nothing."
        ),
        source_step=dict(step),
    )


def _snapshot_status(step: Mapping[str, Any]) -> str:
    rollback_status = _str(step.get("status"))
    rollback_action = _str(step.get("rollback_action"))
    if rollback_status == ROLLBACK_STATUS_NO_ACTION_NEEDED or rollback_action == ROLLBACK_ACTION_NOOP:
        return SNAPSHOT_STATUS_NO_ACTION_NEEDED
    if rollback_status == ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK:
        return SNAPSHOT_STATUS_ALREADY_AVAILABLE
    if rollback_status == ROLLBACK_STATUS_SNAPSHOT_REQUIRED:
        return SNAPSHOT_STATUS_CAPTURE_REQUIRED
    return SNAPSHOT_STATUS_CAPTURE_REQUIRED


def _snapshot_action(snapshot_status: str) -> str:
    if snapshot_status == SNAPSHOT_STATUS_CAPTURE_REQUIRED:
        return SNAPSHOT_ACTION_CAPTURE_EVIDENCE_METADATA
    if snapshot_status == SNAPSHOT_STATUS_ALREADY_AVAILABLE:
        return SNAPSHOT_ACTION_VERIFY_EXISTING_SNAPSHOT
    return SNAPSHOT_ACTION_NOOP


def _snapshot_payload_preview(
    *,
    step: Mapping[str, Any],
    snapshot_status: str,
    evidence_fields: tuple[str, ...],
    key_candidates: tuple[str, ...],
) -> dict[str, Any]:
    if snapshot_status == SNAPSHOT_STATUS_NO_ACTION_NEEDED:
        return {
            "operation": SNAPSHOT_ACTION_NOOP,
            "reason": "Rollback step does not correspond to an allowed manual commit.",
        }
    return {
        "operation": SNAPSHOT_ACTION_CAPTURE_EVIDENCE_METADATA,
        "target": _dict(step.get("target")),
        "capture": {
            "evidence_fields": list(evidence_fields),
            "snapshot_key": key_candidates[0] if key_candidates else "",
            "snapshot_key_candidates": list(key_candidates),
            "format": "pre_commit_evidence_snapshot",
        },
        "required_before": "manual_commit",
        "source_rollback_step_id": _str(step.get("id")),
    }


def _reason(snapshot_status: str) -> str:
    if snapshot_status == SNAPSHOT_STATUS_CAPTURE_REQUIRED:
        return "Capture current evidence metadata before applying this manual repair."
    if snapshot_status == SNAPSHOT_STATUS_ALREADY_AVAILABLE:
        return "A pre-commit snapshot is already available; verify it before commit."
    return "No snapshot is required because this entry is not eligible for manual commit."


def _verification_steps(snapshot_status: str) -> tuple[str, ...]:
    if snapshot_status == SNAPSHOT_STATUS_NO_ACTION_NEEDED:
        return ("verify_no_manual_commit_is_planned",)
    steps = [
        "verify_target_matches_ledger_entry",
        "read_current_evidence_metadata",
        "verify_snapshot_contains_requested_fields",
    ]
    if snapshot_status == SNAPSHOT_STATUS_CAPTURE_REQUIRED:
        steps.append("store_snapshot_before_manual_commit")
    else:
        steps.append("verify_existing_snapshot_before_manual_commit")
    return tuple(steps)


def _step_rows(
    *,
    rollback_plan: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]] | None,
) -> list[Any]:
    if steps is not None:
        return list(steps)
    if "rollback_action" in rollback_plan:
        return [rollback_plan]
    value = rollback_plan.get("steps")
    return value if isinstance(value, list) else []


def _evidence_fields(step: Mapping[str, Any]) -> tuple[str, ...]:
    fields = step.get("evidence_fields")
    if isinstance(fields, list):
        parsed = tuple(_str(field) for field in fields if _str(field))
        if parsed:
            return parsed
    source_entry = step.get("source_entry")
    if isinstance(source_entry, Mapping):
        fields = source_entry.get("evidence_fields")
        if isinstance(fields, list):
            return tuple(_str(field) for field in fields if _str(field))
    return ()


def _snapshot_key_candidates(step: Mapping[str, Any]) -> tuple[str, ...]:
    keys = (
        _str(step.get("ledger_entry_id")),
        _str(step.get("candidate_id")),
        _str(step.get("preview_id")),
        _str(step.get("patch_digest")),
        _str(step.get("id")),
    )
    return tuple(key for key in keys if key)


def _stable_request_id(
    *,
    ledger_entry_id: str,
    rollback_step_id: str,
    patch_digest: str,
    evidence_fields: tuple[str, ...],
) -> str:
    raw = json.dumps(
        {
            "ledger_entry_id": ledger_entry_id,
            "rollback_step_id": rollback_step_id,
            "patch_digest": patch_digest,
            "evidence_fields": list(evidence_fields),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"snapshot-{digest}"


def _dedupe_requests(
    requests: list[MemoryEvidenceRepairSnapshotRequest],
) -> list[MemoryEvidenceRepairSnapshotRequest]:
    seen: set[str] = set()
    deduped: list[MemoryEvidenceRepairSnapshotRequest] = []
    for request in requests:
        if request.id in seen:
            continue
        seen.add(request.id)
        deduped.append(request)
    return deduped


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _str(value: Any) -> str:
    return str(value or "").strip()
