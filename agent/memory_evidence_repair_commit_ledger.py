"""Read-only commit ledger draft for memory evidence repair decisions.

The ledger turns commit gate decisions into audit-ready records. It never
writes to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_commit_gate import (
    DECISION_ALLOW_MANUAL_COMMIT,
    DECISION_BLOCK_COMMIT,
    DECISION_NEEDS_USER_CONFIRMATION,
)


LEDGER_EVENT_TYPE = "memory_evidence_repair_commit_gate_decision"
LEDGER_STATUS_DRAFT = "draft"
OUTCOME_ALLOWED = "allowed_for_manual_commit"
OUTCOME_BLOCKED = "blocked"
OUTCOME_NEEDS_CONFIRMATION = "needs_user_confirmation"


@dataclass(frozen=True)
class MemoryEvidenceRepairCommitLedgerEntry:
    """One read-only audit ledger entry draft."""

    id: str
    sequence: int
    event_type: str
    status: str
    outcome: str
    decision_id: str
    preview_id: str
    candidate_id: str
    provider: str
    priority: str
    repair_action: str
    decision: str
    reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    evidence_fields: tuple[str, ...]
    patch_digest: str
    manual_commit_allowed: bool
    blocked: bool
    requires_followup: bool
    target: dict[str, Any]
    metadata: dict[str, Any]
    safety_note: str
    source_decision: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "status": self.status,
            "outcome": self.outcome,
            "decision_id": self.decision_id,
            "preview_id": self.preview_id,
            "candidate_id": self.candidate_id,
            "provider": self.provider,
            "priority": self.priority,
            "repair_action": self.repair_action,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "required_actions": list(self.required_actions),
            "evidence_fields": list(self.evidence_fields),
            "patch_digest": self.patch_digest,
            "manual_commit_allowed": self.manual_commit_allowed,
            "blocked": self.blocked,
            "requires_followup": self.requires_followup,
            "target": dict(self.target),
            "metadata": dict(self.metadata),
            "safety_note": self.safety_note,
            "source_decision": dict(self.source_decision),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairCommitLedgerDraft:
    """Read-only commit ledger draft."""

    generated_at: str
    entries: tuple[MemoryEvidenceRepairCommitLedgerEntry, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_decision: dict[str, int] = {}
        by_outcome: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        by_repair_action: dict[str, int] = {}
        allow_count = 0
        blocked_count = 0
        followup_count = 0
        for entry in self.entries:
            by_decision[entry.decision] = by_decision.get(entry.decision, 0) + 1
            by_outcome[entry.outcome] = by_outcome.get(entry.outcome, 0) + 1
            by_repair_action[entry.repair_action] = (
                by_repair_action.get(entry.repair_action, 0) + 1
            )
            if entry.provider:
                by_provider[entry.provider] = by_provider.get(entry.provider, 0) + 1
            if entry.manual_commit_allowed:
                allow_count += 1
            if entry.blocked:
                blocked_count += 1
            if entry.requires_followup:
                followup_count += 1
        return {
            "entry_count": len(self.entries),
            "allow_count": allow_count,
            "blocked_count": blocked_count,
            "followup_count": followup_count,
            "by_decision": by_decision,
            "by_outcome": by_outcome,
            "by_provider": by_provider,
            "by_repair_action": by_repair_action,
            "has_allowances": allow_count > 0,
            "has_blocks": blocked_count > 0,
            "requires_followup": followup_count > 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "entries": [entry.to_dict() for entry in self.entries],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_commit_ledger(
    *,
    commit_gate: Mapping[str, Any] | None = None,
    decisions: Sequence[Mapping[str, Any]] | None = None,
    ledger_actor: str = "hermes",
    ledger_reason: str = "",
) -> MemoryEvidenceRepairCommitLedgerDraft:
    """Build a read-only audit ledger draft from commit gate decisions."""

    commit_gate = commit_gate if isinstance(commit_gate, Mapping) else {}
    rows = _decision_rows(commit_gate=commit_gate, decisions=decisions)
    entries: list[MemoryEvidenceRepairCommitLedgerEntry] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        entry = _entry_from_decision(
            row,
            sequence=index,
            ledger_actor=ledger_actor,
            ledger_reason=ledger_reason,
        )
        if entry is not None:
            entries.append(entry)

    return MemoryEvidenceRepairCommitLedgerDraft(
        generated_at=datetime.now(timezone.utc).isoformat(),
        entries=tuple(_dedupe_entries(entries)),
    )


def empty_evidence_repair_commit_ledger() -> MemoryEvidenceRepairCommitLedgerDraft:
    """Return an empty read-only commit ledger draft."""

    return MemoryEvidenceRepairCommitLedgerDraft(
        generated_at=datetime.now(timezone.utc).isoformat(),
        entries=(),
    )


def _entry_from_decision(
    decision: Mapping[str, Any],
    *,
    sequence: int,
    ledger_actor: str,
    ledger_reason: str,
) -> MemoryEvidenceRepairCommitLedgerEntry | None:
    decision_id = _str(decision.get("id"))
    preview_id = _str(decision.get("preview_id"))
    candidate_id = _str(decision.get("candidate_id"))
    decision_value = _str(decision.get("decision"))
    if not decision_id and not preview_id and not candidate_id and not decision_value:
        return None

    evidence_patch = (
        decision.get("evidence_patch")
        if isinstance(decision.get("evidence_patch"), Mapping)
        else {}
    )
    memory_patch_preview = (
        decision.get("memory_patch_preview")
        if isinstance(decision.get("memory_patch_preview"), Mapping)
        else {}
    )
    reasons = tuple(_list_of_str(decision.get("reasons")))
    required_actions = tuple(_list_of_str(decision.get("required_actions")))
    outcome = _outcome(decision_value)
    patch_digest = _patch_digest(
        evidence_patch=evidence_patch,
        memory_patch_preview=memory_patch_preview,
    )
    entry_id = _stable_entry_id(
        decision_id=decision_id,
        preview_id=preview_id,
        candidate_id=candidate_id,
        decision=decision_value,
        patch_digest=patch_digest,
        reasons=reasons,
    )
    return MemoryEvidenceRepairCommitLedgerEntry(
        id=entry_id,
        sequence=sequence,
        event_type=LEDGER_EVENT_TYPE,
        status=LEDGER_STATUS_DRAFT,
        outcome=outcome,
        decision_id=decision_id,
        preview_id=preview_id,
        candidate_id=candidate_id,
        provider=_str(decision.get("provider")) or "memory",
        priority=_str(decision.get("priority")) or "medium",
        repair_action=_str(decision.get("repair_action")) or "attach_provenance",
        decision=decision_value,
        reasons=reasons,
        required_actions=required_actions,
        evidence_fields=tuple(str(field) for field in evidence_patch.keys()),
        patch_digest=patch_digest,
        manual_commit_allowed=decision_value == DECISION_ALLOW_MANUAL_COMMIT,
        blocked=decision_value == DECISION_BLOCK_COMMIT,
        requires_followup=bool(required_actions)
        or decision_value == DECISION_NEEDS_USER_CONFIRMATION,
        target=_dict(decision.get("target")),
        metadata={
            "ledger_actor": _str(ledger_actor) or "hermes",
            "ledger_reason": _str(ledger_reason),
            "conflict_risk": _str(decision.get("conflict_risk")) or "unknown",
        },
        safety_note=(
            "Read-only ledger draft. This record is for audit and planning only; "
            "it does not commit or mutate durable memory."
        ),
        source_decision=dict(decision),
    )


def _decision_rows(
    *,
    commit_gate: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]] | None,
) -> list[Any]:
    if decisions is not None:
        return list(decisions)
    if "decision" in commit_gate:
        return [commit_gate]
    value = commit_gate.get("decisions")
    return value if isinstance(value, list) else []


def _outcome(decision: str) -> str:
    if decision == DECISION_ALLOW_MANUAL_COMMIT:
        return OUTCOME_ALLOWED
    if decision == DECISION_NEEDS_USER_CONFIRMATION:
        return OUTCOME_NEEDS_CONFIRMATION
    return OUTCOME_BLOCKED


def _patch_digest(
    *,
    evidence_patch: Mapping[str, Any],
    memory_patch_preview: Mapping[str, Any],
) -> str:
    raw = json.dumps(
        {
            "evidence_patch": dict(evidence_patch),
            "memory_patch_preview": dict(memory_patch_preview),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_entry_id(
    *,
    decision_id: str,
    preview_id: str,
    candidate_id: str,
    decision: str,
    patch_digest: str,
    reasons: tuple[str, ...],
) -> str:
    raw = json.dumps(
        {
            "decision_id": decision_id,
            "preview_id": preview_id,
            "candidate_id": candidate_id,
            "decision": decision,
            "patch_digest": patch_digest,
            "reasons": list(reasons),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"ledger-{digest}"


def _dedupe_entries(
    entries: list[MemoryEvidenceRepairCommitLedgerEntry],
) -> list[MemoryEvidenceRepairCommitLedgerEntry]:
    seen: set[str] = set()
    deduped: list[MemoryEvidenceRepairCommitLedgerEntry] = []
    for entry in entries:
        if entry.id in seen:
            continue
        seen.add(entry.id)
        deduped.append(entry)
    return deduped


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_str(value: Any) -> list[str]:
    return [_str(item) for item in value] if isinstance(value, list) else []


def _str(value: Any) -> str:
    return str(value or "").strip()
