"""Read-only approval candidates for memory evidence repair.

This layer converts an evidence repair plan into human-confirmable candidate
records. It never writes to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


CANDIDATE_TYPE = "memory_evidence_repair"
STATUS_NEEDS_CONFIRMATION = "needs_user_confirmation"
PENDING_VALUE = "<pending>"


@dataclass(frozen=True)
class MemoryEvidenceRepairApprovalCandidate:
    """One read-only repair candidate that needs user confirmation."""

    id: str
    candidate_type: str
    status: str
    provider: str
    priority: str
    repair_action: str
    source: str
    reason: str
    content_preview: str
    required_evidence: tuple[str, ...]
    proposed_evidence_patch: dict[str, Any]
    confirmation_question: str
    safety_note: str
    source_repair: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate_type": self.candidate_type,
            "status": self.status,
            "provider": self.provider,
            "priority": self.priority,
            "repair_action": self.repair_action,
            "source": self.source,
            "reason": self.reason,
            "content_preview": self.content_preview,
            "required_evidence": list(self.required_evidence),
            "proposed_evidence_patch": dict(self.proposed_evidence_patch),
            "confirmation_question": self.confirmation_question,
            "safety_note": self.safety_note,
            "source_repair": dict(self.source_repair),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairApprovalReport:
    """Read-only approval candidate report."""

    generated_at: str
    candidates: tuple[MemoryEvidenceRepairApprovalCandidate, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_priority: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        by_repair_action: dict[str, int] = {}
        for candidate in self.candidates:
            by_priority[candidate.priority] = by_priority.get(candidate.priority, 0) + 1
            by_repair_action[candidate.repair_action] = (
                by_repair_action.get(candidate.repair_action, 0) + 1
            )
            if candidate.provider:
                by_provider[candidate.provider] = by_provider.get(candidate.provider, 0) + 1
        return {
            "candidate_count": len(self.candidates),
            "by_priority": by_priority,
            "by_provider": by_provider,
            "by_repair_action": by_repair_action,
            "requires_user_confirmation": bool(self.candidates),
            "has_candidates": bool(self.candidates),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "read_only": True,
            "read_only_memory": True,
        }


def build_evidence_repair_approval_candidates(
    *,
    plan: Mapping[str, Any] | None = None,
    proposed_evidence: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairApprovalReport:
    """Build read-only approval candidates from an evidence repair plan."""

    plan = plan if isinstance(plan, Mapping) else {}
    proposed_evidence = (
        proposed_evidence if isinstance(proposed_evidence, Mapping) else {}
    )
    candidates: list[MemoryEvidenceRepairApprovalCandidate] = []
    for repair in _list(plan.get("repairs")):
        if not isinstance(repair, Mapping):
            continue
        candidate = _candidate_from_repair(repair, proposed_evidence)
        if candidate is not None:
            candidates.append(candidate)

    return MemoryEvidenceRepairApprovalReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        candidates=tuple(_dedupe_candidates(candidates)),
    )


def empty_evidence_repair_approval_candidates() -> MemoryEvidenceRepairApprovalReport:
    """Return an empty read-only approval candidate report."""

    return MemoryEvidenceRepairApprovalReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        candidates=(),
    )


def _candidate_from_repair(
    repair: Mapping[str, Any],
    proposed_evidence: Mapping[str, Any],
) -> MemoryEvidenceRepairApprovalCandidate | None:
    action = _str(repair.get("action") or repair.get("repair_action"))
    reason = _str(repair.get("reason"))
    if not action and not reason:
        return None

    provider = _str(repair.get("provider")) or "memory"
    priority = _str(repair.get("priority")) or "medium"
    source = _str(repair.get("source")) or "memory_evidence_repair_plan"
    content_preview = _content_preview(repair)
    required_evidence = _required_evidence(repair.get("required_evidence"))
    candidate_id = _stable_candidate_id(
        provider=provider,
        action=action,
        source=source,
        reason=reason,
        content_preview=content_preview,
    )
    proposed_patch = {
        field: _proposed_value(field, proposed_evidence) for field in required_evidence
    }
    return MemoryEvidenceRepairApprovalCandidate(
        id=candidate_id,
        candidate_type=CANDIDATE_TYPE,
        status=STATUS_NEEDS_CONFIRMATION,
        provider=provider,
        priority=priority,
        repair_action=action,
        source=source,
        reason=reason or "Evidence repair needs user confirmation.",
        content_preview=content_preview,
        required_evidence=required_evidence,
        proposed_evidence_patch=proposed_patch,
        confirmation_question=_confirmation_question(action, required_evidence),
        safety_note=(
            "Read-only candidate. Do not write this repair to memory until the "
            "user explicitly confirms the evidence patch."
        ),
        source_repair=dict(repair),
    )


def _dedupe_candidates(
    candidates: list[MemoryEvidenceRepairApprovalCandidate],
) -> list[MemoryEvidenceRepairApprovalCandidate]:
    seen: set[str] = set()
    deduped: list[MemoryEvidenceRepairApprovalCandidate] = []
    for candidate in candidates:
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        deduped.append(candidate)
    return deduped


def _stable_candidate_id(
    *,
    provider: str,
    action: str,
    source: str,
    reason: str,
    content_preview: str,
) -> str:
    raw = json.dumps(
        {
            "provider": provider,
            "action": action,
            "source": source,
            "reason": reason,
            "content_preview": content_preview,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"repair-{digest}"


def _required_evidence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ("source_url", "observed_at", "verification_signal")
    fields = tuple(_str(item) for item in value if _str(item))
    return fields or ("source_url", "observed_at", "verification_signal")


def _proposed_value(field: str, proposed_evidence: Mapping[str, Any]) -> Any:
    if field in proposed_evidence and proposed_evidence[field] not in (None, ""):
        return proposed_evidence[field]
    return PENDING_VALUE


def _confirmation_question(action: str, required_evidence: tuple[str, ...]) -> str:
    evidence_list = ", ".join(required_evidence) or "the required evidence"
    if action == "refresh_observation":
        return f"Can you confirm the refreshed {evidence_list} for this memory?"
    if action == "review_conflict":
        return f"Can you confirm the conflict resolution and {evidence_list}?"
    return f"Can you confirm {evidence_list} for this memory evidence repair?"


def _content_preview(repair: Mapping[str, Any]) -> str:
    preview = _str(repair.get("content_preview"))
    if not preview:
        preview = _str(repair.get("evidence"))
    return preview[:280]


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str(value: Any) -> str:
    return str(value or "").strip()
