"""Read-only commit gate for memory evidence repair previews.

The gate decides whether a repair preview is ready for a later manual write.
It never writes to durable memory stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_apply_preview import (
    PATCH_OPERATION,
    PREVIEW_STATUS_READY_FOR_MANUAL_APPLY,
)


DECISION_ALLOW_MANUAL_COMMIT = "allow_manual_commit"
DECISION_BLOCK_COMMIT = "block_commit"
DECISION_NEEDS_USER_CONFIRMATION = "needs_user_confirmation"

REASON_MISSING_EVIDENCE = "missing_evidence"
REASON_MISSING_CONFIRMATION = "missing_user_confirmation"
REASON_PREVIEW_NOT_READY = "preview_not_ready"
REASON_UNSAFE_OPERATION = "unsafe_operation"
REASON_PATCH_SHAPE_INVALID = "patch_shape_invalid"
REASON_CONFLICT_REQUIRES_REVIEW = "conflict_requires_review"


@dataclass(frozen=True)
class MemoryEvidenceRepairCommitGateDecision:
    """One read-only commit gate decision for a repair preview."""

    id: str
    preview_id: str
    candidate_id: str
    decision: str
    provider: str
    priority: str
    repair_action: str
    reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    target: dict[str, Any]
    evidence_patch: dict[str, Any]
    memory_patch_preview: dict[str, Any]
    conflict_risk: str
    safety_note: str
    source_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "preview_id": self.preview_id,
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "provider": self.provider,
            "priority": self.priority,
            "repair_action": self.repair_action,
            "reasons": list(self.reasons),
            "required_actions": list(self.required_actions),
            "target": dict(self.target),
            "evidence_patch": dict(self.evidence_patch),
            "memory_patch_preview": dict(self.memory_patch_preview),
            "conflict_risk": self.conflict_risk,
            "safety_note": self.safety_note,
            "source_preview": dict(self.source_preview),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairCommitGateReport:
    """Read-only commit gate report."""

    generated_at: str
    decisions: tuple[MemoryEvidenceRepairCommitGateDecision, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_decision: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        by_repair_action: dict[str, int] = {}
        allow_count = 0
        blocked_count = 0
        needs_confirmation_count = 0
        for decision in self.decisions:
            by_decision[decision.decision] = by_decision.get(decision.decision, 0) + 1
            by_repair_action[decision.repair_action] = (
                by_repair_action.get(decision.repair_action, 0) + 1
            )
            if decision.provider:
                by_provider[decision.provider] = by_provider.get(decision.provider, 0) + 1
            if decision.decision == DECISION_ALLOW_MANUAL_COMMIT:
                allow_count += 1
            elif decision.decision == DECISION_NEEDS_USER_CONFIRMATION:
                needs_confirmation_count += 1
            elif decision.decision == DECISION_BLOCK_COMMIT:
                blocked_count += 1
        return {
            "decision_count": len(self.decisions),
            "allow_count": allow_count,
            "blocked_count": blocked_count,
            "needs_confirmation_count": needs_confirmation_count,
            "by_decision": by_decision,
            "by_provider": by_provider,
            "by_repair_action": by_repair_action,
            "has_allowances": allow_count > 0,
            "has_blocks": blocked_count > 0,
            "requires_user_confirmation": needs_confirmation_count > 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_commit_gate(
    *,
    preview: Mapping[str, Any] | None = None,
    previews: Sequence[Mapping[str, Any]] | None = None,
    confirmed_preview_ids: Sequence[str] | None = None,
    user_confirmed: bool = False,
) -> MemoryEvidenceRepairCommitGateReport:
    """Build read-only commit gate decisions from apply previews."""

    preview = preview if isinstance(preview, Mapping) else {}
    preview_rows = _preview_rows(preview=preview, previews=previews)
    confirmed_ids = _string_set(confirmed_preview_ids)
    decisions: list[MemoryEvidenceRepairCommitGateDecision] = []
    for row in preview_rows:
        if not isinstance(row, Mapping):
            continue
        decision = _decision_from_preview(
            row,
            confirmed_preview_ids=confirmed_ids,
            user_confirmed=user_confirmed,
        )
        if decision is not None:
            decisions.append(decision)

    return MemoryEvidenceRepairCommitGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        decisions=tuple(_dedupe_decisions(decisions)),
    )


def empty_evidence_repair_commit_gate() -> MemoryEvidenceRepairCommitGateReport:
    """Return an empty read-only commit gate report."""

    return MemoryEvidenceRepairCommitGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        decisions=(),
    )


def _decision_from_preview(
    preview: Mapping[str, Any],
    *,
    confirmed_preview_ids: set[str],
    user_confirmed: bool,
) -> MemoryEvidenceRepairCommitGateDecision | None:
    preview_id = _str(preview.get("id"))
    candidate_id = _str(preview.get("candidate_id"))
    repair_action = _str(preview.get("repair_action"))
    if not preview_id and not candidate_id and not repair_action:
        return None

    provider = _str(preview.get("provider")) or "memory"
    priority = _str(preview.get("priority")) or "medium"
    target = preview.get("target") if isinstance(preview.get("target"), Mapping) else {}
    evidence_patch = (
        preview.get("evidence_patch")
        if isinstance(preview.get("evidence_patch"), Mapping)
        else {}
    )
    memory_patch_preview = (
        preview.get("memory_patch_preview")
        if isinstance(preview.get("memory_patch_preview"), Mapping)
        else {}
    )
    reasons, required_actions = _evaluate_reasons(
        preview=preview,
        preview_id=preview_id,
        confirmed_preview_ids=confirmed_preview_ids,
        user_confirmed=user_confirmed,
        evidence_patch=evidence_patch,
        memory_patch_preview=memory_patch_preview,
    )
    decision = _decision_for_reasons(reasons)
    return MemoryEvidenceRepairCommitGateDecision(
        id=f"gate-{preview_id or candidate_id}",
        preview_id=preview_id,
        candidate_id=candidate_id,
        decision=decision,
        provider=provider,
        priority=priority,
        repair_action=repair_action or "attach_provenance",
        reasons=tuple(reasons),
        required_actions=tuple(required_actions),
        target=dict(target),
        evidence_patch=dict(evidence_patch),
        memory_patch_preview=dict(memory_patch_preview),
        conflict_risk=_conflict_risk(preview),
        safety_note=(
            "Read-only gate. An allow decision means this preview is eligible "
            "for a later manual write; this gate never writes memory."
        ),
        source_preview=dict(preview),
    )


def _evaluate_reasons(
    *,
    preview: Mapping[str, Any],
    preview_id: str,
    confirmed_preview_ids: set[str],
    user_confirmed: bool,
    evidence_patch: Mapping[str, Any],
    memory_patch_preview: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    required_actions: list[str] = []

    missing_evidence = _list_of_str(preview.get("missing_evidence"))
    if missing_evidence:
        reasons.append(REASON_MISSING_EVIDENCE)
        required_actions.append("complete_required_evidence")
    if _str(preview.get("status")) != PREVIEW_STATUS_READY_FOR_MANUAL_APPLY:
        reasons.append(REASON_PREVIEW_NOT_READY)
        required_actions.append("regenerate_ready_apply_preview")
    if not _is_preview_confirmed(
        preview_id=preview_id,
        confirmed_preview_ids=confirmed_preview_ids,
        user_confirmed=user_confirmed,
    ):
        reasons.append(REASON_MISSING_CONFIRMATION)
        required_actions.append("obtain_explicit_user_confirmation")
    if _str(preview.get("operation")) != PATCH_OPERATION:
        reasons.append(REASON_UNSAFE_OPERATION)
        required_actions.append("use_evidence_metadata_merge_operation")
    if _str(memory_patch_preview.get("operation")) != PATCH_OPERATION:
        reasons.append(REASON_PATCH_SHAPE_INVALID)
        required_actions.append("repair_memory_patch_preview_shape")
    if not evidence_patch:
        reasons.append(REASON_PATCH_SHAPE_INVALID)
        required_actions.append("provide_non_empty_evidence_patch")
    if _conflict_risk(preview) == "high":
        evidence_resolution = _str(evidence_patch.get("conflict_resolution"))
        if not evidence_resolution:
            reasons.append(REASON_CONFLICT_REQUIRES_REVIEW)
            required_actions.append("provide_conflict_resolution")

    return _dedupe_strings(reasons), _dedupe_strings(required_actions)


def _decision_for_reasons(reasons: list[str]) -> str:
    if not reasons:
        return DECISION_ALLOW_MANUAL_COMMIT
    if reasons == [REASON_MISSING_CONFIRMATION]:
        return DECISION_NEEDS_USER_CONFIRMATION
    if (
        REASON_MISSING_CONFIRMATION in reasons
        and all(reason == REASON_MISSING_CONFIRMATION for reason in reasons)
    ):
        return DECISION_NEEDS_USER_CONFIRMATION
    return DECISION_BLOCK_COMMIT


def _is_preview_confirmed(
    *,
    preview_id: str,
    confirmed_preview_ids: set[str],
    user_confirmed: bool,
) -> bool:
    if confirmed_preview_ids:
        return preview_id in confirmed_preview_ids
    return bool(user_confirmed)


def _conflict_risk(preview: Mapping[str, Any]) -> str:
    repair_action = _str(preview.get("repair_action"))
    if repair_action == "review_conflict":
        return "high"
    source_candidate = preview.get("source_candidate")
    if isinstance(source_candidate, Mapping):
        source_repair = source_candidate.get("source_repair")
        if isinstance(source_repair, Mapping):
            source = f"{source_repair.get('action', '')} {source_repair.get('reason', '')}"
            if "conflict" in source.lower():
                return "high"
        reason = _str(source_candidate.get("reason"))
        if "conflict" in reason.lower():
            return "medium"
    reason = _str(preview.get("reason"))
    if "conflict" in reason.lower():
        return "medium"
    return "low"


def _preview_rows(
    *,
    preview: Mapping[str, Any],
    previews: Sequence[Mapping[str, Any]] | None,
) -> list[Any]:
    if previews is not None:
        return list(previews)
    if "id" in preview:
        return [preview]
    value = preview.get("previews")
    return value if isinstance(value, list) else []


def _dedupe_decisions(
    decisions: list[MemoryEvidenceRepairCommitGateDecision],
) -> list[MemoryEvidenceRepairCommitGateDecision]:
    seen: set[str] = set()
    deduped: list[MemoryEvidenceRepairCommitGateDecision] = []
    for decision in decisions:
        if decision.id in seen:
            continue
        seen.add(decision.id)
        deduped.append(decision)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _string_set(value: Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    return {_str(item) for item in value if _str(item)}


def _list_of_str(value: Any) -> list[str]:
    return [_str(item) for item in value] if isinstance(value, list) else []


def _str(value: Any) -> str:
    return str(value or "").strip()
