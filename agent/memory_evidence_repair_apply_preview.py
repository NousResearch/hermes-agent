"""Read-only apply previews for memory evidence repair candidates.

The preview layer shows the exact evidence metadata patch that would be applied
after a user confirms a repair candidate. It never writes to durable memory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


PATCH_OPERATION = "merge_evidence_metadata"
PREVIEW_STATUS_AWAITING_CONFIRMATION = "awaiting_user_confirmation"
PREVIEW_STATUS_READY_FOR_MANUAL_APPLY = "ready_for_manual_apply"
PREVIEW_STATUS_BLOCKED_MISSING_EVIDENCE = "blocked_missing_evidence"
PENDING_VALUE = "<pending>"


@dataclass(frozen=True)
class MemoryEvidenceRepairApplyPreview:
    """One read-only evidence repair patch preview."""

    id: str
    candidate_id: str
    status: str
    provider: str
    priority: str
    repair_action: str
    operation: str
    target: dict[str, Any]
    evidence_patch: dict[str, Any]
    memory_patch_preview: dict[str, Any]
    would_update_fields: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    preconditions: tuple[str, ...]
    content_preview: str
    source: str
    reason: str
    safety_note: str
    source_candidate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "provider": self.provider,
            "priority": self.priority,
            "repair_action": self.repair_action,
            "operation": self.operation,
            "target": dict(self.target),
            "evidence_patch": dict(self.evidence_patch),
            "memory_patch_preview": dict(self.memory_patch_preview),
            "would_update_fields": list(self.would_update_fields),
            "missing_evidence": list(self.missing_evidence),
            "preconditions": list(self.preconditions),
            "content_preview": self.content_preview,
            "source": self.source,
            "reason": self.reason,
            "safety_note": self.safety_note,
            "source_candidate": dict(self.source_candidate),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairApplyPreviewReport:
    """Read-only evidence repair apply preview report."""

    generated_at: str
    previews: tuple[MemoryEvidenceRepairApplyPreview, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        by_repair_action: dict[str, int] = {}
        missing_evidence_count = 0
        ready_count = 0
        awaiting_confirmation_count = 0
        for preview in self.previews:
            by_status[preview.status] = by_status.get(preview.status, 0) + 1
            by_repair_action[preview.repair_action] = (
                by_repair_action.get(preview.repair_action, 0) + 1
            )
            if preview.provider:
                by_provider[preview.provider] = by_provider.get(preview.provider, 0) + 1
            if preview.missing_evidence:
                missing_evidence_count += 1
            if preview.status == PREVIEW_STATUS_READY_FOR_MANUAL_APPLY:
                ready_count += 1
            if preview.status == PREVIEW_STATUS_AWAITING_CONFIRMATION:
                awaiting_confirmation_count += 1
        return {
            "preview_count": len(self.previews),
            "ready_count": ready_count,
            "awaiting_confirmation_count": awaiting_confirmation_count,
            "missing_evidence_count": missing_evidence_count,
            "by_status": by_status,
            "by_provider": by_provider,
            "by_repair_action": by_repair_action,
            "requires_user_confirmation": awaiting_confirmation_count > 0,
            "has_previews": bool(self.previews),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "previews": [preview.to_dict() for preview in self.previews],
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_apply_preview(
    *,
    approval: Mapping[str, Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    approved_candidate_ids: Sequence[str] | None = None,
    proposed_evidence: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairApplyPreviewReport:
    """Build read-only patch previews from evidence repair approval candidates."""

    approval = approval if isinstance(approval, Mapping) else {}
    proposed_evidence = (
        proposed_evidence if isinstance(proposed_evidence, Mapping) else {}
    )
    candidate_rows = _candidate_rows(approval=approval, candidates=candidates)
    approved_ids = _string_set(approved_candidate_ids)
    filter_to_approved = approved_candidate_ids is not None

    previews: list[MemoryEvidenceRepairApplyPreview] = []
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = _str(candidate.get("id"))
        if filter_to_approved and candidate_id not in approved_ids:
            continue
        preview = _preview_from_candidate(
            candidate,
            approved_ids=approved_ids,
            approval_filter_enabled=filter_to_approved,
            proposed_evidence=proposed_evidence,
        )
        if preview is not None:
            previews.append(preview)

    return MemoryEvidenceRepairApplyPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        previews=tuple(_dedupe_previews(previews)),
    )


def empty_evidence_repair_apply_preview() -> MemoryEvidenceRepairApplyPreviewReport:
    """Return an empty read-only apply preview report."""

    return MemoryEvidenceRepairApplyPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        previews=(),
    )


def _preview_from_candidate(
    candidate: Mapping[str, Any],
    *,
    approved_ids: set[str],
    approval_filter_enabled: bool,
    proposed_evidence: Mapping[str, Any],
) -> MemoryEvidenceRepairApplyPreview | None:
    candidate_id = _str(candidate.get("id"))
    repair_action = _str(candidate.get("repair_action") or candidate.get("action"))
    reason = _str(candidate.get("reason"))
    if not candidate_id and not repair_action and not reason:
        return None

    if not candidate_id:
        candidate_id = _stable_preview_source_id(candidate)
    provider = _str(candidate.get("provider")) or "memory"
    priority = _str(candidate.get("priority")) or "medium"
    source = _str(candidate.get("source")) or "memory_evidence_repair_approval"
    required_evidence = _required_evidence(candidate)
    evidence_patch = _evidence_patch(
        candidate_id=candidate_id,
        required_evidence=required_evidence,
        candidate=candidate,
        proposed_evidence=proposed_evidence,
    )
    missing_evidence = tuple(
        field for field in required_evidence if _is_missing(evidence_patch.get(field))
    )
    status = _preview_status(
        candidate_id=candidate_id,
        approved_ids=approved_ids,
        approval_filter_enabled=approval_filter_enabled,
        missing_evidence=missing_evidence,
    )
    target = {
        "candidate_id": candidate_id,
        "provider": provider,
        "source": source,
    }
    memory_patch = {
        "operation": PATCH_OPERATION,
        "target": target,
        "set": {
            "evidence": evidence_patch,
            "evidence_repair": {
                "candidate_id": candidate_id,
                "repair_action": repair_action,
                "reason": reason,
                "source": source,
            },
        },
    }
    return MemoryEvidenceRepairApplyPreview(
        id=_stable_preview_id(candidate_id, evidence_patch),
        candidate_id=candidate_id,
        status=status,
        provider=provider,
        priority=priority,
        repair_action=repair_action or "attach_provenance",
        operation=PATCH_OPERATION,
        target=target,
        evidence_patch=evidence_patch,
        memory_patch_preview=memory_patch,
        would_update_fields=tuple(evidence_patch.keys()),
        missing_evidence=missing_evidence,
        preconditions=_preconditions(status, missing_evidence),
        content_preview=_str(candidate.get("content_preview"))[:280],
        source=source,
        reason=reason or "Evidence repair apply preview needs review.",
        safety_note=(
            "Preview only. This patch must not be applied to durable memory "
            "until the user explicitly confirms it."
        ),
        source_candidate=dict(candidate),
    )


def _candidate_rows(
    *,
    approval: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]] | None,
) -> list[Any]:
    if candidates is not None:
        return list(candidates)
    value = approval.get("candidates")
    return value if isinstance(value, list) else []


def _evidence_patch(
    *,
    candidate_id: str,
    required_evidence: tuple[str, ...],
    candidate: Mapping[str, Any],
    proposed_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_patch = candidate.get("proposed_evidence_patch")
    candidate_patch = candidate_patch if isinstance(candidate_patch, Mapping) else {}
    scoped_patch = proposed_evidence.get(candidate_id)
    scoped_patch = scoped_patch if isinstance(scoped_patch, Mapping) else {}
    patch: dict[str, Any] = {}
    for field in required_evidence:
        if field in scoped_patch and not _is_missing(scoped_patch[field]):
            patch[field] = scoped_patch[field]
        elif field in proposed_evidence and not _is_missing(proposed_evidence[field]):
            patch[field] = proposed_evidence[field]
        elif field in candidate_patch:
            patch[field] = candidate_patch[field]
        else:
            patch[field] = PENDING_VALUE
    return patch


def _preview_status(
    *,
    candidate_id: str,
    approved_ids: set[str],
    approval_filter_enabled: bool,
    missing_evidence: tuple[str, ...],
) -> str:
    if missing_evidence:
        return PREVIEW_STATUS_BLOCKED_MISSING_EVIDENCE
    if approval_filter_enabled and candidate_id in approved_ids:
        return PREVIEW_STATUS_READY_FOR_MANUAL_APPLY
    return PREVIEW_STATUS_AWAITING_CONFIRMATION


def _preconditions(status: str, missing_evidence: tuple[str, ...]) -> tuple[str, ...]:
    preconditions = ["explicit_user_confirmation"]
    if missing_evidence:
        preconditions.append("complete_required_evidence")
    if status == PREVIEW_STATUS_READY_FOR_MANUAL_APPLY:
        preconditions.append("manual_apply_only")
    return tuple(preconditions)


def _required_evidence(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    required = candidate.get("required_evidence")
    if isinstance(required, (list, tuple)):
        fields = tuple(_str(item) for item in required if _str(item))
        if fields:
            return fields
    patch = candidate.get("proposed_evidence_patch")
    if isinstance(patch, Mapping) and patch:
        return tuple(_str(field) for field in patch.keys() if _str(field))
    return ("source_url", "observed_at", "verification_signal")


def _dedupe_previews(
    previews: list[MemoryEvidenceRepairApplyPreview],
) -> list[MemoryEvidenceRepairApplyPreview]:
    seen: set[str] = set()
    deduped: list[MemoryEvidenceRepairApplyPreview] = []
    for preview in previews:
        if preview.id in seen:
            continue
        seen.add(preview.id)
        deduped.append(preview)
    return deduped


def _stable_preview_id(candidate_id: str, evidence_patch: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"candidate_id": candidate_id, "evidence_patch": dict(evidence_patch)},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"preview-{digest}"


def _stable_preview_source_id(candidate: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(candidate), sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"repair-{digest}"


def _string_set(value: Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    return {_str(item) for item in value if _str(item)}


def _is_missing(value: Any) -> bool:
    return value in (None, "", PENDING_VALUE)


def _str(value: Any) -> str:
    return str(value or "").strip()
