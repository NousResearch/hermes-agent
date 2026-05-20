"""Read-only planner for repairing weak memory provenance.

The planner converts gate/audit/diagnostic/policy signals into repair
candidates. It never writes to durable memory stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


ACTION_ATTACH_PROVENANCE = "attach_provenance"
ACTION_VERIFY_SOURCE = "verify_source"
ACTION_REFRESH_OBSERVATION = "refresh_observation"
ACTION_REVIEW_CONFLICT = "review_conflict"

REQUIRE_PROVENANCE = "require_provenance_before_reuse"


@dataclass(frozen=True)
class MemoryEvidenceRepairItem:
    """One read-only evidence repair candidate."""

    provider: str
    priority: str
    action: str
    reason: str
    source: str
    evidence: str = ""
    content_preview: str = ""
    blocked_by_gate: bool = False
    required_evidence: tuple[str, ...] = ()
    recommended_next_step: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "priority": self.priority,
            "action": self.action,
            "source": self.source,
            "reason": self.reason,
            "evidence": self.evidence,
            "content_preview": self.content_preview,
            "blocked_by_gate": self.blocked_by_gate,
            "required_evidence": list(self.required_evidence),
            "recommended_next_step": self.recommended_next_step,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairPlan:
    """Read-only evidence repair plan."""

    generated_at: str
    repairs: tuple[MemoryEvidenceRepairItem, ...]

    @property
    def summary(self) -> dict[str, Any]:
        by_priority: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        by_action: dict[str, int] = {}
        blocked_count = 0
        for repair in self.repairs:
            by_priority[repair.priority] = by_priority.get(repair.priority, 0) + 1
            by_action[repair.action] = by_action.get(repair.action, 0) + 1
            if repair.provider:
                by_provider[repair.provider] = by_provider.get(repair.provider, 0) + 1
            if repair.blocked_by_gate:
                blocked_count += 1
        return {
            "repair_count": len(self.repairs),
            "blocked_count": blocked_count,
            "by_priority": by_priority,
            "by_provider": by_provider,
            "by_action": by_action,
            "has_repairs": bool(self.repairs),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "repairs": [repair.to_dict() for repair in self.repairs],
            "read_only": True,
        }


def build_evidence_repair_plan(
    *,
    gate: Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairPlan:
    """Build a read-only evidence repair plan from memory quality signals."""

    repairs: list[MemoryEvidenceRepairItem] = []
    gate = gate if isinstance(gate, Mapping) else {}
    audit = audit if isinstance(audit, Mapping) else {}
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    policy = policy if isinstance(policy, Mapping) else {}

    repairs.extend(_repairs_from_gate(gate))
    repairs.extend(_repairs_from_diagnostics(diagnostics))
    repairs.extend(_repairs_from_policy(policy))
    repairs.extend(_repairs_from_audit(audit))

    return MemoryEvidenceRepairPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        repairs=tuple(_dedupe_repairs(repairs)),
    )


def empty_evidence_repair_plan() -> MemoryEvidenceRepairPlan:
    """Return an empty read-only evidence repair plan."""

    return MemoryEvidenceRepairPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        repairs=(),
    )


def _repairs_from_gate(gate: Mapping[str, Any]) -> list[MemoryEvidenceRepairItem]:
    rows = _list(gate.get("provider_results")) or _list(gate.get("decisions"))
    repairs: list[MemoryEvidenceRepairItem] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        action = str(row.get("action", "") or "")
        effect = str(row.get("effect", "") or "")
        reason = str(row.get("reason", "") or "")
        if action != REQUIRE_PROVENANCE and "provenance" not in reason.lower():
            continue
        repairs.append(
            MemoryEvidenceRepairItem(
                provider=str(row.get("provider", "") or ""),
                priority="high" if effect == "filtered" else "medium",
                action=ACTION_ATTACH_PROVENANCE,
                source="policy_gate",
                reason=reason or "Memory recall needs explicit provenance before reuse.",
                evidence=str(row.get("evidence", "") or ""),
                blocked_by_gate=effect == "filtered",
                required_evidence=_default_required_evidence(),
                recommended_next_step=(
                    "Attach source_url, observed_at, and a verification signal before "
                    "allowing this memory back into recall."
                ),
            )
        )
    return repairs


def _repairs_from_diagnostics(
    diagnostics: Mapping[str, Any],
) -> list[MemoryEvidenceRepairItem]:
    repairs: list[MemoryEvidenceRepairItem] = []
    for issue in _list(diagnostics.get("issues")):
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("code", "") or "")
        if code == "collect_evidence_before_reuse":
            repairs.append(
                MemoryEvidenceRepairItem(
                    provider=str(issue.get("provider", "") or ""),
                    priority="high",
                    action=ACTION_ATTACH_PROVENANCE,
                    source="diagnostics",
                    reason=str(issue.get("reason", "") or "Recall has weak evidence."),
                    evidence=str(issue.get("evidence", "") or ""),
                    required_evidence=_default_required_evidence(),
                    recommended_next_step=str(
                        issue.get("recommended_next_step", "")
                        or "Attach provenance before reuse."
                    ),
                )
            )
        elif code == "refresh_stale_recall":
            repairs.append(
                MemoryEvidenceRepairItem(
                    provider=str(issue.get("provider", "") or ""),
                    priority="medium",
                    action=ACTION_REFRESH_OBSERVATION,
                    source="diagnostics",
                    reason=str(issue.get("reason", "") or "Recall may be stale."),
                    evidence=str(issue.get("evidence", "") or ""),
                    required_evidence=("observed_at", "freshness_check"),
                    recommended_next_step=str(
                        issue.get("recommended_next_step", "")
                        or "Refresh observed_at and confirm the memory still applies."
                    ),
                )
            )
        elif code == "review_conflicting_recall":
            repairs.append(
                MemoryEvidenceRepairItem(
                    provider=str(issue.get("provider", "") or ""),
                    priority="high",
                    action=ACTION_REVIEW_CONFLICT,
                    source="diagnostics",
                    reason=str(issue.get("reason", "") or "Recall may conflict."),
                    evidence=str(issue.get("evidence", "") or ""),
                    required_evidence=("conflict_resolution", "source_url"),
                    recommended_next_step=str(
                        issue.get("recommended_next_step", "")
                        or "Resolve conflict before this memory is reused."
                    ),
                )
            )
    return repairs


def _repairs_from_policy(policy: Mapping[str, Any]) -> list[MemoryEvidenceRepairItem]:
    repairs: list[MemoryEvidenceRepairItem] = []
    for decision in _list(policy.get("decisions")):
        if not isinstance(decision, Mapping):
            continue
        if str(decision.get("action", "") or "") != REQUIRE_PROVENANCE:
            continue
        repairs.append(
            MemoryEvidenceRepairItem(
                provider=str(decision.get("provider", "") or ""),
                priority="high" if decision.get("provider") else "medium",
                action=ACTION_ATTACH_PROVENANCE,
                source="auto_policy",
                reason=str(
                    decision.get("reason", "")
                    or "Auto-policy requires provenance before memory reuse."
                ),
                evidence=str(decision.get("evidence", "") or ""),
                required_evidence=_default_required_evidence(),
                recommended_next_step=str(
                    decision.get("recommended_next_step", "")
                    or "Repair provenance before reusing this memory."
                ),
            )
        )
    return repairs


def _repairs_from_audit(audit: Mapping[str, Any]) -> list[MemoryEvidenceRepairItem]:
    repairs: list[MemoryEvidenceRepairItem] = []
    for entry in _list(audit.get("entries")):
        if not isinstance(entry, Mapping):
            continue
        recommendation = str(entry.get("recommendation", "") or "")
        dimensions = (
            entry.get("dimensions") if isinstance(entry.get("dimensions"), Mapping) else {}
        )
        weak_source = _float(dimensions.get("source_strength")) < 0.5
        weak_evidence = _float(dimensions.get("evidence_strength")) < 0.5
        if recommendation != "needs_evidence" and not (weak_source or weak_evidence):
            continue
        decision = str(entry.get("decision", "") or "")
        repairs.append(
            MemoryEvidenceRepairItem(
                provider=str(entry.get("provider", "") or ""),
                priority="high" if decision in {"injected", "fallback_injected"} else "medium",
                action=ACTION_ATTACH_PROVENANCE,
                source="recall_audit",
                reason=str(
                    entry.get("reason", "")
                    or "Recall audit found weak source or evidence strength."
                ),
                evidence=f"source_strength={dimensions.get('source_strength', 'n/a')}; "
                f"evidence_strength={dimensions.get('evidence_strength', 'n/a')}",
                content_preview=str(entry.get("content_preview", "") or ""),
                required_evidence=_default_required_evidence(),
                recommended_next_step=(
                    "Attach source_url, observed_at, and verification before trusting "
                    "this recall again."
                ),
            )
        )
    return repairs


def _dedupe_repairs(
    repairs: list[MemoryEvidenceRepairItem],
) -> list[MemoryEvidenceRepairItem]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[MemoryEvidenceRepairItem] = []
    priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    repairs.sort(
        key=lambda item: (
            priority_rank.get(item.priority, 9),
            item.provider,
            item.action,
            item.source,
        )
    )
    for repair in repairs:
        key = (
            repair.provider,
            repair.action,
            repair.reason,
            repair.content_preview,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(repair)
    return deduped


def _default_required_evidence() -> tuple[str, ...]:
    return ("source_url", "observed_at", "verification_signal")


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
