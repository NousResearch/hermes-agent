"""Goal Discovery Runtime canary.

Pure, local, read-only discovery over existing Hermes goal and Executive
Runtime artifacts. The module accepts an ObjectiveContext and returns
``goal_discovery_report.json``-shaped data. It searches for prior goals,
reports, checkpoints, contracts, Kanban references, capabilities, possible
duplicates, and reusable work before any new work is created elsewhere.

This canary intentionally does not create goals, build strategies, create
execution contracts, apply Goal Runner state, touch Kanban, spawn workers,
call providers, or write to knowledge stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .capability_discovery_runtime import (
    CapabilityDiscoveryIndex,
    CapabilityMatch,
    ObjectiveContext,
    _assets,
    _confidence,
    _dedupe,
    _match_files,
    _missing,
    _objective_tokens,
)

REPORT_FIELDS = (
    "matched_goals",
    "matched_reports",
    "matched_checkpoints",
    "matched_contracts",
    "matched_kanban_refs",
    "matched_capabilities",
    "possible_duplicates",
    "related_work",
    "confidence",
    "reusable_prior_work",
    "missing_goal_context",
)

_GOAL_HINTS = (
    "goal",
    "objective",
    "objective_id",
    "goal_id",
    "standing goal",
    "runtime goal",
    "objetivo",
)

_REPORT_HINTS = (
    "report",
    "validation",
    "manifest",
    "verified_hashes",
    "canary_validation",
    "rollback",
)

_CHECKPOINT_HINTS = (
    "checkpoint",
    "checkpoint_pass",
    "official",
    "frozen",
    "promote_to_checkpoint",
)

_CONTRACT_HINTS = (
    "contract",
    "execution_contract",
    "contract_id",
    "hard_constraints",
    "rollback_strategy",
)

_KANBAN_REF_HINTS = (
    "kanban",
    "taskspec",
    "board",
    "worker",
    "dispatch",
    "claim",
)

_CAPABILITY_HINTS = (
    "capability",
    "discovery",
    "analyzer",
    "runtime",
    "canary",
    "schema",
)


@dataclass(frozen=True)
class GoalDiscoveryIndex:
    """Read-only roots used by the Goal Discovery canary.

    The defaults intentionally mirror the already validated Capability
    Discovery roots and add only read-only goal/contract/Kanban-reference
    search locations. Missing roots are ignored by the shared matcher.
    """

    capability_index: CapabilityDiscoveryIndex = CapabilityDiscoveryIndex()
    goal_roots: tuple[Path, ...] = ()
    report_roots: tuple[Path, ...] = ()
    checkpoint_roots: tuple[Path, ...] = ()
    contract_roots: tuple[Path, ...] = ()
    kanban_ref_roots: tuple[Path, ...] = ()
    capability_roots: tuple[Path, ...] = ()

    def roots_for_goals(self) -> tuple[Path, ...]:
        return self.goal_roots or (
            self.capability_index.workflow_roots
            + self.capability_index.report_roots
            + (self.capability_index.capability_roots[0],)
        )

    def roots_for_reports(self) -> tuple[Path, ...]:
        return self.report_roots or self.capability_index.report_roots

    def roots_for_checkpoints(self, context: ObjectiveContext) -> tuple[Path, ...]:
        roots = self.checkpoint_roots or self.capability_index.checkpoint_roots
        if context.source_checkpoint:
            return (Path(context.source_checkpoint),) + tuple(roots)
        return tuple(roots)

    def roots_for_contracts(self) -> tuple[Path, ...]:
        return self.contract_roots or (
            self.capability_index.workflow_roots
            + (self.capability_index.capability_roots[0],)
        )

    def roots_for_kanban_refs(self) -> tuple[Path, ...]:
        return self.kanban_ref_roots or (
            self.capability_index.workflow_roots
            + (self.capability_index.capability_roots[0],)
        )

    def roots_for_capabilities(self) -> tuple[Path, ...]:
        return self.capability_roots or self.capability_index.capability_roots


@dataclass(frozen=True)
class GoalDiscoveryReport:
    matched_goals: tuple[CapabilityMatch, ...]
    matched_reports: tuple[CapabilityMatch, ...]
    matched_checkpoints: tuple[CapabilityMatch, ...]
    matched_contracts: tuple[CapabilityMatch, ...]
    matched_kanban_refs: tuple[CapabilityMatch, ...]
    matched_capabilities: tuple[CapabilityMatch, ...]
    possible_duplicates: tuple[dict[str, Any], ...]
    related_work: tuple[dict[str, Any], ...]
    confidence: float
    reusable_prior_work: tuple[dict[str, Any], ...]
    missing_goal_context: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_goals": [m.to_dict() for m in self.matched_goals],
            "matched_reports": [m.to_dict() for m in self.matched_reports],
            "matched_checkpoints": [m.to_dict() for m in self.matched_checkpoints],
            "matched_contracts": [m.to_dict() for m in self.matched_contracts],
            "matched_kanban_refs": [m.to_dict() for m in self.matched_kanban_refs],
            "matched_capabilities": [m.to_dict() for m in self.matched_capabilities],
            "possible_duplicates": list(self.possible_duplicates),
            "related_work": list(self.related_work),
            "confidence": round(float(self.confidence), 4),
            "reusable_prior_work": list(self.reusable_prior_work),
            "missing_goal_context": list(self.missing_goal_context),
        }


def _related_work(matches: Iterable[CapabilityMatch], limit: int = 30) -> tuple[dict[str, Any], ...]:
    related: list[dict[str, Any]] = []
    for match in tuple(matches)[:limit]:
        related.append(
            {
                "kind": match.kind,
                "name": match.name,
                "path": match.path,
                "score": round(float(match.score), 4),
                "evidence_digest": match.digest,
                "relationship": "; ".join(match.reasons[:3]) or "search_match",
            }
        )
    return tuple(related)


def _name_terms(name: str) -> set[str]:
    normalized = name.lower().translate(str.maketrans({"_": "-"}))
    return {part for part in normalized.split("-") if part}


def _possible_duplicates(
    matched_goals: tuple[CapabilityMatch, ...],
    matched_reports: tuple[CapabilityMatch, ...],
    matched_checkpoints: tuple[CapabilityMatch, ...],
) -> tuple[dict[str, Any], ...]:
    candidates = tuple(matched_goals) + tuple(matched_reports) + tuple(matched_checkpoints)
    duplicates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for left_index, left in enumerate(candidates):
        if left.score < 0.45:
            continue
        left_terms = _name_terms(left.name)
        for right in candidates[left_index + 1 :]:
            if right.score < 0.45 or left.path == right.path:
                continue
            pair: tuple[str, str] = (
                (left.path, right.path)
                if left.path <= right.path
                else (right.path, left.path)
            )
            if pair in seen_pairs:
                continue
            right_terms = _name_terms(right.name)
            shared_terms = sorted(left_terms & right_terms)
            if left.digest == right.digest or len(shared_terms) >= 2:
                seen_pairs.add(pair)
                duplicates.append(
                    {
                        "left": left.to_dict(),
                        "right": right.to_dict(),
                        "reason": "same_digest" if left.digest == right.digest else "shared_name_terms",
                        "shared_terms": shared_terms[:8],
                    }
                )
    return tuple(duplicates[:20])


def discover_goal_related_work(
    context: ObjectiveContext | dict[str, Any],
    *,
    index: GoalDiscoveryIndex | None = None,
) -> GoalDiscoveryReport:
    """Search existing goal-related work and return a serializable report.

    This canary only reads text artifacts under configured roots and computes
    deterministic matches. It has no writer or applier function by design;
    callers that need a file artifact must serialize ``to_dict()`` outside the
    runtime boundary they control.
    """

    objective_context = (
        ObjectiveContext.from_mapping(context) if isinstance(context, dict) else context
    )
    if not objective_context.objective_text.strip():
        raise ValueError("objective_text must be non-empty")
    if not objective_context.user_id.strip():
        raise ValueError("user_id must be non-empty")

    discovery_index = index or GoalDiscoveryIndex()
    tokens = _objective_tokens(objective_context)

    matched_goals = _match_files(
        kind="goal",
        roots=discovery_index.roots_for_goals(),
        tokens=tokens,
        hints=_GOAL_HINTS,
        threshold=0.12,
    )
    matched_reports = _match_files(
        kind="report",
        roots=discovery_index.roots_for_reports(),
        tokens=tokens,
        hints=_REPORT_HINTS,
        threshold=0.14,
    )
    matched_checkpoints = _match_files(
        kind="checkpoint",
        roots=discovery_index.roots_for_checkpoints(objective_context),
        tokens=tokens,
        hints=_CHECKPOINT_HINTS,
        threshold=0.12,
    )
    matched_contracts = _match_files(
        kind="contract",
        roots=discovery_index.roots_for_contracts(),
        tokens=tokens,
        hints=_CONTRACT_HINTS,
        threshold=0.14,
    )
    matched_kanban_refs = _match_files(
        kind="kanban_ref",
        roots=discovery_index.roots_for_kanban_refs(),
        tokens=tokens,
        hints=_KANBAN_REF_HINTS,
        threshold=0.18,
    )
    explicit_capabilities = _match_files(
        kind="capability",
        roots=discovery_index.roots_for_capabilities(),
        tokens=tokens,
        hints=_CAPABILITY_HINTS,
        threshold=0.14,
    )

    all_matches = _dedupe(
        tuple(matched_goals)
        + tuple(matched_reports)
        + tuple(matched_checkpoints)
        + tuple(matched_contracts)
        + tuple(matched_kanban_refs)
        + tuple(explicit_capabilities)
    )
    matched_capabilities = _dedupe(tuple(explicit_capabilities) + tuple(all_matches))[:40]
    groups = (
        matched_goals,
        matched_reports,
        matched_checkpoints,
        matched_contracts,
        matched_kanban_refs,
        matched_capabilities,
    )

    return GoalDiscoveryReport(
        matched_goals=matched_goals,
        matched_reports=matched_reports,
        matched_checkpoints=matched_checkpoints,
        matched_contracts=matched_contracts,
        matched_kanban_refs=matched_kanban_refs,
        matched_capabilities=matched_capabilities,
        possible_duplicates=_possible_duplicates(
            matched_goals,
            matched_reports,
            matched_checkpoints,
        ),
        related_work=_related_work(all_matches),
        confidence=_confidence(groups),
        reusable_prior_work=_assets(all_matches),
        missing_goal_context=_missing(
            goals=matched_goals,
            reports=matched_reports,
            checkpoints=matched_checkpoints,
            contracts=matched_contracts,
            kanban_refs=matched_kanban_refs,
            capabilities=matched_capabilities,
        ),
    )


__all__ = [
    "GoalDiscoveryIndex",
    "GoalDiscoveryReport",
    "ObjectiveContext",
    "REPORT_FIELDS",
    "discover_goal_related_work",
]
