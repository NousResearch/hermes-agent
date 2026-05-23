from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

from agent.memory_bitemporal_fact_graph import BitemporalFact, normalize_fact, select_current_facts
from agent.memory_contradiction_engine import explain_contradiction_group, group_contradictions


MEMORY_COMPILER_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
    "creates_review_candidates_only": True,
}


def compile_memory_patterns(
    facts: Iterable[Mapping[str, Any] | BitemporalFact],
    contradiction_groups: Iterable[Mapping[str, Any]] | None = None,
    project_scope: str | None = None,
) -> dict[str, Any]:
    """Compile low-level facts into review-only methodology/procedure candidates."""
    normalized = [normalize_fact(fact) for fact in facts]
    scoped_facts = [fact for fact in normalized if project_scope is None or fact.project_id == project_scope]
    compilation_time = _compilation_time(scoped_facts)
    current_facts = select_current_facts(scoped_facts, compilation_time, project_scope=project_scope)
    groups = _scoped_contradiction_groups(
        contradiction_groups if contradiction_groups is not None else group_contradictions(scoped_facts),
        project_scope,
    )
    patterns = _extract_patterns(scoped_facts, current_facts, groups, project_scope)
    methodology = compile_methodology_candidate(patterns, project_scope=project_scope)
    procedure = compile_procedure_block_candidate(methodology, project_scope=project_scope)
    result = {
        "project_scope": project_scope,
        "input_fact_count": len(normalized),
        "current_fact_count": len(current_facts),
        "contradiction_group_count": len(groups),
        "patterns": patterns,
        "methodology_candidate": methodology,
        "procedure_block_candidate": procedure,
        "review_recommendation": recommend_compilation_action(
            {
                "patterns": patterns,
                "methodology_candidate": methodology,
                "procedure_block_candidate": procedure,
                "policy": dict(MEMORY_COMPILER_POLICY),
            }
        ),
        "trace": [],
        "policy": dict(MEMORY_COMPILER_POLICY),
    }
    result["trace"] = explain_compilation_trace(result)
    return result


def compile_methodology_candidate(
    patterns: Iterable[Mapping[str, Any]],
    project_scope: str | None = None,
) -> dict[str, Any]:
    pattern_list = [dict(pattern) for pattern in patterns]
    rules = [_rule_for_pattern(pattern) for pattern in pattern_list]
    return {
        "candidate_type": "methodology_candidate",
        "status": "review_required",
        "project_scope": project_scope,
        "summary": "Review-only methodology candidate compiled from provided memory facts; it has not been applied.",
        "rules": rules,
        "source_pattern_ids": [pattern["pattern_id"] for pattern in pattern_list],
        "pattern_count": len(pattern_list),
        "policy": dict(MEMORY_COMPILER_POLICY),
    }


def compile_procedure_block_candidate(
    methodology: Mapping[str, Any],
    project_scope: str | None = None,
) -> dict[str, Any]:
    return {
        "block_type": "procedure_candidate",
        "status": "review_required",
        "project_scope": project_scope,
        "rules": list(methodology.get("rules", [])),
        "source_pattern_ids": list(methodology.get("source_pattern_ids", [])),
        "safety_notes": [
            "Candidate only; requires human review before any durable memory or graph action.",
            "Compiler is read-only and does not create proposals or operation-ledger events.",
            "Do not infer beyond the cited source patterns.",
        ],
        "policy": dict(MEMORY_COMPILER_POLICY),
    }


def explain_compilation_trace(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": "normalize_facts",
            "detail": "Input facts were normalized with agent.memory_bitemporal_fact_graph.normalize_fact.",
            "count": result.get("input_fact_count", 0),
        },
        {
            "step": "select_current_facts",
            "detail": "Current facts were selected with project-scope isolation.",
            "count": result.get("current_fact_count", 0),
        },
        {
            "step": "group_contradictions",
            "detail": "Contradiction groups were included as review-required evidence.",
            "count": result.get("contradiction_group_count", 0),
        },
        {
            "step": "compile_candidates",
            "detail": "Patterns were compiled into review-only methodology and procedure candidates.",
            "pattern_count": len(result.get("patterns", [])),
        },
    ]


def recommend_compilation_action(result: Mapping[str, Any]) -> dict[str, Any]:
    patterns = list(result.get("patterns", []))
    contradiction_patterns = [pattern for pattern in patterns if pattern.get("pattern_type") == "contradiction_review_required"]
    if contradiction_patterns:
        action = "review_contradictions_before_methodology"
        reason = "One or more compiled patterns comes from contradiction groups and must be resolved before use."
    elif patterns:
        action = "review_candidate"
        reason = "Compiled methodology and procedure candidates require review before any application or write path."
    else:
        action = "no_candidate"
        reason = "No stable, evolutionary, or contradictory memory pattern was detected."
    return {
        "action": action,
        "reason": reason,
        "creates_review_candidates_only": True,
        "policy": dict(MEMORY_COMPILER_POLICY),
    }


def _extract_patterns(
    scoped_facts: list[BitemporalFact],
    current_facts: list[BitemporalFact],
    contradiction_groups: list[dict[str, Any]],
    project_scope: str | None,
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    patterns.extend(_stable_claim_patterns(scoped_facts))
    patterns.extend(_evolution_patterns(scoped_facts, current_facts))
    patterns.extend(_contradiction_patterns(contradiction_groups))
    return sorted(
        patterns,
        key=lambda item: (
            item.get("project_id") or project_scope or "",
            item["pattern_type"],
            item["pattern_id"],
        ),
    )


def _stable_claim_patterns(facts: list[BitemporalFact]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str | None, str], list[BitemporalFact]] = {}
    for fact in facts:
        grouped.setdefault((fact.subject, fact.predicate, fact.project_id, fact.object), []).append(fact)

    patterns: list[dict[str, Any]] = []
    for (subject, predicate, project_id, obj), group in grouped.items():
        if len(group) < 2:
            continue
        fact_ids = sorted(fact.fact_id for fact in group)
        patterns.append(
            {
                "pattern_id": _pattern_id("stable", project_id, subject, predicate, obj),
                "pattern_type": "stable_repeated_claim",
                "subject": subject,
                "predicate": predicate,
                "project_id": project_id,
                "object": obj,
                "fact_ids": fact_ids,
                "support_count": len(group),
                "status": "review_required",
                "summary": f"Repeated claim observed for {subject} / {predicate}; candidate only.",
            }
        )
    return patterns


def _evolution_patterns(scoped_facts: list[BitemporalFact], current_facts: list[BitemporalFact]) -> list[dict[str, Any]]:
    by_id = {fact.fact_id: fact for fact in scoped_facts}
    current_ids = {fact.fact_id for fact in current_facts}
    patterns: list[dict[str, Any]] = []
    for fact in scoped_facts:
        superseded_ids = [fact_id for fact_id in fact.supersedes if fact_id in by_id]
        if not superseded_ids:
            continue
        patterns.append(
            {
                "pattern_id": _pattern_id("evolution", fact.project_id, fact.subject, fact.predicate, fact.fact_id),
                "pattern_type": "superseded_lineage_evolution",
                "subject": fact.subject,
                "predicate": fact.predicate,
                "project_id": fact.project_id,
                "current_fact_id": fact.fact_id,
                "current_fact_is_selected": fact.fact_id in current_ids,
                "superseded_fact_ids": sorted(superseded_ids),
                "fact_ids": sorted([fact.fact_id, *superseded_ids]),
                "status": "review_required",
                "summary": "A fact supersedes earlier provided facts; treat as an evolution candidate requiring review.",
            }
        )
    return patterns


def _contradiction_patterns(contradiction_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for group in contradiction_groups:
        explanation = explain_contradiction_group(group)
        patterns.append(
            {
                "pattern_id": _pattern_id("contradiction", group.get("project_id"), group.get("subject"), group.get("predicate")),
                "pattern_type": "contradiction_review_required",
                "subject": group.get("subject"),
                "predicate": group.get("predicate"),
                "project_id": group.get("project_id"),
                "group_id": group.get("group_id"),
                "objects": list(group.get("objects", [])),
                "fact_ids": list(group.get("fact_ids", [])),
                "status": "review_required",
                "explanation": explanation,
                "summary": "Contradictory claims were detected; no methodology should be applied before review.",
            }
        )
    return patterns


def _rule_for_pattern(pattern: Mapping[str, Any]) -> dict[str, Any]:
    pattern_type = pattern.get("pattern_type")
    if pattern_type == "stable_repeated_claim":
        text = "Candidate rule from repeated evidence: preserve the observed subject/predicate/object claim pending review."
    elif pattern_type == "superseded_lineage_evolution":
        text = "Candidate rule from lineage: preserve both current and superseded facts as an evolution trail pending review."
    elif pattern_type == "contradiction_review_required":
        text = "Candidate rule from contradiction: require human review before selecting or applying any conflicting claim."
    else:
        text = "Candidate rule requires review before use."
    return {
        "rule_id": f"rule:{pattern.get('pattern_id')}",
        "status": "review_required",
        "text": text,
        "source_pattern_id": pattern.get("pattern_id"),
    }


def _scoped_contradiction_groups(
    groups: Iterable[Mapping[str, Any]],
    project_scope: str | None,
) -> list[dict[str, Any]]:
    scoped = [dict(group) for group in groups if project_scope is None or group.get("project_id") == project_scope]
    return sorted(scoped, key=lambda item: (item.get("project_id") or "", item.get("subject") or "", item.get("predicate") or ""))


def _compilation_time(facts: list[BitemporalFact]) -> datetime:
    timestamps: list[datetime] = []
    for fact in facts:
        timestamps.extend(
            timestamp
            for timestamp in (fact.valid_from, fact.system_created_at, fact.valid_until, fact.system_invalidated_at)
            if timestamp is not None
        )
    if not timestamps:
        return datetime.max.replace(tzinfo=UTC)
    return max(timestamps)


def _pattern_id(kind: str, project_id: Any, subject: Any, predicate: Any, suffix: Any = None) -> str:
    parts = [kind, str(project_id or "*"), str(subject or ""), str(predicate or "")]
    if suffix is not None:
        parts.append(str(suffix))
    return "|".join(part.strip().replace(" ", "_") for part in parts if part is not None)
