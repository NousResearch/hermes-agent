from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping


FUSION_POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
}

SCORING_DIMENSIONS = (
    "semantic_score",
    "keyword_score",
    "entity_score",
    "temporal_score",
    "project_scope_score",
    "source_trust_score",
    "governance_score",
    "final_score",
)

WEIGHTS = {
    "semantic_score": 0.22,
    "keyword_score": 0.2,
    "entity_score": 0.12,
    "temporal_score": 0.14,
    "project_scope_score": 0.13,
    "source_trust_score": 0.09,
    "governance_score": 0.1,
}

SELECTION_THRESHOLD = 0.35
DEFAULT_LIMIT = 5

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_UNSAFE_TEXT_MARKERS = (
    "direct_write",
    "direct memory write",
    "write_memory",
    "write graph",
    "modify config",
    "approve allowlist",
    "operation ledger event",
    "create proposal",
)
_UNSAFE_BOOL_KEYS = (
    "unsafe",
    "write_allowed",
    "would_write_memory",
    "would_modify_config",
    "would_write_graph",
    "creates_operation_event",
    "create_operation_event",
    "approves_allowlist",
    "approve_allowlist",
    "external_auto_recall_allowed",
)


def fuse_memory_retrieval(
    *,
    query: str,
    candidates: Iterable[Mapping[str, Any]],
    project_scope: str | None = None,
    entity_ids: Iterable[str] | None = None,
    now: str | datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Rank candidate memory records without writing memory, config, graph, or ledger state."""
    evaluation_now = _coerce_datetime(now) or datetime.now(UTC)
    query_tokens = _tokens(query)
    entity_set = {str(entity_id) for entity_id in entity_ids or [] if str(entity_id)}
    scored = [
        _score_candidate(
            query=query,
            query_tokens=query_tokens,
            candidate=dict(candidate),
            project_scope=project_scope,
            query_entity_ids=entity_set,
            now=evaluation_now,
            index=index,
        )
        for index, candidate in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item["final_score"], str(item["id"]), item["_index"]))

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in scored:
        hard_reject = item.pop("_hard_reject")
        index = item.pop("_index")
        item["_index"] = index
        if hard_reject:
            rejected.append(_rejected_item(item, item["reason"]))
        elif len(selected) < limit and item["final_score"] >= SELECTION_THRESHOLD:
            selected.append(_selected_item(item))
        else:
            reason = "ranked_below_limit" if len(selected) >= limit else "below_selection_threshold"
            rejected.append(_rejected_item(item, reason))

    return {
        "query": query,
        "selected_memories": selected,
        "rejected_memories": rejected,
        "scores": {str(item["id"]): item["component_scores"] | {"final_score": item["final_score"]} for item in scored},
        "policy": dict(FUSION_POLICY),
        "explanation": {
            "version": "hybrid_retrieval_fusion_v0.1",
            "scoring_dimensions": list(SCORING_DIMENSIONS),
            "weights": dict(WEIGHTS),
            "selection_threshold": SELECTION_THRESHOLD,
            "limit": limit,
            "now": evaluation_now.isoformat().replace("+00:00", "Z"),
            "rules": [
                "Unsafe governance indicators are rejected before selection.",
                "Project-scoped queries hard-reject candidates from unrelated projects.",
                "Expired or future-validity candidates are penalized by temporal_score.",
                "All non-selected candidates remain visible in rejected_memories.",
            ],
        },
    }


def _score_candidate(
    *,
    query: str,
    query_tokens: set[str],
    candidate: dict[str, Any],
    project_scope: str | None,
    query_entity_ids: set[str],
    now: datetime,
    index: int,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("id") or f"candidate_{index}")
    text = _candidate_text(candidate)
    candidate_tokens = _tokens(text)
    governance_score, unsafe_reasons = _governance_score(candidate.get("governance"))
    project_score = _project_scope_score(candidate.get("project_id"), project_scope)
    temporal_score = _temporal_score(candidate, now)
    component_scores = {
        "semantic_score": _semantic_score(query_tokens, candidate_tokens),
        "keyword_score": _keyword_score(query, query_tokens, text, candidate_tokens),
        "entity_score": _entity_score(candidate.get("entity_ids"), query_entity_ids),
        "temporal_score": temporal_score,
        "project_scope_score": project_score,
        "source_trust_score": _source_trust_score(candidate),
        "governance_score": governance_score,
    }
    final_score = round(sum(component_scores[name] * weight for name, weight in WEIGHTS.items()), 4)

    reason = "eligible"
    hard_reject = False
    if unsafe_reasons:
        reason = "unsafe_governance:" + ",".join(unsafe_reasons)
        hard_reject = True
    elif project_scope and candidate.get("project_id") and str(candidate.get("project_id")) != str(project_scope):
        reason = "project_scope_mismatch"
        hard_reject = True

    return {
        "id": candidate_id,
        "text": text,
        "project_id": candidate.get("project_id"),
        "source": candidate.get("source"),
        "provenance": candidate.get("provenance"),
        "component_scores": component_scores,
        "final_score": final_score,
        "reason": reason,
        "_hard_reject": hard_reject,
        "_index": index,
    }


def _selected_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "text": item["text"],
        "project_id": item.get("project_id"),
        "source": item.get("source"),
        "provenance": item.get("provenance"),
        "final_score": item["final_score"],
        "component_scores": dict(item["component_scores"]),
    }


def _rejected_item(item: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": item["id"],
        "reason": reason,
        "final_score": item["final_score"],
        "component_scores": dict(item["component_scores"]),
    }


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    parts = [
        candidate.get("title"),
        candidate.get("summary"),
        candidate.get("content"),
        candidate.get("text"),
    ]
    return " ".join(str(part) for part in parts if part)


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")}


def _semantic_score(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    return round(overlap / len(query_tokens | candidate_tokens), 4)


def _keyword_score(query: str, query_tokens: set[str], text: str, candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    exact_token_score = len(query_tokens & candidate_tokens) / len(query_tokens)
    normalized_query = " ".join(str(query).lower().split())
    normalized_text = " ".join(str(text).lower().split())
    phrase_score = 1.0 if normalized_query and normalized_query in normalized_text else 0.0
    return round((exact_token_score * 0.75) + (phrase_score * 0.25), 4)


def _entity_score(candidate_entity_ids: Any, query_entity_ids: set[str]) -> float:
    if not query_entity_ids:
        return 0.5
    candidate_set = {str(entity_id) for entity_id in _as_list(candidate_entity_ids) if str(entity_id)}
    if not candidate_set:
        return 0.0
    return round(len(candidate_set & query_entity_ids) / len(query_entity_ids), 4)


def _temporal_score(candidate: Mapping[str, Any], now: datetime) -> float:
    valid_from = _coerce_datetime(candidate.get("valid_from"))
    valid_until = _coerce_datetime(candidate.get("valid_until"))
    if valid_until and valid_until < now:
        return 0.05
    if valid_from and valid_from > now:
        return 0.2
    if valid_from or valid_until:
        return 1.0

    created_at = _coerce_datetime(candidate.get("created_at"))
    if not created_at:
        return 0.5
    age_days = max((now - created_at).total_seconds() / 86400, 0)
    if age_days <= 7:
        return 0.9
    if age_days <= 30:
        return 0.75
    if age_days <= 365:
        return 0.5
    return 0.25


def _project_scope_score(project_id: Any, project_scope: str | None) -> float:
    if not project_scope:
        return 0.5
    if project_id is None:
        return 0.25
    return 1.0 if str(project_id) == str(project_scope) else 0.0


def _source_trust_score(candidate: Mapping[str, Any]) -> float:
    if candidate.get("provenance"):
        return 1.0
    if candidate.get("source"):
        return 0.6
    return 0.25


def _governance_score(governance: Any) -> tuple[float, list[str]]:
    if governance is None:
        return 0.5, []
    if isinstance(governance, str):
        lowered = governance.lower()
        reasons = [marker for marker in _UNSAFE_TEXT_MARKERS if marker in lowered]
        if reasons:
            return 0.0, reasons
        if "read_only" in lowered or "proposal" in lowered or "governed" in lowered:
            return 1.0, []
        return 0.5, []
    if not isinstance(governance, Mapping):
        return 0.5, []

    reasons = [key for key in _UNSAFE_BOOL_KEYS if governance.get(key) is True]
    mode = str(governance.get("mode", "")).lower()
    if any(marker in mode for marker in ("write", "mutate", "modify")):
        reasons.append("mode")
    if reasons:
        return 0.0, reasons
    if governance.get("read_only") is True or governance.get("proposal_governed") is True:
        return 1.0, []
    return 0.5, []


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]
