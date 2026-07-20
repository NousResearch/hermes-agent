#!/usr/bin/env python3
"""Strict judgment parsing and bounded source assembly; no model execution."""

from __future__ import annotations

import json
from typing import Any

ROLES = frozenset({"architect", "builder"})
TOP_KEYS = {
    "consensus",
    "uniqueFindings",
    "divergences",
    "rejected",
    "finalRecommendation",
    "confidence",
    "unverifiedAssumptions",
}
PROVENANCE_KEYS = (
    "status",
    "lane",
    "requested_provider",
    "requested_model",
    "actual_provider",
    "actual_model",
    "fallback_used",
)


def extract_judge_output(completion: dict[str, Any]) -> str:
    """Return native judge summary only when terminal provenance is exact."""
    if not isinstance(completion, dict) or completion.get("status") != "completed":
        raise ValueError("judge completion is not completed")
    if completion.get("lane") != "judge" or completion.get("fallback_used") is not False:
        raise ValueError("judge lane or fallback provenance is invalid")
    requested_provider = completion.get("requested_provider")
    requested_model = completion.get("requested_model")
    if (
        not isinstance(requested_provider, str)
        or requested_provider != completion.get("actual_provider")
        or not isinstance(requested_model, str)
        or requested_model != completion.get("actual_model")
    ):
        raise ValueError("judge requested and actual provenance do not match")
    summary = completion.get("summary")
    if not isinstance(summary, str) or not summary:
        raise ValueError("judge completion has no terminal summary")
    return summary


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sources(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("sources must be a non-empty list")
    if any(not isinstance(role, str) or role not in ROLES for role in value):
        raise ValueError("sources contains an unknown role")
    if len(set(value)) != len(value):
        raise ValueError("sources contains duplicates")
    return value


def _items(value: Any, *, rejected: bool = False) -> None:
    if not isinstance(value, list):
        raise ValueError("judgment sections must be lists")
    expected = {"statement", "sources", "reason"} if rejected else {"statement", "sources"}
    for item in value:
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("judgment item has invalid fields")
        _text(item["statement"], "statement")
        _sources(item["sources"])
        if rejected:
            _text(item["reason"], "reason")


def parse_judgment(raw: str) -> dict[str, Any]:
    """Parse exactly one raw JSON object and enforce the fusion schema."""
    if not isinstance(raw, str):
        raise ValueError("judgment must be text")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("judgment is not raw JSON") from exc
    if not isinstance(value, dict) or set(value) != TOP_KEYS:
        raise ValueError("judgment has invalid top-level fields")
    _items(value["consensus"])
    _items(value["uniqueFindings"])
    _items(value["divergences"])
    _items(value["rejected"], rejected=True)
    _text(value["finalRecommendation"], "finalRecommendation")
    if value["confidence"] not in {"low", "medium", "high"}:
        raise ValueError("confidence is invalid")
    assumptions = value["unverifiedAssumptions"]
    if not isinstance(assumptions, list):
        raise ValueError("unverifiedAssumptions must be a list")
    for assumption in assumptions:
        _text(assumption, "assumption")
    return value


def build_judge_input(sources: list[dict[str, Any]], max_source_chars: int = 8_000) -> str:
    """Build independently bounded, attributed source blocks for the judge."""
    if not isinstance(max_source_chars, int) or max_source_chars < 1:
        raise ValueError("max_source_chars must be positive")
    seen: set[str] = set()
    attributed: list[dict[str, Any]] = []
    for source in sources:
        role = source.get("role")
        if role not in ROLES or role in seen:
            raise ValueError("source role is invalid or duplicated")
        seen.add(role)
        if source.get("status") != "completed":
            raise ValueError("only completed sources may reach the judge")
        content = source.get("content")
        if not isinstance(content, str):
            raise ValueError("source content must be text")
        attributed.append({
            "role": role,
            **{key: source.get(key) for key in PROVENANCE_KEYS},
            "content": content[:max_source_chars],
        })
    if not attributed:
        raise ValueError("at least one completed source is required")
    return json.dumps(
        {"sources": attributed},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
