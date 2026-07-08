"""Ranking utilities for recall results."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from llmwiki_hermes.types import MemoryType, NoteKind
from llmwiki_hermes.utils.slug import slugify

SEMANTIC_QUERY_HINTS = ("what", "define", "definition", "是什么", "定义")
EPISODIC_QUERY_HINTS = ("when", "meeting", "project", "case", "讨论", "会议", "项目", "何时")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")


def classify_query_bias(query: str) -> str:
    """Choose which memory type should receive a default boost."""

    lowered = query.lower()
    if any(token in lowered for token in SEMANTIC_QUERY_HINTS):
        return MemoryType.SEMANTIC.value
    if any(token in lowered for token in EPISODIC_QUERY_HINTS):
        return MemoryType.EPISODIC.value
    return MemoryType.AUTO.value


def recency_bonus(raw_value: str) -> float:
    """Return a small positive score for recent content."""

    if not raw_value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(raw_value).date()
    except ValueError:
        try:
            parsed = date.fromisoformat(raw_value)
        except ValueError:
            return 0.0
    days_old = max((date.today() - parsed).days, 0)
    return max(0.0, 1.0 - min(days_old, 365) / 365)


def _normalized_text(value: str) -> str:
    return slugify(value.replace("_", "-"))


def _contains_query(candidate: str, query: str) -> bool:
    query_key = _normalized_text(query)
    candidate_key = _normalized_text(candidate)
    return bool(query_key and candidate_key and query_key in candidate_key)


def _overlap_bonus(query: str, candidate: str) -> float:
    query_tokens = {token for token in _normalized_text(query).split("-") if token}
    candidate_tokens = {token for token in _normalized_text(candidate).split("-") if token}
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    return min(overlap * 0.2, 0.6)


def _metadata_bonus(row: dict[str, Any], query: str) -> float:
    bonus = 0.0
    project = str(row.get("project") or "")
    if project and _contains_query(query, project):
        bonus += 0.8

    query_date = DATE_PATTERN.search(query)
    row_date = str(row.get("date") or "")
    if query_date and row_date == query_date.group(0):
        bonus += 0.8

    source_refs = json.loads(row.get("source_refs_json") or "[]")
    if any(_contains_query(query, str(source_ref)) for source_ref in source_refs):
        bonus += 0.4
    return bonus


def score_row(row: dict[str, Any], query: str, memory_type: str) -> float:
    """Compute a deterministic score for a search row."""

    bias = classify_query_bias(query)
    title_key = _normalized_text(str(row["title"]))
    query_key = _normalized_text(query)
    title_hit = (
        2.0 if title_key == query_key else 1.0 if _contains_query(row["title"], query) else 0.0
    )
    title_hit += _overlap_bonus(query, str(row["title"]))

    alias_hit = 0.0
    for alias in json.loads(row.get("aliases_json") or "[]"):
        alias_text = str(alias)
        alias_key = _normalized_text(alias_text)
        if alias_key == query_key:
            alias_hit = max(alias_hit, 1.25)
        elif _contains_query(alias_text, query):
            alias_hit = max(alias_hit, 0.75)
        alias_hit = max(alias_hit, _overlap_bonus(query, alias_text))

    fts_score = float(row.get("fts_score") or 0.0)
    fts_bonus = 1.0 / (1.0 + max(fts_score, 0.0))
    kind_bonus = 0.0
    if memory_type == MemoryType.SEMANTIC.value and row["kind"] == NoteKind.SEMANTIC.value:
        kind_bonus += 2.0
    if memory_type == MemoryType.EPISODIC.value and row["kind"] == NoteKind.EPISODIC.value:
        kind_bonus += 2.0
    if memory_type == MemoryType.AUTO.value:
        if bias == MemoryType.SEMANTIC.value and row["kind"] == NoteKind.SEMANTIC.value:
            kind_bonus += 1.5
        if bias == MemoryType.EPISODIC.value and row["kind"] == NoteKind.EPISODIC.value:
            kind_bonus += 1.5
        if bias == MemoryType.AUTO.value and row["kind"] == NoteKind.SEMANTIC.value:
            kind_bonus += 0.5

    return round(
        title_hit
        + alias_hit
        + fts_bonus
        + kind_bonus
        + _metadata_bonus(row, query)
        + recency_bonus(str(row.get("updated_at", ""))),
        5,
    )
