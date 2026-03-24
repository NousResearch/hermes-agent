"""Formatting and dedupe helpers for RetainDB turn-time recall."""

from __future__ import annotations

import re
from typing import Any


def _compact_text(value: str | None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:320].rstrip()


def _normalize_text(value: str | None) -> str:
    text = _compact_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_duplicate(candidate_norm: str, corpus: list[str]) -> bool:
    if not candidate_norm:
        return True
    for existing in corpus:
        if not existing:
            continue
        if candidate_norm == existing:
            return True
        if len(candidate_norm) > 18 and candidate_norm in existing:
            return True
        if len(existing) > 18 and existing in candidate_norm:
            return True
    return False


def _dedupe_items(items: list[str], corpus: list[str], max_items: int) -> list[str]:
    deduped: list[str] = []
    for item in items:
        compact = _compact_text(item)
        norm = _normalize_text(compact)
        if not norm or _is_duplicate(norm, corpus):
            continue
        corpus.append(norm)
        deduped.append(compact)
        if len(deduped) >= max_items:
            break
    return deduped


def _extract_profile_items(profile: dict[str, Any] | None) -> list[str]:
    memories = list((profile or {}).get("memories") or [])
    return [
        _compact_text(
            (memory or {}).get("content")
            or (memory or {}).get("memory", {}).get("content")
        )
        for memory in memories
        if _compact_text(
            (memory or {}).get("content")
            or (memory or {}).get("memory", {}).get("content")
        )
    ]


def _extract_query_items(query_result: dict[str, Any] | None) -> list[str]:
    items: list[str] = []
    for result in list((query_result or {}).get("results") or []):
        content = _compact_text((result or {}).get("content"))
        if content:
            items.append(content)

    if items:
        return items

    context = _compact_text((query_result or {}).get("context"))
    if not context:
        return []
    return [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", context)
        if segment.strip()
    ]


def _extract_update_items(profile: dict[str, Any] | None, query_result: dict[str, Any] | None) -> list[str]:
    candidates = _extract_profile_items(profile) + _extract_query_items(query_result)
    update_markers = (
        "correct",
        "changed",
        "no longer",
        "instead",
        "prefer",
        "updated",
        "switch",
        "moved",
        "now uses",
    )
    return [
        item
        for item in candidates
        if any(marker in item.lower() for marker in update_markers)
    ]


def build_retaindb_overlay(
    *,
    profile: dict[str, Any] | None,
    query_result: dict[str, Any] | None,
    local_entries: list[str] | None = None,
    recent_texts: list[str] | None = None,
    max_profile_items: int = 5,
    max_memory_items: int = 5,
    max_update_items: int = 3,
) -> str:
    """Build the strict RetainDB overlay block for a single turn."""

    corpus = [
        _normalize_text(item)
        for item in (local_entries or []) + (recent_texts or [])
        if _normalize_text(item)
    ]
    profile_items = _dedupe_items(_extract_profile_items(profile), corpus, max_profile_items)
    relevant_items = _dedupe_items(_extract_query_items(query_result), corpus, max_memory_items)
    update_items = _dedupe_items(_extract_update_items(profile, query_result), corpus, max_update_items)

    if not profile_items and not relevant_items and not update_items:
        return ""

    lines = ["[RetainDB Context]", "Profile:"]
    lines.extend(f"- {item}" for item in (profile_items or ["None"]))
    lines.append("Relevant memories:")
    lines.extend(f"- {item}" for item in (relevant_items or ["None"]))
    lines.append("Open corrections / recent updates:")
    lines.extend(f"- {item}" for item in (update_items or ["None"]))
    return "\n".join(lines)
