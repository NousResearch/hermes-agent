"""Deterministic helpers for ranking recalled memory records.

The helpers in this module are intentionally dependency-free and provider
agnostic. Memory providers can use them to combine an existing backend score
with lexical query overlap, soft freshness decay, durability/category hints,
and lightweight access reinforcement. They never delete or mutate a provider's
stored records; callers own any persisted access-state file.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from datetime import datetime
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]*", re.IGNORECASE)


def memory_key(memory: str) -> str:
    """Return a stable short identifier for a memory string."""
    normalized = " ".join(str(memory or "").split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _parse_timestamp(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text or "") if len(token) > 1}


def _category(memory: str, metadata: dict[str, Any]) -> str:
    category = str(metadata.get("category") or "").strip().lower()
    if category:
        return category

    temporal = metadata.get("temporal") or {}
    if isinstance(temporal, dict):
        temporal_type = str(temporal.get("type") or "").strip().lower()
        if temporal_type:
            return temporal_type

    source = str(metadata.get("source") or "").lower()
    text = f"{source} {memory}".lower()
    if "open issue" in text or "open_issues" in source:
        return "open_issue"
    if "current context" in text or "current_context" in source or "checkpoint" in text:
        return "current_context"
    if "preference" in text or "standing constraint" in text:
        return "preference"
    if "service" in text or "host" in text or "route" in text:
        return "service"
    if "cron" in text or "automation" in text:
        return "automation"
    return "generic"


def _durability(category: str, metadata: dict[str, Any]) -> str:
    durability = str(metadata.get("durability") or "").strip().lower()
    if durability:
        return durability
    if category in {"preference", "service", "automation", "generic"}:
        return "stable"
    return "working"


def _last_seen(metadata: dict[str, Any]) -> float | None:
    temporal = metadata.get("temporal") or {}
    if isinstance(temporal, dict):
        for key in ("last_verified", "observed_at", "updated", "created"):
            parsed = _parse_timestamp(temporal.get(key))
            if parsed is not None:
                return parsed
    for key in ("last_accessed", "last_verified", "updated", "created", "observed_at"):
        parsed = _parse_timestamp(metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def rerank_memories(
    items: list[dict[str, Any]],
    *,
    query: str = "",
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Annotate and rank memory items by backend score, query match, and decay.

    Input items should contain at least ``memory`` and may include ``score`` and
    ``metadata``. Returned dicts preserve original fields plus ``score``,
    ``freshness``, ``age_days``, ``category``, ``durability``, ``memory_key``,
    and ``rank_input``. Stable durable facts decay less aggressively than working
    context, while current-context/open-issue records are protected from strong
    decay because old active work can still be relevant until explicitly closed.
    """
    now = float(now or time.time())
    query_tokens = _tokenize(query)
    ranked: list[dict[str, Any]] = []

    for idx, item in enumerate(items):
        memory = str(item.get("memory") or "")
        metadata = dict(item.get("metadata") or {})
        category = _category(memory, metadata)
        durability = _durability(category, metadata)
        seen = _last_seen(metadata)
        age_days = None if seen is None else max(0.0, (now - seen) / 86400.0)

        if age_days is None:
            freshness = "untracked"
            decay = 0.84
        elif age_days <= 7:
            freshness = "fresh"
            decay = 1.0
        elif age_days <= 30:
            freshness = "aging"
            decay = 0.92
        elif age_days <= 90:
            freshness = "stale"
            decay = 0.78
        else:
            freshness = "stale"
            decay = 0.62

        if durability == "stable":
            decay = max(decay, 0.88)
        if category in {"current_context", "open_issue", "checkpoint"}:
            decay = max(decay, 0.96)

        try:
            base_score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            base_score = 0.0

        if query_tokens:
            tokens = _tokenize(memory)
            overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
        else:
            overlap = 0.15

        try:
            access_count = float(metadata.get("access_count") or 0.0)
        except (TypeError, ValueError):
            access_count = 0.0
        access_bonus = min(0.08, math.log1p(access_count) / 50.0)

        score = max(0.0, min(1.0, (base_score * 0.75 + overlap * 0.25 + access_bonus) * decay))
        out = dict(item)
        out.update(
            {
                "score": score,
                "freshness": freshness,
                "age_days": age_days,
                "category": category,
                "durability": durability,
                "memory_key": memory_key(memory),
                "rank_input": idx,
            }
        )
        ranked.append(out)

    ranked.sort(key=lambda item: (float(item.get("score") or 0.0), -int(item.get("rank_input") or 0)), reverse=True)
    return ranked


def reinforce_access(
    state: dict[str, Any],
    memories: list[str],
    *,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Record access metadata for selected memories in ``state`` and return it.

    ``state`` is caller-owned so providers can persist it wherever their backend
    expects. The function mutates and returns the same dict for convenient use in
    load-update-save flows.
    """
    timestamp = float(timestamp or time.time())
    state.setdefault("version", 1)
    bucket = state.setdefault("memories", {})
    for memory in memories:
        key = memory_key(memory)
        meta = bucket.setdefault(key, {})
        meta["last_accessed"] = timestamp
        meta["access_count"] = int(meta.get("access_count") or 0) + 1
    return state
