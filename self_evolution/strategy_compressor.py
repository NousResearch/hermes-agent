"""
Self Evolution Plugin — Strategy Compressor
=============================================

Compresses and merges redundant strategy rules into concise hints.

Called after dream consolidation to keep strategies.json compact.
Each hint_text must be ≤ 30 chars; strategies without conditions are
either merged into conditional rules or discarded.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Maximum allowed length for hint_text (characters)
MAX_HINT_LENGTH = 30

# Keyword clusters used to group similar strategies
_CLUSTERS: List[Dict[str, Any]] = [
    {
        "keywords": ["bash", "路径", "path", "校验", "预检", "验证", "存在"],
        "hint": "bash前先read验证路径",
        "condition": {"field": "tool_name", "operator": "contains", "pattern": "bash"},
    },
    {
        "keywords": ["api", "调试", "debug", "降级", "只读", "探查"],
        "hint": "API失败时降级只读探查",
        "condition": {"field": "task_type", "operator": "contains", "pattern": "api"},
    },
    {
        "keywords": ["browser", "浏览器", "timeout", "超时", "网页"],
        "hint": "浏览器操作设置超时保护",
        "condition": {"field": "tool_name", "operator": "contains", "pattern": "browser"},
    },
    {
        "keywords": ["重试", "retry", "浪费", "重复", "循环"],
        "hint": "避免重复重试相同操作",
        "condition": {},
    },
]


def compress_strategies(rules: List[dict]) -> List[dict]:
    """Compress strategy rules by merging similar ones.

    Returns a new list of rules with:
    - Duplicate hint_texts removed
    - Similar rules merged into cluster summaries
    - hint_text truncated to MAX_HINT_LENGTH
    - Non-matching rules dropped if they have no conditions
    """
    if not rules:
        return []

    # Deduplicate by hint_text
    seen_hints: set[str] = set()
    unique: list[dict] = []
    for r in rules:
        key = r.get("hint_text", "").strip().lower()
        if key and key not in seen_hints:
            seen_hints.add(key)
            unique.append(r)

    # Cluster similar rules
    clustered = _cluster_rules(unique)

    # Enforce constraints: hint_text ≤ 30 chars, must have conditions
    result: list[dict] = []
    for r in clustered:
        hint = r.get("hint_text", "").strip()
        conditions = r.get("conditions", [])

        # Skip rules without conditions (they won't be injected anyway)
        if not conditions:
            logger.debug("Dropping unconditioned strategy: %s", hint[:40])
            continue

        # Truncate hint if needed
        if len(hint) > MAX_HINT_LENGTH:
            hint = hint[:MAX_HINT_LENGTH]
            r["hint_text"] = hint

        result.append(r)

    # Also keep any manual/default rules that already have conditions
    for r in unique:
        if r.get("source") in ("manual", "default") and r.get("conditions"):
            if r not in result:
                hint = r.get("hint_text", "").strip()
                if len(hint) > MAX_HINT_LENGTH:
                    r["hint_text"] = hint[:MAX_HINT_LENGTH]
                result.append(r)

    logger.info("Compressed strategies: %d → %d rules", len(rules), len(result))
    return result


def _cluster_rules(rules: list[dict]) -> list[dict]:
    """Group rules by keyword clusters and merge each group into one rule."""
    matched_indices: set[int] = set()
    merged: list[dict] = []

    for cluster in _CLUSTERS:
        group: list[dict] = []
        for i, r in enumerate(rules):
            text = f"{r.get('name', '')} {r.get('hint_text', '')}".lower()
            if any(kw in text for kw in cluster["keywords"]):
                group.append(r)
                matched_indices.add(i)

        if not group:
            continue

        # Merge group into one rule
        first = group[0]
        condition = cluster.get("condition")
        merged_rule = {
            "id": first.get("id", ""),
            "name": cluster["hint"],
            "type": "learned",
            "description": cluster["hint"],
            "hint_text": cluster["hint"],
            "conditions": [condition] if condition else [],
            "severity": "medium",
            "enabled": True,
            "source": "learned",
            "created_at": first.get("created_at", 0),
        }
        merged.append(merged_rule)

    # Add unmatched rules as-is
    for i, r in enumerate(rules):
        if i not in matched_indices:
            merged.append(r)

    return merged
