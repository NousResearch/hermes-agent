"""Deterministic classification heuristics."""

from __future__ import annotations

import re

SEMANTIC_HINTS = (
    "definition",
    "stable facts",
    "principle",
    "rule",
    "concept",
    "是什么",
    "定义",
)

EPISODIC_HINTS = (
    "meeting",
    "kickoff",
    "decision",
    "project",
    "transcript",
    "讨论",
    "会议",
    "项目",
)

SEMANTIC_SECONDARY_HINTS = ("decision", "scope", "principle", "rule", "决定", "范围", "原则")

DATE_PATTERN = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")


def classify_text(content: str, source_type: str | None = None) -> dict[str, bool]:
    """Classify content as semantic, episodic, or both."""

    lowered = content.lower()
    semantic = any(token in lowered for token in SEMANTIC_HINTS)
    episodic = any(token in lowered for token in EPISODIC_HINTS) or bool(
        DATE_PATTERN.search(content)
    )
    if not semantic and any(token in lowered for token in SEMANTIC_SECONDARY_HINTS):
        semantic = True

    if source_type:
        lowered_type = source_type.lower()
        if lowered_type in {"meeting", "transcript", "chat", "project"}:
            episodic = True
        if lowered_type in {"concept", "note", "research"}:
            semantic = True

    if not semantic and not episodic:
        semantic = True

    return {"semantic": semantic, "episodic": episodic}


def detect_date(content: str) -> str | None:
    """Extract an ISO-like date from text when present."""

    match = DATE_PATTERN.search(content)
    if not match:
        return None
    return match.group(0).replace("/", "-")
