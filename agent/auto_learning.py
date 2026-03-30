from __future__ import annotations

import json
from typing import Any

VALID_CATEGORIES = {"memory", "skill", "unknown"}


def build_auto_learning_review_prompt(
    *,
    allow_memory: bool,
    allow_skills: bool,
    min_tool_iterations: int,
    promotion_threshold: float,
) -> str:
    allowed_categories = []
    if allow_memory:
        allowed_categories.append('"memory"')
    if allow_skills:
        allowed_categories.append('"skill"')
    if not allowed_categories:
        allowed_categories.append('"unknown"')

    return (
        "Review the conversation and propose durable staged learnings only when justified.\n\n"
        f"This review is intended for tool-heavy work (minimum {min_tool_iterations} tool iterations).\n"
        f"High-confidence candidates may later be promoted at threshold {promotion_threshold}.\n\n"
        "Return strict JSON with this shape only:\n"
        '{"candidates": [{"category": "memory", "summary": "...", "confidence": 0.93, '
        '"reason": "...", "target": "...", "payload": {}}]}\n\n'
        f"Allowed categories for this review: {', '.join(allowed_categories)}.\n"
        "Rules:\n"
        "- prefer an empty candidates list over weak guesses\n"
        "- only propose memory for durable cross-session facts\n"
        "- only propose skill for reusable non-trivial procedures\n"
        "- include confidence and reason for every candidate\n"
    )



def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    category = str(candidate.get("category", "unknown")).strip().lower()
    if category not in VALID_CATEGORIES:
        category = "unknown"

    summary = " ".join(str(candidate.get("summary", "")).split()).strip()

    try:
        confidence = float(candidate.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    normalized = {
        "category": category,
        "summary": summary,
        "confidence": confidence,
        "reason": str(candidate.get("reason", "")).strip(),
        "target": str(candidate.get("target", "")).strip(),
        "payload": candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {},
    }
    return normalized



def parse_auto_learning_review(text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return []

    raw_candidates = parsed.get("candidates") if isinstance(parsed, dict) else None
    if not isinstance(raw_candidates, list):
        return []

    normalized_candidates = []
    for candidate in raw_candidates:
        if isinstance(candidate, dict):
            normalized_candidates.append(normalize_candidate(candidate))
    return normalized_candidates



def should_promote_candidate(candidate: dict[str, Any], threshold: float) -> bool:
    try:
        confidence = float(candidate.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    return confidence >= threshold
