"""Deterministic scoring metrics for Memory v2 evals."""

from __future__ import annotations

import math


def score_source_recall(retrieved_source_refs: list[str], expected_source_refs: list[str]) -> float:
    if not expected_source_refs:
        return 1.0 if not retrieved_source_refs else 0.0
    retrieved = set(retrieved_source_refs)
    expected = set(expected_source_refs)
    return len(retrieved & expected) / len(expected)


def score_text_contains(answer: str, expected_fragments: list[str]) -> float:
    answer_text = str(answer or "")
    if not expected_fragments:
        return 1.0 if not answer_text.strip() else 0.0
    answer_lower = answer_text.lower()
    hits = sum(1 for fragment in expected_fragments if str(fragment).lower() in answer_lower)
    return hits / len(expected_fragments)


def score_irrelevant_suppression(*, should_retrieve: bool, retrieved_count: int) -> float:
    if should_retrieve:
        return 1.0 if retrieved_count > 0 else 0.0
    return 1.0 if retrieved_count == 0 else 0.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative rough estimate for English-ish text/code/YAML packets.
    # Char/4 avoids the prior 4-words-per-token undercount.
    return max(1, math.ceil(len(str(text)) / 4))
