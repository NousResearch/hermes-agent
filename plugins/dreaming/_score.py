"""
Candidate scoring for the Deep Sleep promotion gate.

Weights match the issue spec:
  relevance          30%
  frequency          24%
  query_diversity    15%
  recency            15%
  consolidation      10%
  conceptual_richness 6%

All inputs are normalised to [0.0, 1.0] before weighting.
"""
from __future__ import annotations

import math
import time
from typing import Any

_WEIGHTS = {
    "relevance": 0.30,
    "frequency": 0.24,
    "query_diversity": 0.15,
    "recency": 0.15,
    "consolidation": 0.10,
    "conceptual_richness": 0.06,
}

# Keywords that indicate a meta-memory-management entry.
# These are routed to a skill file rather than promoted to MEMORY.md.
# Suggested by @vingeraycn in issue #25309.
_META_KEYWORDS = frozenset([
    "memory is full", "memory capacity", "memory management",
    "memory.md", "skill.md", "store in skill", "update skill",
    "memory limit", "memory overflow",
])


def is_meta_entry(text: str) -> bool:
    """Return True if this looks like a memory-management meta-entry."""
    lower = text.lower()
    return any(kw in lower for kw in _META_KEYWORDS)


def score(candidate: dict[str, Any], now: float | None = None) -> float:
    """
    Score a staging candidate dict. Returns a float in [0.0, 1.0].

    Expected candidate fields (all optional, defaulted gracefully):
      text             str   — the memory text
      relevance        float — semantic relevance score from retrieval [0,1]
      frequency        int   — times this fact appeared across sessions
      query_count      int   — distinct queries that surfaced this fact
      created_at       float — unix timestamp of first observation
      consolidation    float — similarity to existing MEMORY.md entries [0,1]
                               (lower = more novel = higher score)
      word_count       int   — approximate word count of the text
    """
    now = now or time.time()

    relevance = float(candidate.get("relevance", 0.5))

    raw_freq = int(candidate.get("frequency", 1))
    frequency = min(1.0, math.log1p(raw_freq) / math.log1p(20))

    raw_qd = int(candidate.get("query_count", 1))
    query_diversity = min(1.0, math.log1p(raw_qd) / math.log1p(10))

    age_seconds = now - float(candidate.get("created_at", now))
    age_days = age_seconds / 86400
    recency = math.exp(-age_days / 14)  # half-life ~14 days

    # consolidation: 0 = already in MEMORY.md (penalise), 1 = fully novel (reward)
    existing_sim = float(candidate.get("consolidation", 0.0))
    consolidation = 1.0 - existing_sim

    raw_wc = int(candidate.get("word_count", len(candidate.get("text", "").split())))
    conceptual_richness = min(1.0, math.log1p(raw_wc) / math.log1p(80))

    components = {
        "relevance": relevance,
        "frequency": frequency,
        "query_diversity": query_diversity,
        "recency": recency,
        "consolidation": consolidation,
        "conceptual_richness": conceptual_richness,
    }
    return sum(_WEIGHTS[k] * v for k, v in components.items())
