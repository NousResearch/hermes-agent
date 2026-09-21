"""
Memory hygiene (P2-8 — longevity and quality-of-life).

Provides decay/pruning policies for Mnemosyne working memory and
contradiction detection for facts being promoted to durable storage.

Background
----------
Mnemosyne holds 5,850 facts + 8,907 working entries with no decay
logic.  Stale and contradictory facts increasingly skew agent
decisions; unbounded growth inflates retrieval cost and noise.

This module provides:

1. **Decay policy** — importance-weighted aging of working memory
   entries.  Low-importance entries that haven't been accessed in
   N days are pruned.  Medium-importance entries decay to low after
   M days.  High-importance entries are preserved.

2. **Pruning** — batch removal of decayed entries below a configurable
   importance threshold.

3. **Contradiction detection** — before a fact is promoted from
   working memory to durable storage (~/brain/), check for
   contradictory facts already in the durable store.

Usage
-----
    from hermes_cli.memory_hygiene import prune_stale_entries, detect_contradictions

    # Run daily via cron
    pruned = prune_stale_entries(
        low_importance_days=7,
        medium_decay_days=30,
    )

    # Check before memory-promotion pipeline
    conflicts = detect_contradictions(
        new_fact="User prefers dark mode",
        target_store="~/brain/references/preferences.md",
    )
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

# Days before a low-importance entry is pruned
DEFAULT_LOW_IMPORTANCE_TTL_DAYS = 7

# Days before a medium-importance entry decays to low
DEFAULT_MEDIUM_DECAY_DAYS = 30

# Days before a high-importance entry decays to medium
DEFAULT_HIGH_DECAY_DAYS = 90

# Importance thresholds
HIGH_IMPORTANCE = 0.70
MEDIUM_IMPORTANCE = 0.40
LOW_IMPORTANCE = 0.0

# Minimum importance to survive pruning
MIN_SURVIVAL_IMPORTANCE = 0.20

# Contradiction detection similarity threshold (0.0-1.0)
CONTRADICTION_SIMILARITY = 0.70


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A simplified memory entry for hygiene operations."""
    memory_id: str
    content: str
    importance: float = 0.5
    created_at: str = ""  # ISO timestamp
    last_accessed_at: str = ""  # ISO timestamp
    source: str = "unknown"
    veracity: str = "unknown"

    def age_days(self) -> int:
        """Days since creation (or last access, whichever is newer)."""
        reference = self.last_accessed_at or self.created_at
        if not reference:
            return 0
        try:
            created = datetime.fromisoformat(
                reference.replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            return (now - created).days
        except (ValueError, TypeError):
            return 0


@dataclass
class PruneResult:
    """Outcome of a prune operation."""
    total_entries: int
    pruned_count: int
    decayed_count: int  # importance downgraded but kept
    preserved_count: int
    pruned_ids: list[str] = field(default_factory=list)


@dataclass
class ContradictionResult:
    """Outcome of contradiction detection."""
    has_conflict: bool
    conflicting_entries: list[str] = field(default_factory=list)
    similarity_scores: list[float] = field(default_factory=list)
    recommendation: str = ""  # "promote", "review", or "block"


# ---------------------------------------------------------------------------
# Decay and pruning
# ---------------------------------------------------------------------------


def compute_decayed_importance(
    entry: MemoryEntry,
    *,
    low_ttl_days: int = DEFAULT_LOW_IMPORTANCE_TTL_DAYS,
    medium_decay_days: int = DEFAULT_MEDIUM_DECAY_DAYS,
    high_decay_days: int = DEFAULT_HIGH_DECAY_DAYS,
) -> tuple[float, bool]:
    """Compute the decayed importance of a memory entry.

    Returns:
        (new_importance, should_prune) tuple.
        should_prune is True when the entry should be removed entirely.
    """
    age = entry.age_days()
    importance = entry.importance

    if importance < MIN_SURVIVAL_IMPORTANCE:
        return 0.0, True

    if importance >= HIGH_IMPORTANCE:
        # High-importance decays to medium after high_decay_days
        if age > high_decay_days:
            return MEDIUM_IMPORTANCE, False
        return importance, False

    elif importance >= MEDIUM_IMPORTANCE:
        # Medium decays to low after medium_decay_days
        if age > medium_decay_days:
            return LOW_IMPORTANCE + 0.05, False
        return importance, False

    elif importance >= LOW_IMPORTANCE:
        # Low importance pruned after low_ttl_days
        if age > low_ttl_days:
            return 0.0, True
        return importance, False

    return importance, False


def prune_stale_entries(
    entries: list[MemoryEntry],
    *,
    low_importance_days: int = DEFAULT_LOW_IMPORTANCE_TTL_DAYS,
    medium_decay_days: int = DEFAULT_MEDIUM_DECAY_DAYS,
    high_decay_days: int = DEFAULT_HIGH_DECAY_DAYS,
    dry_run: bool = False,
) -> PruneResult:
    """Prune stale working memory entries.

    Args:
        entries: All working memory entries.
        low_importance_days: TTL for low-importance entries.
        medium_decay_days: Days before medium decays to low.
        high_decay_days: Days before high decays to medium.
        dry_run: If True, compute results without mutating.

    Returns:
        PruneResult with counts and IDs of pruned entries.
    """
    pruned_ids = []
    decayed = 0
    preserved = 0

    for entry in entries:
        new_importance, should_prune = compute_decayed_importance(
            entry,
            low_ttl_days=low_importance_days,
            medium_decay_days=medium_decay_days,
            high_decay_days=high_decay_days,
        )

        if should_prune:
            pruned_ids.append(entry.memory_id)
        elif new_importance < entry.importance:
            decayed += 1
            preserved += 1
        else:
            preserved += 1

    return PruneResult(
        total_entries=len(entries),
        pruned_count=len(pruned_ids),
        decayed_count=decayed,
        preserved_count=preserved,
        pruned_ids=pruned_ids,
    )


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------


def detect_contradictions(
    new_fact: str,
    existing_facts: list[str],
    *,
    similarity_threshold: float = CONTRADICTION_SIMILARITY,
) -> ContradictionResult:
    """Check a new fact for contradictions against existing facts.

    Uses a lightweight keyword-overlap similarity heuristic.  For
    production use, an LLM judge (P2-2 harness) should validate
    flagged contradictions before blocking promotion.

    Args:
        new_fact: The fact being considered for promotion.
        existing_facts: Existing facts in the target store.
        similarity_threshold: Overlap ratio that triggers a flag.

    Returns:
        ContradictionResult with conflict details and recommendation.
    """
    new_tokens = set(_tokenize(new_fact))
    if not new_tokens:
        return ContradictionResult(
            has_conflict=False,
            recommendation="promote",
        )

    conflicts = []
    scores = []

    for i, existing in enumerate(existing_facts):
        existing_tokens = set(_tokenize(existing))
        if not existing_tokens:
            continue

        # Keyword overlap ratio
        overlap = new_tokens & existing_tokens
        # Jaccard-like similarity
        union = new_tokens | existing_tokens
        similarity = len(overlap) / len(union) if union else 0.0

        # Check for contradiction signals (negation patterns)
        has_contradiction_signal = _has_negation_conflict(new_fact, existing)

        if similarity >= similarity_threshold or has_contradiction_signal:
            conflicts.append(f"fact-{i}")
            scores.append(similarity)

    if not conflicts:
        return ContradictionResult(
            has_conflict=False,
            recommendation="promote",
        )

    if len(conflicts) == 1 and max(scores) < 0.85:
        return ContradictionResult(
            has_conflict=True,
            conflicting_entries=conflicts,
            similarity_scores=scores,
            recommendation="review",
        )
    else:
        return ContradictionResult(
            has_conflict=True,
            conflicting_entries=conflicts,
            similarity_scores=scores,
            recommendation="block",
        )


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens for comparison."""
    import re
    return [w.lower() for w in re.findall(r'\b\w{3,}\b', text)]


def _has_negation_conflict(new_text: str, existing_text: str) -> bool:
    """Detect explicit negation contradictions (e.g., 'not X' vs 'is X')."""
    negation_patterns = [
        ("prefers", "does not prefer"),
        ("uses", "does not use"),
        ("is ", "is not "),
        ("enabled", "disabled"),
        ("yes", "no"),
        ("always", "never"),
        ("should", "should not"),
        ("must", "must not"),
    ]
    new_lower = new_text.lower()
    existing_lower = existing_text.lower()

    for positive, negative in negation_patterns:
        if positive in new_lower and negative in existing_lower:
            return True
        if negative in new_lower and positive in existing_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Memory promotion gate (integration with memory-promotion pipeline)
# ---------------------------------------------------------------------------


def memory_promotion_gate(
    new_facts: list[str],
    existing_store_path: str,
    *,
    similarity_threshold: float = CONTRADICTION_SIMILARITY,
) -> list[tuple[str, str]]:
    """Check a batch of facts before promotion to durable storage.

    Args:
        new_facts: Facts being promoted.
        existing_store_path: Path to the target storage file.
        similarity_threshold: Overlap threshold for flagging.

    Returns:
        List of (fact, recommendation) tuples.  Each fact is classified
        as "promote", "review", or "block".
    """
    # Load existing facts
    existing = []
    expanded_path = os.path.expanduser(existing_store_path)
    if os.path.exists(expanded_path):
        try:
            with open(expanded_path) as f:
                # Read line-by-line — most brain files are markdown
                # with one fact per bullet or paragraph
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        existing.append(stripped)
        except (OSError, UnicodeDecodeError):
            pass

    results = []
    for fact in new_facts:
        detection = detect_contradictions(
            fact, existing, similarity_threshold=similarity_threshold,
        )
        results.append((fact, detection.recommendation))

    return results


# ---------------------------------------------------------------------------
# Cron-friendly batch interface
# ---------------------------------------------------------------------------


def memory_hygiene_report(
    entries: list[MemoryEntry],
    *,
    dry_run: bool = True,
) -> dict:
    """Generate a human-readable memory hygiene report.

    Intended for use by a daily cron job.  Includes pruning
    recommendations and contradiction stats.
    """
    prune = prune_stale_entries(entries, dry_run=dry_run)

    # Group by importance for visibility
    by_importance = {"high": 0, "medium": 0, "low": 0, "prune_candidate": 0}
    for entry in entries:
        _, should_prune = compute_decayed_importance(entry)
        if should_prune:
            by_importance["prune_candidate"] += 1
        elif entry.importance >= HIGH_IMPORTANCE:
            by_importance["high"] += 1
        elif entry.importance >= MEDIUM_IMPORTANCE:
            by_importance["medium"] += 1
        else:
            by_importance["low"] += 1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_entries": prune.total_entries,
        "by_importance": by_importance,
        "prune_recommendation": {
            "prune_count": prune.pruned_count,
            "decay_count": prune.decayed_count,
            "preserved_count": prune.preserved_count,
            "dry_run": dry_run,
        },
    }
