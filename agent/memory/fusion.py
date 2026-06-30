"""Cross-provider memory fusion — deduplication, priority weighting, and merge.

Stage 3 of the memory relevance pipeline:
  Multi-Provider Recall → Fusion → Relevance Gating

Takes scored candidates from all providers and produces a unified ranked list.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from agent.memory.scored_memory import ScoredMemory

logger = logging.getLogger(__name__)

# Default trust scores for known providers.
# Builtin (user-curated) is highest; observational providers get lower trust.
PROVIDER_TRUST: Dict[str, float] = {
    "builtin": 1.0,
    "holographic": 0.85,
    "hindsight": 0.75,
    "supermemory": 0.8,
    "byterover": 0.7,
}

# Near-duplicate detection thresholds
JACCARD_DUP_THRESHOLD = 0.7   # token overlap ratio for near-dup
EMBEDDING_DUP_THRESHOLD = 0.85  # cosine sim threshold (future phase)


def set_provider_trust(provider: str, trust: float) -> None:
    """Override a provider's trust score at runtime (from config)."""
    PROVIDER_TRUST[provider] = max(0.0, min(1.0, trust))


def deduplicate(candidates: List[ScoredMemory]) -> List[ScoredMemory]:
    """Remove exact and near-duplicate entries, keeping the highest-scored.

    Three-stage dedup:
      1. Exact match (normalised content)
      2. Jaccard token overlap > 0.7
      3. Embedding similarity > 0.85 (stub — requires Phase 2)

    Returns a deduplicated list preserving score order.
    """
    if not candidates:
        return []

    # Sort by score descending so highest-scored version of any dup survives
    sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

    seen_exact: set[str] = set()
    seen_stems: List[str] = []
    result: List[ScoredMemory] = []

    for c in sorted_candidates:
        key = c.content.lower().strip()

        # Exact match: normalised content identical
        if key in seen_exact:
            logger.debug("Dedup: exact match dropped '%s...'", key[:60])
            continue

        # Near-duplicate: Jaccard token overlap
        is_dup = False
        tokens_c = set(key.split())
        if tokens_c:
            for existing in seen_stems:
                tokens_e = set(existing.split())
                if not tokens_e:
                    continue
                intersection = tokens_c & tokens_e
                union = tokens_c | tokens_e
                jaccard = len(intersection) / max(len(union), 1)
                if jaccard >= JACCARD_DUP_THRESHOLD:
                    logger.debug(
                        "Dedup: Jaccard=%.2f dropped '%s...' keeping '%s...'",
                        jaccard, c.content[:50], existing[:50],
                    )
                    is_dup = True
                    break

        if not is_dup:
            seen_exact.add(key)
            seen_stems.append(key)
            result.append(c)

    return result


def apply_provider_weights(candidates: List[ScoredMemory]) -> List[ScoredMemory]:
    """Scale candidate scores by their provider's trust weight.

    Provider trust is how much we trust a third-party store's relevance
    assessment vs. the user-curated builtin store.
    """
    for c in candidates:
        trust = PROVIDER_TRUST.get(c.provider, 0.5)
        c.score = c.score * trust
    return candidates


def fuse(
    *provider_candidate_lists: List[ScoredMemory],
    provider_trust_override: Optional[Dict[str, float]] = None,
) -> List[ScoredMemory]:
    """Fuse scored candidates from multiple providers into a single ranked list.

    Pipeline:
      1. Flatten all provider lists
      2. Apply provider trust weights
      3. Deduplicate (exact + near-duplicate)
      4. Sort by composite score descending
      5. Re-index entries

    Args:
        provider_candidate_lists: One or more lists of ScoredMemory from
            individual providers.
        provider_trust_override: Optional override for provider trust scores.

    Returns:
        Unified ranked list of deduplicated, weighted ScoredMemory entries.
    """
    if provider_trust_override:
        for p_name, p_trust in provider_trust_override.items():
            set_provider_trust(p_name, p_trust)

    # Flatten
    flat: List[ScoredMemory] = []
    for lst in provider_candidate_lists:
        flat.extend(lst)

    if not flat:
        return []

    # Apply provider trust weights
    flat = apply_provider_weights(flat)

    # Deduplicate
    deduped = deduplicate(flat)

    # Sort by composite score descending
    deduped.sort(key=lambda x: x.score, reverse=True)

    # Re-index
    for i, c in enumerate(deduped):
        c.entry_index = i

    return deduped
