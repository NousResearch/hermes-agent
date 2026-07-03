"""Semantic retrieval for kv-memory provider.

Performs cosine similarity search over stored embeddings with:
  - Temporal decay: newer turns weighted higher
  - Causal boosting: same-session and linked-session results boosted
  - MMR diversity reranking: avoids returning 5 copies of the same memory
  - Configurable thresholds and top-K

Uses brute-force cosine similarity (linear scan) by default.
When sqlite-vec is available, ANN index search is used for speedup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import KVMemoryConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Cosine similarity
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two float32 vectors. Safe against overflow."""
    a_f64 = a.astype(np.float64)
    b_f64 = b.astype(np.float64)
    a_norm = np.linalg.norm(a_f64)
    b_norm = np.linalg.norm(b_f64)
    if a_norm < 1e-12 or b_norm < 1e-12:
        return 0.0
    return float(np.dot(a_f64 / a_norm, b_f64 / b_norm))


def batch_cosine_similarity(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between query and all candidates.

    Args:
        query: (D,) float32 vector
        candidates: (N, D) float32 matrix

    Returns:
        (N,) float64 similarity scores
    """
    query_f64 = query.astype(np.float64)
    candidates_f64 = candidates.astype(np.float64)

    q_norm = query_f64 / (np.linalg.norm(query_f64) + 1e-12)
    c_norms = candidates_f64 / (
        np.linalg.norm(candidates_f64, axis=1, keepdims=True) + 1e-12
    )

    return np.dot(c_norms, q_norm)


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal decay
# ═══════════════════════════════════════════════════════════════════════════════

def compute_temporal_weight(
    timestamp: float,
    now: float,
    half_life_days: float = 7.0,
) -> float:
    """Compute decay weight: 0.5^(age/half_life)."""
    if half_life_days <= 0:
        return 1.0
    age_days = (now - timestamp) / 86400.0
    return float(0.5 ** (age_days / half_life_days))


# ═══════════════════════════════════════════════════════════════════════════════
# MMR diversity reranking
# ═══════════════════════════════════════════════════════════════════════════════

def mmr_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: List[np.ndarray],
    candidate_scores: List[float],
    k: int,
    lambda_param: float = 0.7,
) -> List[int]:
    """Maximal Marginal Relevance reranking.

    Balances relevance to query (lambda) against novelty (1 - lambda).
    Returns indices into the candidate list, in reranked order.

    lambda=1.0 → pure relevance ranking
    lambda=0.0 → pure diversity
    lambda=0.7 → balance (default)
    """
    if len(candidate_scores) <= k:
        return list(range(len(candidate_scores)))

    selected: List[int] = []
    remaining = list(range(len(candidate_scores)))

    while len(selected) < k and remaining:
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            # Relevance term
            relevance = candidate_scores[idx]

            # Diversity term: max similarity to any already-selected item
            if selected:
                max_sim = max(
                    cosine_similarity(
                        candidate_embeddings[idx],
                        candidate_embeddings[s],
                    )
                    for s in selected
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval engine
# ═══════════════════════════════════════════════════════════════════════════════

class KVRetriever:
    """Semantic search over stored kv-memory embeddings."""

    def __init__(
        self,
        db,  # KVMemoryDB
        config: KVMemoryConfig,
    ):
        self._db = db
        self._config = config

    def retrieve(
        self,
        query_embedding: np.ndarray,
        current_session_id: str = "",
        k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Full retrieval pipeline.

        1. Semantic search (cosine similarity over all stored embeddings)
        2. Temporal decay weighting
        3. Causal boost (same-session and linked sessions get bonus)
        4. MMR diversity reranking
        5. Threshold filtering

        Returns list of result dicts with keys:
          turn_id, session_id, similarity, summary_text, timestamp, score
        """
        k = k or self._config.top_k
        min_sim = self._config.min_similarity

        # ── Step 1: Semantic search ───────────────────────────────
        all_embeddings = self._db.get_all_embeddings(dequantize=True)
        if not all_embeddings:
            return []

        turn_ids = [t[0] for t in all_embeddings]
        embeddings = np.stack([t[1] for t in all_embeddings], axis=0)
        metas = [t[2] for t in all_embeddings]

        # Batch cosine similarity
        similarities = batch_cosine_similarity(query_embedding, embeddings)
        now = __import__("time").time()

        # ── Step 2: Temporal decay ────────────────────────────────
        scores = []
        for i, sim in enumerate(similarities):
            ts = metas[i].get("timestamp", now)
            temporal_weight = compute_temporal_weight(
                ts, now, self._config.temporal_decay_half_life
            )
            score = float(sim) * temporal_weight
            scores.append(score)

        # ── Step 3: Causal boost ──────────────────────────────────
        if current_session_id and self._config.causal_boost > 0:
            # Boost turns from the same session
            for i, meta in enumerate(metas):
                if meta.get("session_id") == current_session_id:
                    scores[i] += self._config.causal_boost

            # Boost turns from linked sessions
            linked = self._db.get_linked_sessions(
                current_session_id, min_similarity=min_sim
            )
            linked_ids = set()
            for link in linked:
                if link["source_session_id"] != current_session_id:
                    linked_ids.add(link["source_session_id"])
                if link["target_session_id"] != current_session_id:
                    linked_ids.add(link["target_session_id"])

            for i, meta in enumerate(metas):
                if meta.get("session_id") in linked_ids:
                    scores[i] += self._config.causal_boost * 0.5

        # ── Step 4: MMR diversity reranking ───────────────────────
        if self._config.diversity_lambda < 1.0 and len(embeddings) > k:
            lambda_param = self._config.diversity_lambda
            reranked_indices = mmr_rerank(
                query_embedding,
                [embeddings[j] for j in range(len(embeddings))],
                scores,
                k,
                lambda_param=lambda_param,
            )
        else:
            # Pure relevance ranking
            reranked_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:k]

        # ── Step 5: Threshold filter ──────────────────────────────
        results = []
        for idx in reranked_indices:
            if similarities[idx] < min_sim:
                continue
            results.append({
                "turn_id": turn_ids[idx],
                "session_id": metas[idx].get("session_id", ""),
                "similarity": round(float(similarities[idx]), 4),
                "score": round(scores[idx], 4),
                "summary_text": metas[idx].get("summary_text", ""),
                "timestamp": metas[idx].get("timestamp", now),
            })

        return results

    def retrieve_text(
        self,
        query_embedding: np.ndarray,
        current_session_id: str = "",
        k: int | None = None,
    ) -> str:
        """Retrieve and format results as text for system prompt injection.

        Returns empty string if no results found.
        """
        results = self.retrieve(query_embedding, current_session_id, k)
        if not results:
            return ""

        lines = ["## KV Memory (model-native semantic recall)"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r['similarity']:.2f}] {r['summary_text']}"
            )

        return "\n".join(lines)
