"""
Semantic Recall Engine for cognitive memory.

Provides composite-scored retrieval combining:
  - Semantic similarity (cosine distance between embeddings)
  - Recency (exponential decay based on last access time)
  - Importance (stored importance score from the memory)

The engine wraps CognitiveStore and an Embedder to offer a single
`recall()` method that returns the most relevant memories for a query.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cognitive_memory.embeddings import Embedder, cosine_similarity
from cognitive_memory.store import CognitiveStore, Memory, ScoredMemory

logger = logging.getLogger(__name__)

# Default weights for composite scoring
DEFAULT_SIMILARITY_WEIGHT = 0.50
DEFAULT_RECENCY_WEIGHT = 0.30
DEFAULT_IMPORTANCE_WEIGHT = 0.20

# Recency half-life in days: recency = 0.5^(age_days / half_life)
DEFAULT_RECENCY_HALF_LIFE = 30.0


@dataclass
class RecallConfig:
    """Configuration for the recall engine."""

    similarity_weight: float = DEFAULT_SIMILARITY_WEIGHT
    recency_weight: float = DEFAULT_RECENCY_WEIGHT
    importance_weight: float = DEFAULT_IMPORTANCE_WEIGHT
    recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE
    similarity_threshold: float = 0.3
    default_limit: int = 10

    def __post_init__(self):
        total = self.similarity_weight + self.recency_weight + self.importance_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.3f} "
                f"(sim={self.similarity_weight}, rec={self.recency_weight}, "
                f"imp={self.importance_weight})"
            )


def compute_recency(last_accessed: float, now: float, half_life_days: float) -> float:
    """
    Compute recency score using exponential decay.

    Formula: recency = 0.5 ^ (age_days / half_life_days)

    Returns a value between 0.0 and 1.0, where 1.0 means just accessed.
    """
    age_seconds = now - last_accessed
    if age_seconds <= 0:
        return 1.0

    age_days = age_seconds / 86400.0
    return 0.5 ** (age_days / half_life_days)


def composite_score(
    similarity: float,
    recency: float,
    importance: float,
    weights: RecallConfig,
) -> float:
    """
    Compute weighted composite score.

    score = w_sim * similarity + w_rec * recency + w_imp * importance
    """
    return (
        weights.similarity_weight * similarity
        + weights.recency_weight * recency
        + weights.importance_weight * importance
    )


class RecallEngine:
    """
    Semantic recall engine with composite scoring.

    Combines vector similarity search with recency and importance
    to rank memories by overall relevance.
    """

    def __init__(
        self,
        store: CognitiveStore,
        embedder: Embedder,
        config: Optional[RecallConfig] = None,
    ):
        self._store = store
        self._embedder = embedder
        self._config = config or RecallConfig()

    @property
    def config(self) -> RecallConfig:
        return self._config

    def recall(
        self,
        query: str,
        limit: Optional[int] = None,
        scope: Optional[str] = None,
        categories: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        include_forgotten: bool = False,
    ) -> List[ScoredMemory]:
        """
        Recall memories relevant to a natural language query.

        Steps:
          1. Embed the query text
          2. Find semantically similar memories from the store
          3. Apply scope and category filters
          4. Compute composite score (similarity + recency + importance)
          5. Re-rank by composite score and return top results

        Args:
            query: Natural language query text
            limit: Max results to return (default from config)
            scope: Filter to memories under this scope prefix
            categories: Filter to memories with any of these categories
            similarity_threshold: Min cosine similarity (default from config)
            include_forgotten: Include soft-deleted memories

        Returns:
            List of ScoredMemory sorted by composite score (descending)
        """
        effective_limit = limit or self._config.default_limit
        threshold = similarity_threshold or self._config.similarity_threshold

        # Step 1: Embed the query
        try:
            query_embedding = self._embedder.embed_text(query)
        except Exception as e:
            logger.warning("Failed to embed query: %s", e)
            return []

        # Step 2: Get similar memories from store
        # Fetch more than needed since we'll filter and re-rank
        fetch_limit = effective_limit * 3
        candidates = self._store.search_similar(
            query_embedding,
            threshold=threshold,
            limit=fetch_limit,
            include_forgotten=include_forgotten,
        )

        if not candidates:
            return []

        # Step 3: Apply filters
        filtered = candidates
        if scope:
            filtered = [
                sm for sm in filtered
                if sm.memory.scope.startswith(scope)
            ]
        if categories:
            cat_set = set(categories)
            filtered = [
                sm for sm in filtered
                if cat_set.intersection(sm.memory.categories)
            ]

        if not filtered:
            return []

        # Step 4: Compute composite scores
        now = time.time()
        scored = []
        for sm in filtered:
            recency = compute_recency(
                sm.memory.last_accessed, now, self._config.recency_half_life_days
            )
            score = composite_score(
                similarity=sm.similarity,
                recency=recency,
                importance=sm.memory.importance,
                weights=self._config,
            )

            match_reasons = list(sm.match_reasons)
            if recency > 0.8:
                match_reasons.append("recent")
            if sm.memory.importance > 0.7:
                match_reasons.append("important")

            scored.append(ScoredMemory(
                memory=sm.memory,
                score=score,
                similarity=sm.similarity,
                match_reasons=match_reasons,
            ))

        # Step 5: Sort by composite score descending
        scored.sort(key=lambda s: s.score, reverse=True)

        # Record access for returned memories
        results = scored[:effective_limit]
        for sm in results:
            self._store.record_access(sm.memory.id)

        return results

    def recall_by_embedding(
        self,
        query_embedding: List[float],
        limit: Optional[int] = None,
        scope: Optional[str] = None,
        categories: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        include_forgotten: bool = False,
    ) -> List[ScoredMemory]:
        """
        Recall memories using a pre-computed embedding vector.

        Same as recall() but skips the embedding step. Useful when
        the caller already has an embedding (e.g., from a batch operation).
        """
        effective_limit = limit or self._config.default_limit
        threshold = similarity_threshold or self._config.similarity_threshold

        fetch_limit = effective_limit * 3
        candidates = self._store.search_similar(
            query_embedding,
            threshold=threshold,
            limit=fetch_limit,
            include_forgotten=include_forgotten,
        )

        if not candidates:
            return []

        filtered = candidates
        if scope:
            filtered = [
                sm for sm in filtered
                if sm.memory.scope.startswith(scope)
            ]
        if categories:
            cat_set = set(categories)
            filtered = [
                sm for sm in filtered
                if cat_set.intersection(sm.memory.categories)
            ]

        if not filtered:
            return []

        now = time.time()
        scored = []
        for sm in filtered:
            recency = compute_recency(
                sm.memory.last_accessed, now, self._config.recency_half_life_days
            )
            score = composite_score(
                similarity=sm.similarity,
                recency=recency,
                importance=sm.memory.importance,
                weights=self._config,
            )

            match_reasons = list(sm.match_reasons)
            if recency > 0.8:
                match_reasons.append("recent")
            if sm.memory.importance > 0.7:
                match_reasons.append("important")

            scored.append(ScoredMemory(
                memory=sm.memory,
                score=score,
                similarity=sm.similarity,
                match_reasons=match_reasons,
            ))

        scored.sort(key=lambda s: s.score, reverse=True)

        results = scored[:effective_limit]
        for sm in results:
            self._store.record_access(sm.memory.id)

        return results

    def add_and_recall(
        self,
        content: str,
        scope: str = "/",
        importance: float = 0.5,
        categories: Optional[List[str]] = None,
        recall_limit: int = 5,
    ) -> Dict:
        """
        Add a new memory and find related existing memories.

        This is the primary operation for cognitive encoding:
        embed the content, store it, then find related memories
        that might need consolidation or contradiction detection.

        Returns:
            Dict with 'memory_id' and 'related' (list of ScoredMemory)
        """
        # Embed the content
        try:
            embedding = self._embedder.embed_text(content)
        except Exception as e:
            logger.warning("Failed to embed content for add_and_recall: %s", e)
            # Store without embedding
            memory_id = self._store.add_memory(
                content, scope=scope, importance=importance, categories=categories
            )
            return {"memory_id": memory_id, "related": []}

        # Store with embedding
        memory_id = self._store.add_memory(
            content,
            embedding=embedding,
            scope=scope,
            importance=importance,
            categories=categories,
        )

        # Find related memories (exclude the one we just added)
        related = self.recall_by_embedding(
            embedding,
            limit=recall_limit + 1,  # +1 to account for self-match
            scope=scope if scope != "/" else None,
        )

        # Filter out the memory we just added
        related = [sm for sm in related if sm.memory.id != memory_id][:recall_limit]

        return {"memory_id": memory_id, "related": related}
