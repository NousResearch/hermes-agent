"""
cosmos Self-Refining RAG - RL-Optimized Retrieval

Novel Approaches:
1. Feedback Learning - Learn from retrieval success/failure
2. Strategy Mutation - Genetic evolution of retrieval strategies
3. Query Transformation - Learn optimal query rewrites
4. Dynamic Chunking - Adapt chunk sizes based on content
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Callable
from collections import defaultdict

from loguru import logger

try:
    import numpy as np
except ImportError:
    np = None

from Cosmos.rag.hybrid_retriever import HybridRetriever, RetrievalResult, Document
from Cosmos.rag.embeddings import EmbeddingManager


@dataclass
class RetrievalStrategy:
    """A retrieval strategy with evolvable parameters."""
    id: str
    semantic_weight: float = 0.7
    query_expansion: bool = True
    use_reranking: bool = False
    chunk_preference: str = "medium"  # small, medium, large
    diversity_boost: float = 0.0

    # Performance tracking
    uses: int = 0
    successes: int = 0
    avg_score: float = 0.0

    def fitness(self) -> float:
        """Calculate fitness score."""
        if self.uses == 0:
            return 0.5  # Unknown strategy
        return (self.successes / self.uses) * 0.7 + self.avg_score * 0.3

    def mutate(self) -> "RetrievalStrategy":
        """Create mutated copy."""
        return RetrievalStrategy(
            id=f"{self.id}_mut_{random.randint(0, 999):03d}",
            semantic_weight=max(0.1, min(0.9, self.semantic_weight + random.gauss(0, 0.1))),
            query_expansion=random.random() > 0.2,  # 80% chance to keep
            use_reranking=random.random() > 0.5,
            chunk_preference=random.choice(["small", "medium", "large"]),
            diversity_boost=max(0, min(0.3, self.diversity_boost + random.gauss(0, 0.05))),
        )


@dataclass
class QueryTransformation:
    """A learned query transformation."""
    original_pattern: str
    transformed_pattern: str
    success_rate: float = 0.0
    uses: int = 0


@dataclass
class RAGFeedback:
    """Feedback on a RAG result."""
    query: str
    retrieved_ids: list[str]
    selected_id: Optional[str]  # Which doc the user found useful
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


class SelfRefiningRAG:
    """
    Self-improving RAG system that learns from feedback.

    Features:
    - Tracks retrieval success/failure
    - Evolves retrieval strategies via genetic algorithms
    - Learns query transformations
    - Adapts weights based on feedback
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        feedback_window: int = 100,
        mutation_rate: float = 0.1,
        strategy_population: int = 5,
    ):
        self.retriever = retriever
        self.feedback_window = feedback_window
        self.mutation_rate = mutation_rate
        self.strategy_population = strategy_population

        # Strategy population
        self.strategies: list[RetrievalStrategy] = [
            RetrievalStrategy(id="default"),
            RetrievalStrategy(id="semantic_heavy", semantic_weight=0.9),
            RetrievalStrategy(id="keyword_heavy", semantic_weight=0.3),
            RetrievalStrategy(id="balanced", semantic_weight=0.5, diversity_boost=0.1),
            RetrievalStrategy(id="reranked", use_reranking=True),
        ]

        # Current best strategy
        self.active_strategy = self.strategies[0]

        # Feedback history
        self.feedback_history: list[RAGFeedback] = []

        # Query transformations
        self.query_transforms: list[QueryTransformation] = []

        # Query-specific weights (learned)
        self.query_type_weights: dict[str, dict] = defaultdict(lambda: {
            "semantic_weight": 0.7,
            "success_count": 0,
        })

        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "strategy_switches": 0,
            "mutations_applied": 0,
        }

    async def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> list[RetrievalResult]:
        """
        Search with the current best strategy.

        Optionally override with specific strategy.
        """
        self.stats["total_queries"] += 1

        # Use provided strategy or select best
        strat = strategy or self._select_strategy(query)

        # Apply query transformation if available
        transformed_query = self._transform_query(query)

        # Execute search with strategy parameters
        results = await self.retriever.search(
            query=transformed_query,
            top_k=top_k,
            semantic_weight=strat.semantic_weight,
            expand_query=strat.query_expansion,
        )

        # Apply diversity boost if needed
        if strat.diversity_boost > 0 and len(results) > 2:
            results = self._apply_diversity_boost(results, strat.diversity_boost)

        # Track strategy use
        strat.uses += 1

        return results

    def _select_strategy(self, query: str) -> RetrievalStrategy:
        """Select best strategy for query, with exploration."""
        # Exploration: occasionally try random strategy
        if random.random() < self.mutation_rate:
            return random.choice(self.strategies)

        # Check if we have query-type specific knowledge
        query_type = self._classify_query(query)
        if query_type in self.query_type_weights:
            weights = self.query_type_weights[query_type]
            if weights["success_count"] > 10:
                # Find matching strategy
                for strat in self.strategies:
                    if abs(strat.semantic_weight - weights["semantic_weight"]) < 0.1:
                        return strat

        # Default: use highest fitness strategy
        return max(self.strategies, key=lambda s: s.fitness())

    def _classify_query(self, query: str) -> str:
        """Classify query type."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["how", "why", "explain"]):
            return "explanatory"
        elif any(kw in query_lower for kw in ["what is", "define", "meaning"]):
            return "definitional"
        elif any(kw in query_lower for kw in ["code", "function", "implement"]):
            return "code"
        elif any(kw in query_lower for kw in ["error", "bug", "fix"]):
            return "troubleshooting"

        return "general"

    def _transform_query(self, query: str) -> str:
        """Apply learned query transformations."""
        for transform in self.query_transforms:
            if transform.success_rate > 0.6 and transform.original_pattern in query.lower():
                # Apply transformation
                query = query + " " + transform.transformed_pattern
                break

        return query

    def _apply_diversity_boost(
        self,
        results: list[RetrievalResult],
        boost: float,
    ) -> list[RetrievalResult]:
        """Boost diversity in results using MMR-like approach."""
        if not results or boost == 0:
            return results

        selected = [results[0]]
        remaining = results[1:]

        while remaining and len(selected) < len(results):
            best_idx = 0
            best_score = -float('inf')

            for i, candidate in enumerate(remaining):
                # Original score
                relevance = candidate.score

                # Diversity penalty (similarity to already selected)
                max_similarity = 0.0
                if candidate.document.embedding and np is not None:
                    cand_emb = np.array(candidate.document.embedding)
                    for sel in selected:
                        if sel.document.embedding:
                            sel_emb = np.array(sel.document.embedding)
                            sim = np.dot(cand_emb, sel_emb) / (
                                np.linalg.norm(cand_emb) * np.linalg.norm(sel_emb) + 1e-8
                            )
                            max_similarity = max(max_similarity, sim)

                # Combined score (MMR-style)
                combined = (1 - boost) * relevance - boost * max_similarity

                if combined > best_score:
                    best_score = combined
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def add_feedback(
        self,
        query: str,
        retrieved_ids: list[str],
        selected_id: Optional[str],
        success: bool,
    ):
        """
        Add feedback on retrieval results.

        This drives the learning process.
        """
        feedback = RAGFeedback(
            query=query,
            retrieved_ids=retrieved_ids,
            selected_id=selected_id,
            success=success,
        )
        self.feedback_history.append(feedback)

        # Update statistics
        if success:
            self.stats["successful_retrievals"] += 1

        # Update active strategy
        self.active_strategy.successes += int(success)
        self.active_strategy.avg_score = (
            self.active_strategy.avg_score * 0.9 +
            (1.0 if success else 0.0) * 0.1
        )

        # Update query-type weights
        query_type = self._classify_query(query)
        if success:
            weights = self.query_type_weights[query_type]
            weights["success_count"] += 1
            # Reinforce current strategy's semantic weight
            weights["semantic_weight"] = (
                weights["semantic_weight"] * 0.9 +
                self.active_strategy.semantic_weight * 0.1
            )

        # Learn query transformation if selected doc wasn't top result
        if success and selected_id and retrieved_ids:
            if selected_id != retrieved_ids[0]:
                self._learn_query_transformation(query, selected_id)

        # Trigger evolution periodically
        if len(self.feedback_history) % self.feedback_window == 0:
            self._evolve_strategies()

        # Keep bounded history
        if len(self.feedback_history) > self.feedback_window * 2:
            self.feedback_history = self.feedback_history[-self.feedback_window:]

    def _learn_query_transformation(self, query: str, successful_doc_id: str):
        """Learn a query transformation from successful retrieval."""
        # Get the successful document
        doc = self.retriever.documents.get(successful_doc_id)
        if not doc:
            return

        # Extract key terms from document that weren't in query
        query_terms = set(query.lower().split())
        doc_terms = set(doc.content.lower().split())
        new_terms = doc_terms - query_terms

        # Filter to meaningful terms
        meaningful_terms = [t for t in new_terms if len(t) > 4][:3]

        if meaningful_terms:
            transform = QueryTransformation(
                original_pattern=query[:20].lower(),
                transformed_pattern=" ".join(meaningful_terms),
            )
            self.query_transforms.append(transform)

            # Keep bounded
            if len(self.query_transforms) > 50:
                # Remove low-performing transforms
                self.query_transforms = sorted(
                    self.query_transforms,
                    key=lambda t: t.success_rate,
                    reverse=True
                )[:30]

    def _evolve_strategies(self):
        """Evolve strategy population based on feedback."""
        self.stats["mutations_applied"] += 1

        # Calculate fitness for all strategies
        for strat in self.strategies:
            _ = strat.fitness()  # Update fitness

        # Sort by fitness
        self.strategies.sort(key=lambda s: s.fitness(), reverse=True)

        # Keep top performers
        survivors = self.strategies[:self.strategy_population - 1]

        # Create mutation of best strategy
        if survivors:
            new_strategy = survivors[0].mutate()
            survivors.append(new_strategy)

        self.strategies = survivors

        # Update active strategy to best
        if self.strategies:
            self.active_strategy = self.strategies[0]
            self.stats["strategy_switches"] += 1

        logger.info(f"Strategy evolution complete. Best: {self.active_strategy.id}")

    async def get_explanation(self, query: str, results: list[RetrievalResult]) -> str:
        """Get explanation of retrieval decisions."""
        return f"""Retrieval Explanation:
- Query: {query}
- Strategy: {self.active_strategy.id}
- Semantic weight: {self.active_strategy.semantic_weight:.0%}
- Results: {len(results)}
- Top result score: {results[0].score:.3f if results else 0}
- Search type: {results[0].search_type if results else 'N/A'}
"""

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_retrievals"] /
                max(1, self.stats["total_queries"])
            ),
            "active_strategy": self.active_strategy.id,
            "strategy_count": len(self.strategies),
            "learned_transforms": len(self.query_transforms),
            "feedback_count": len(self.feedback_history),
            "strategy_fitnesses": {
                s.id: s.fitness() for s in self.strategies
            },
        }
