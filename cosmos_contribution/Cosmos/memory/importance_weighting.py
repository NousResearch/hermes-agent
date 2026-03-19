"""
COSMOS IMPORTANCE-WEIGHTED MEMORY
=====================================

Score memories by value, not just relevance.
Uses recency, frequency, emotional significance, and importance marking.

"Good news everyone! I finally know what's actually important!"
"""

import math
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from loguru import logger


@dataclass
class WeightedMemory:
    """A memory with importance weighting"""
    content: str
    key: str
    timestamp: datetime
    importance: float = 0.5           # User-set importance 0-1
    emotional_valence: float = 0.0    # -1 to 1
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class ImportanceCalculator:
    """
    Calculate importance scores for memories.

    Combines multiple factors:
    - Recency: Recent memories score higher
    - Frequency: Frequently accessed memories score higher
    - Importance: User-marked importance
    - Emotional: Emotionally significant memories retained longer
    - Relevance: Semantic similarity to query
    """

    def __init__(
        self,
        recency_half_life_days: float = 30.0,
        frequency_weight: float = 0.2,
        importance_weight: float = 0.3,
        emotional_weight: float = 0.15,
        relevance_weight: float = 0.35,
    ):
        self.recency_half_life = recency_half_life_days
        self.weights = {
            "frequency": frequency_weight,
            "importance": importance_weight,
            "emotional": emotional_weight,
            "relevance": relevance_weight,
        }

    def calculate_score(
        self,
        memory: WeightedMemory,
        query: str = None,
        semantic_similarity: float = 0.0,
    ) -> float:
        """
        Calculate overall importance score for a memory.

        Returns score 0-1.
        """
        scores = {}

        # 1. Recency score (exponential decay)
        age_days = (datetime.now() - memory.timestamp).total_seconds() / 86400
        recency = math.exp(-age_days * math.log(2) / self.recency_half_life)
        scores["recency"] = recency

        # 2. Frequency score (logarithmic)
        frequency = math.log(1 + memory.access_count) / math.log(100)  # Normalize to ~1 at 100 accesses
        frequency = min(1.0, frequency)
        scores["frequency"] = frequency

        # 3. Importance score (user-set)
        scores["importance"] = memory.importance

        # 4. Emotional score (absolute value of valence)
        emotional = abs(memory.emotional_valence)
        scores["emotional"] = emotional

        # 5. Relevance score (semantic similarity)
        scores["relevance"] = semantic_similarity

        # Weighted combination
        total_weight = sum(self.weights.values())
        weighted_sum = (
            scores["frequency"] * self.weights["frequency"] +
            scores["importance"] * self.weights["importance"] +
            scores["emotional"] * self.weights["emotional"] +
            scores["relevance"] * self.weights["relevance"]
        )

        # Multiply by recency (recent memories get boost)
        final_score = (weighted_sum / total_weight) * (0.5 + 0.5 * recency)

        return min(1.0, max(0.0, final_score))

    def should_forget(
        self,
        memory: WeightedMemory,
        threshold: float = 0.1,
        min_age_days: float = 7.0,
    ) -> bool:
        """
        Decide if a memory should be forgotten.

        Returns True if memory is:
        - Old enough (past min_age)
        - Low importance score
        - Rarely accessed
        - Not emotionally significant
        """
        # Don't forget recent memories
        age_days = (datetime.now() - memory.timestamp).total_seconds() / 86400
        if age_days < min_age_days:
            return False

        # Don't forget high-importance memories
        if memory.importance > 0.7:
            return False

        # Don't forget emotionally significant memories
        if abs(memory.emotional_valence) > 0.7:
            return False

        # Calculate score without relevance (for forgetting decision)
        score = self.calculate_score(memory, semantic_similarity=0.0)

        return score < threshold


class WeightedMemoryManager:
    """
    Manage memories with importance weighting.

    Features:
    - Importance-based retrieval
    - Automatic forgetting
    - Access tracking
    - Emotional weighting
    """

    def __init__(self):
        self.memories: Dict[str, WeightedMemory] = {}
        self.calculator = ImportanceCalculator()

    def remember(
        self,
        key: str,
        content: str,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> WeightedMemory:
        """Store a memory with importance metadata"""
        memory = WeightedMemory(
            content=content,
            key=key,
            timestamp=datetime.now(),
            importance=importance,
            emotional_valence=emotional_valence,
            tags=tags or [],
            metadata=metadata or {},
        )

        self.memories[key] = memory
        logger.debug(f"Stored memory: {key} (importance={importance})")
        return memory

    def recall(
        self,
        query: str,
        top_k: int = 5,
        get_similarity: callable = None,
    ) -> List[WeightedMemory]:
        """
        Recall memories ranked by importance-weighted score.

        Args:
            query: Search query
            top_k: Number of results
            get_similarity: Function to get semantic similarity (query, content) -> float
        """
        scored_memories = []

        for key, memory in self.memories.items():
            # Get semantic similarity if function provided
            similarity = 0.0
            if get_similarity:
                try:
                    similarity = get_similarity(query, memory.content)
                except Exception:
                    # Simple keyword matching fallback
                    query_words = set(query.lower().split())
                    content_words = set(memory.content.lower().split())
                    overlap = len(query_words & content_words)
                    similarity = overlap / max(1, len(query_words))

            # Calculate importance-weighted score
            score = self.calculator.calculate_score(memory, query, similarity)

            scored_memories.append((memory, score))

        # Sort by score descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Update access counts for returned memories
        results = []
        for memory, score in scored_memories[:top_k]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            results.append(memory)

        return results

    def forget_low_importance(self, threshold: float = 0.1) -> int:
        """
        Forget memories below importance threshold.

        Returns number of memories forgotten.
        """
        to_forget = []

        for key, memory in self.memories.items():
            if self.calculator.should_forget(memory, threshold):
                to_forget.append(key)

        for key in to_forget:
            del self.memories[key]
            logger.debug(f"Forgot low-importance memory: {key}")

        if to_forget:
            logger.info(f"Forgot {len(to_forget)} low-importance memories")

        return len(to_forget)

    def mark_important(self, key: str, importance: float):
        """Update importance of a memory"""
        if key in self.memories:
            self.memories[key].importance = importance

    def mark_emotional(self, key: str, valence: float):
        """Update emotional valence of a memory"""
        if key in self.memories:
            self.memories[key].emotional_valence = valence

    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        if not self.memories:
            return {"count": 0}

        importances = [m.importance for m in self.memories.values()]
        ages = [(datetime.now() - m.timestamp).days for m in self.memories.values()]

        return {
            "count": len(self.memories),
            "avg_importance": sum(importances) / len(importances),
            "avg_age_days": sum(ages) / len(ages),
            "oldest_days": max(ages),
            "high_importance_count": sum(1 for i in importances if i > 0.7),
        }


# Integration with existing memory system
async def enhance_memory_recall(memory_system, query: str, top_k: int = 5) -> List[Dict]:
    """
    Enhanced recall with importance weighting.

    Wraps existing memory system with importance scoring.
    """
    # Get base results from existing system
    base_results = await memory_system.recall(query, top_k=top_k * 2)

    # Apply importance weighting
    calculator = ImportanceCalculator()
    scored_results = []

    for result in base_results:
        # Create weighted memory from result
        memory = WeightedMemory(
            content=result.get("content", ""),
            key=result.get("id", ""),
            timestamp=datetime.fromisoformat(result.get("timestamp", datetime.now().isoformat())),
            importance=result.get("importance", 0.5),
            emotional_valence=result.get("emotional_valence", 0.0),
            access_count=result.get("access_count", 0),
        )

        # Calculate score
        similarity = result.get("similarity", 0.0)
        score = calculator.calculate_score(memory, query, similarity)

        scored_results.append((result, score))

    # Sort and return top_k
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r, s in scored_results[:top_k]]
