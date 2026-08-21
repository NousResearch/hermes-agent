"""
Flat Memory Store — the simplest possible memory backend.

Stores memories as a list of strings. Recall is substring matching.
No decay, no importance weighting, no consolidation.
This is the baseline that memory plugins must beat.
"""

from typing import Dict, List, Any, Optional

from benchmarks.interface import BenchmarkableStore


class FlatMemoryStore(BenchmarkableStore):
    """Baseline memory: Python list with substring search."""

    def __init__(self, **kwargs):
        self._memories: List[Dict[str, Any]] = []
        self._clock: float = 0.0  # simulated days
        self._rewards: Dict[int, float] = {}  # memory index -> cumulative reward

    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        self._memories.append({
            "content": content,
            "category": category,
            "scope": scope,
            "importance": importance,
            "stored_at": self._clock,
            "access_count": 0,
        })

    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> List[str]:
        """Recall by simple word overlap scoring."""
        query_words = set(query.lower().split())
        # Remove stop words
        stop = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                "where", "when", "who", "which", "do", "does", "did", "can",
                "could", "should", "would", "will", "shall", "may", "might",
                "has", "have", "had", "be", "been", "being", "to", "of", "in",
                "for", "on", "with", "at", "by", "from", "and", "or", "not",
                "that", "this", "it", "its", "we", "our", "you", "your",
                "they", "their", "about", "into", "through", "during"}
        query_words -= stop

        if not query_words:
            # Fallback: return most recent memories
            results = sorted(self._memories, key=lambda m: m["stored_at"], reverse=True)
            return [m["content"] for m in results[:top_k]]

        scored = []
        for mem in self._memories:
            if scope and mem["scope"] != scope:
                continue
            content_words = set(mem["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored.append((score, mem))

        # Sort by score descending, then by recency
        scored.sort(key=lambda x: (x[0], x[1]["stored_at"]), reverse=True)

        if scored:
            return [m["content"] for _, m in scored[:top_k]]

        # Fallback: return most recent memories when no word overlap found
        results = sorted(self._memories, key=lambda m: m["stored_at"], reverse=True)
        return [m["content"] for m in results[:top_k]]

    def simulate_time(self, days: float) -> None:
        self._clock += days

    def simulate_access(self, content_substring: str) -> None:
        for mem in self._memories:
            if content_substring in mem["content"]:
                mem["access_count"] += 1
                break

    def consolidate(self) -> None:
        pass  # No-op for flat store

    def reward_memory(self, memory_id: int, signal: float) -> None:
        """Apply reward signal to a memory by its index."""
        if memory_id in self._rewards:
            self._rewards[memory_id] += signal
        else:
            self._rewards[memory_id] = signal

    def recall_with_ids(self, query: str, top_k: int = 10,
                        scope: Optional[str] = None) -> List[tuple]:
        """Recall memories with their indices.

        Returns list of (content, id) tuples where id is the memory's index.
        """
        query_words = set(query.lower().split())
        stop = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                "where", "when", "who", "which", "do", "does", "did", "can",
                "could", "should", "would", "will", "shall", "may", "might",
                "has", "have", "had", "be", "been", "being", "to", "of", "in",
                "for", "on", "with", "at", "by", "from", "and", "or", "not",
                "that", "this", "it", "its", "we", "our", "you", "your",
                "they", "their", "about", "into", "through", "during"}
        query_words -= stop

        scored = []
        for idx, mem in enumerate(self._memories):
            if scope and mem["scope"] != scope:
                continue
            content_words = set(mem["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                base_score = overlap / len(query_words) if query_words else 0
                # Add reward bonus
                reward = self._rewards.get(idx, 0.0)
                score = base_score + reward * 0.1
                scored.append((score, idx, mem))

        scored.sort(key=lambda x: (x[0], x[2]["stored_at"]), reverse=True)

        if scored:
            return [(m["content"], i) for _, i, m in scored[:top_k]]

        # Fallback: return most recent memories
        results = sorted(enumerate(self._memories), key=lambda x: x[1]["stored_at"], reverse=True)
        return [(m["content"], i) for i, m in results[:top_k]]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": len(self._memories),
            "clock": self._clock,
        }

    def reset(self) -> None:
        self._memories.clear()
        self._clock = 0.0
        self._rewards.clear()

from benchmarks.capabilities import BackendCapabilities

BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,   # basic store + recall
    scopes=True,                   # recall() filters by scope parameter
    time_simulation=True,          # simulate_time() advances internal clock
    reward_learning=True,          # reward_memory() and recall_with_ids() available
)
BACKEND_CLASS = FlatMemoryStore
