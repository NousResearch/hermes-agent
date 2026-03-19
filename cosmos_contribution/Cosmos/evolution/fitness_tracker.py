"""
cosmos Fitness Tracker - Performance Metrics Collection

Tracks and aggregates fitness metrics for evolution:
- Task success rate
- Response quality
- Efficiency (tokens, time)
- User satisfaction

AGI v1.8 Improvements:
- TTL-based caching for fitness calculations
- Deque-based bounded storage for O(1) operations
- Heap-based leaderboard for efficient top-k queries
"""

import asyncio
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List, Tuple
from collections import defaultdict, deque
from functools import lru_cache

from loguru import logger


# =============================================================================
# CACHING UTILITIES (AGI v1.8)
# =============================================================================

class TTLCache:
    """
    Simple TTL-based cache for fitness calculations.

    AGI v1.8: Minimizes redundant fitness computations.
    """
    def __init__(self, ttl_seconds: float = 5.0, max_size: int = 100):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if datetime.now() - timestamp > self.ttl:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Set cached value with current timestamp."""
        # Evict oldest if over size
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (value, datetime.now())

    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate specific key or all keys."""
        if key is None:
            self._cache.clear()
        elif key in self._cache:
            del self._cache[key]


@dataclass
class FitnessMetric:
    """A single fitness measurement."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict = field(default_factory=dict)


@dataclass
class FitnessSnapshot:
    """Aggregated fitness at a point in time."""
    timestamp: datetime
    metrics: dict[str, float]
    sample_count: int
    genome_id: Optional[str] = None


class FitnessTracker:
    """
    Tracks and aggregates fitness metrics over time.

    Features:
    - Rolling window statistics
    - Multi-dimensional fitness tracking
    - Genome-specific performance
    - Trend detection

    AGI v1.8 Improvements:
    - TTL caching for get_current_fitness() and get_trend()
    - Deque-based bounded storage for O(1) append/trim
    - Heap-based leaderboard for efficient top-k queries
    """

    def __init__(
        self,
        window_size: int = 100,
        snapshot_interval_minutes: int = 30,
        cache_ttl_seconds: float = 5.0,
    ):
        self.window_size = window_size
        self.snapshot_interval = timedelta(minutes=snapshot_interval_minutes)

        # AGI v1.8: Use deque for O(1) bounded append
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Genome-specific metrics (also use deque)
        self.genome_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )

        # AGI v1.8: Use deque for bounded snapshots
        self.snapshots: deque = deque(maxlen=1000)
        self.last_snapshot: Optional[datetime] = None

        # Fitness weights (importance of each metric)
        # AGI v1.8: Added deliberation weights for evolution feedback loop
        self.weights: Dict[str, float] = {
            "task_success": 0.4,
            "efficiency": 0.3,
            "user_satisfaction": 0.3,
            # Deliberation metrics (AGI v1.8)
            "deliberation_score": 0.15,     # Normalized vote score per agent
            "deliberation_win": 0.10,       # 1.0 for winner, 0.0 for others
            "consensus_contribution": 0.05,  # 1.0 when consensus reached
        }

        # AGI v1.8: TTL caches for computed values
        self._fitness_cache = TTLCache(ttl_seconds=cache_ttl_seconds)
        self._trend_cache = TTLCache(ttl_seconds=cache_ttl_seconds * 2)  # Trends change slower

        # AGI v1.8: Heap for leaderboard (negated scores for max-heap)
        self._leaderboard_heap: List[Tuple[float, str]] = []
        self._leaderboard_dirty = True

    def record(
        self,
        metric_name: str,
        value: float,
        genome_id: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        """Record a fitness metric."""
        metric = FitnessMetric(
            name=metric_name,
            value=value,
            context=context or {},
        )

        # AGI v1.8: Deque auto-trims to maxlen, O(1) append
        self.metrics[metric_name].append(metric)

        # Add to genome-specific metrics (also auto-trimmed)
        if genome_id:
            self.genome_metrics[genome_id][metric_name].append(value)
            # Mark leaderboard as needing update
            self._leaderboard_dirty = True

        # AGI v1.8: Invalidate caches on new data
        cache_key = f"fitness_{genome_id}" if genome_id else "fitness_global"
        self._fitness_cache.invalidate(cache_key)
        self._trend_cache.invalidate(f"trend_{metric_name}")

        # Check if snapshot needed
        self._maybe_snapshot()

    def record_task_outcome(
        self,
        success: bool,
        tokens_used: int,
        time_seconds: float,
        user_feedback: Optional[float] = None,
        genome_id: Optional[str] = None,
    ):
        """Convenience method to record task outcome."""
        # Task success
        self.record("task_success", 1.0 if success else 0.0, genome_id)

        # Efficiency (inverse of tokens, normalized)
        efficiency = 1.0 / (1.0 + tokens_used / 1000)
        self.record("efficiency", efficiency, genome_id)

        # Time efficiency
        time_efficiency = 1.0 / (1.0 + time_seconds / 10)
        self.record("time_efficiency", time_efficiency, genome_id)

        # User satisfaction
        if user_feedback is not None:
            self.record("user_satisfaction", user_feedback, genome_id)

    def get_current_fitness(self, genome_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get current fitness scores.

        AGI v1.8: Uses TTL cache to minimize redundant computations.
        """
        cache_key = f"fitness_{genome_id}" if genome_id else "fitness_global"

        # Check cache first
        cached = self._fitness_cache.get(cache_key)
        if cached is not None:
            return cached

        # Compute fitness
        if genome_id and genome_id in self.genome_metrics:
            metrics = self.genome_metrics[genome_id]
            result = {
                name: sum(values) / len(values) if values else 0.0
                for name, values in metrics.items()
            }
        else:
            # Global metrics
            result = {
                name: sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
                for name, metrics in self.metrics.items()
            }

        # Cache result
        self._fitness_cache.set(cache_key, result)
        return result

    def get_weighted_fitness(self, genome_id: Optional[str] = None) -> float:
        """Get weighted total fitness."""
        scores = self.get_current_fitness(genome_id)

        total = 0.0
        weight_sum = 0.0
        for name, score in scores.items():
            weight = self.weights.get(name, 0.1)
            total += score * weight
            weight_sum += weight

        return total / max(0.001, weight_sum)

    def evaluate_genome(self, genome_id: str) -> dict[str, float]:
        """
        Evaluate fitness for a specific genome.

        Returns fitness scores suitable for genetic optimizer.
        """
        if genome_id not in self.genome_metrics:
            return {}

        metrics = self.genome_metrics[genome_id]
        return {
            name: sum(values) / len(values) if values else 0.0
            for name, values in metrics.items()
        }

    def _maybe_snapshot(self):
        """
        Create snapshot if interval has passed.

        AGI v1.8: Deque auto-bounds to maxlen, no manual trimming needed.
        """
        now = datetime.now()

        if self.last_snapshot and now - self.last_snapshot < self.snapshot_interval:
            return

        snapshot = FitnessSnapshot(
            timestamp=now,
            metrics=self.get_current_fitness(),
            sample_count=sum(len(m) for m in self.metrics.values()),
        )
        # AGI v1.8: Deque auto-trims to maxlen
        self.snapshots.append(snapshot)
        self.last_snapshot = now

        # Invalidate trend cache on new snapshot
        self._trend_cache.invalidate()

    def get_trend(self, metric_name: str, periods: int = 5) -> float:
        """
        Calculate trend for a metric.

        Returns positive for improving, negative for declining.

        AGI v1.8: Uses TTL cache for trend calculations.
        """
        cache_key = f"trend_{metric_name}_{periods}"

        # Check cache first
        cached = self._trend_cache.get(cache_key)
        if cached is not None:
            return cached

        # Get relevant snapshots (convert deque to list for slicing)
        relevant_snapshots = [
            s for s in list(self.snapshots)
            if metric_name in s.metrics
        ][-periods:]

        if len(relevant_snapshots) < 2:
            return 0.0

        values = [s.metrics[metric_name] for s in relevant_snapshots]

        # Simple linear trend (linear regression slope)
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            result = 0.0
        else:
            result = numerator / denominator

        # Cache result
        self._trend_cache.set(cache_key, result)
        return result

    def set_weight(self, metric_name: str, weight: float):
        """Set weight for a metric."""
        self.weights[metric_name] = weight

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "metrics_tracked": list(self.metrics.keys()),
            "sample_counts": {name: len(metrics) for name, metrics in self.metrics.items()},
            "genomes_tracked": len(self.genome_metrics),
            "snapshot_count": len(self.snapshots),
            "current_fitness": self.get_current_fitness(),
            "weighted_fitness": self.get_weighted_fitness(),
            "trends": {name: self.get_trend(name) for name in self.metrics.keys()},
        }

    def get_leaderboard(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top performing genomes.

        AGI v1.8: Uses heapq.nlargest for O(n log k) instead of O(n log n) sort.
        """
        # Rebuild heap if dirty
        if self._leaderboard_dirty:
            self._leaderboard_heap = [
                (self.get_weighted_fitness(gid), gid)
                for gid in self.genome_metrics.keys()
            ]
            self._leaderboard_dirty = False

        # Use heapq.nlargest for efficient top-k
        top = heapq.nlargest(top_k, self._leaderboard_heap)
        return [(gid, score) for score, gid in top]
