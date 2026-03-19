"""
cosmos Memory Dreaming - Unsupervised Memory Consolidation

Novel Approaches:
1. Idle-Time Processing - Consolidate memories when system is idle
2. Pattern Discovery - Cluster similar memories for organization
3. Creative Recombination - Generate novel insights from memory mixing
4. Forgetting Optimization - Intelligent memory pruning based on utility
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from collections import defaultdict

from loguru import logger

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class DreamResult:
    """Result from a dreaming session."""
    session_id: str
    start_time: datetime
    end_time: datetime
    memories_processed: int
    clusters_formed: int
    insights_generated: list[str]
    memories_forgotten: int
    consolidation_score: float


@dataclass
class MemoryCluster:
    """A cluster of related memories."""
    id: str
    memory_ids: list[str]
    centroid: Optional[list[float]] = None
    label: str = ""
    coherence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class MemoryDreamer:
    """
    Background memory consolidation during idle periods.

    Inspired by how biological memory consolidation occurs during sleep.
    Performs:
    - Clustering of similar memories
    - Pattern discovery
    - Creative recombination
    - Utility-based forgetting
    """

    def __init__(
        self,
        idle_threshold_minutes: int = 5,
        consolidation_interval_hours: float = 1.0,
        creativity_factor: float = 0.3,
        forgetting_threshold: float = 0.2,
    ):
        self.idle_threshold = timedelta(minutes=idle_threshold_minutes)
        self.consolidation_interval = timedelta(hours=consolidation_interval_hours)
        self.creativity_factor = creativity_factor
        self.forgetting_threshold = forgetting_threshold

        self.clusters: dict[str, MemoryCluster] = {}
        self.dream_history: list[DreamResult] = []

        self.last_activity: datetime = datetime.now()
        self.last_dream: Optional[datetime] = None

        self._is_dreaming = False
        self._dream_task: Optional[asyncio.Task] = None

        # Callbacks for memory access
        self.get_memories_fn: Optional[Callable] = None
        self.get_embedding_fn: Optional[Callable] = None
        self.store_memory_fn: Optional[Callable] = None
        self.delete_memory_fn: Optional[Callable] = None
        self.update_memory_fn: Optional[Callable] = None

    def set_callbacks(
        self,
        get_memories: Callable,
        get_embedding: Callable,
        store_memory: Optional[Callable] = None,
        delete_memory: Optional[Callable] = None,
        update_memory: Optional[Callable] = None,
    ):
        """Set memory access callbacks."""
        self.get_memories_fn = get_memories
        self.get_embedding_fn = get_embedding
        self.store_memory_fn = store_memory
        self.delete_memory_fn = delete_memory
        self.update_memory_fn = update_memory

    def record_activity(self):
        """Record user/system activity to reset idle timer."""
        self.last_activity = datetime.now()

    def is_idle(self) -> bool:
        """Check if system has been idle long enough."""
        return datetime.now() - self.last_activity > self.idle_threshold

    def should_dream(self) -> bool:
        """Check if it's time for a dreaming session."""
        if self._is_dreaming:
            return False

        if not self.is_idle():
            return False

        if self.last_dream is None:
            return True

        return datetime.now() - self.last_dream > self.consolidation_interval

    async def start_background_dreaming(self):
        """Start background dreaming process."""
        if self._dream_task is not None:
            return

        self._dream_task = asyncio.create_task(self._dream_loop())
        logger.info("Background dreaming started")

    async def stop_background_dreaming(self):
        """Stop background dreaming."""
        if self._dream_task:
            self._dream_task.cancel()
            try:
                await self._dream_task
            except asyncio.CancelledError:
                pass
            self._dream_task = None
        logger.info("Background dreaming stopped")

    async def _dream_loop(self):
        """Main dreaming loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                if self.should_dream():
                    await self.dream()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dream loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 min on error

    async def dream(self) -> DreamResult:
        """
        Perform a dreaming/consolidation session.

        This is where the magic happens:
        1. Cluster similar memories
        2. Discover patterns
        3. Generate insights through recombination
        4. Prune low-utility memories
        """
        if self._is_dreaming:
            raise RuntimeError("Already dreaming")

        self._is_dreaming = True
        self.last_dream = datetime.now()

        session_id = f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        logger.info(f"Starting dream session {session_id}")

        memories_processed = 0
        clusters_formed = 0
        insights = []
        memories_forgotten = 0

        try:
            # Get memories to process
            memories = await self._get_memories_for_dreaming()
            memories_processed = len(memories)

            if not memories:
                logger.info("No memories to process")
                return DreamResult(
                    session_id=session_id,
                    start_time=start_time,
                    end_time=datetime.now(),
                    memories_processed=0,
                    clusters_formed=0,
                    insights_generated=[],
                    memories_forgotten=0,
                    consolidation_score=0.0,
                )

            # Phase 1: Cluster similar memories
            new_clusters = await self._cluster_memories(memories)
            clusters_formed = len(new_clusters)

            # Phase 2: Generate insights from clusters
            for cluster in new_clusters:
                cluster_insights = await self._generate_cluster_insights(cluster, memories)
                insights.extend(cluster_insights)

            # Phase 3: Creative recombination
            if self.creativity_factor > 0 and len(memories) > 5:
                creative_insights = await self._creative_recombination(memories)
                insights.extend(creative_insights)

            # Phase 4: Forgetting (prune low-utility memories)
            memories_forgotten = await self._intelligent_forgetting(memories)

            # Store insights as new memories
            for insight in insights[:5]:  # Limit to top 5
                if self.store_memory_fn:
                    await self.store_memory_fn(
                        content=insight,
                        tags=["dream_insight", session_id],
                        importance=0.7,
                    )

            consolidation_score = self._calculate_consolidation_score(
                memories_processed, clusters_formed, len(insights), memories_forgotten
            )

            result = DreamResult(
                session_id=session_id,
                start_time=start_time,
                end_time=datetime.now(),
                memories_processed=memories_processed,
                clusters_formed=clusters_formed,
                insights_generated=insights,
                memories_forgotten=memories_forgotten,
                consolidation_score=consolidation_score,
            )

            self.dream_history.append(result)
            logger.info(f"Dream session complete: {clusters_formed} clusters, {len(insights)} insights")

            return result

        finally:
            self._is_dreaming = False

    async def _get_memories_for_dreaming(self) -> list[dict]:
        """Get memories to process during dreaming."""
        if not self.get_memories_fn:
            return []

        try:
            memories = await self.get_memories_fn(limit=200)
            return memories
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return []

    async def _cluster_memories(self, memories: list[dict]) -> list[MemoryCluster]:
        """Cluster similar memories using embeddings."""
        if np is None or not self.get_embedding_fn:
            return []

        # Get embeddings
        embeddings = []
        memory_ids = []

        for mem in memories:
            if "embedding" in mem and mem["embedding"]:
                embeddings.append(mem["embedding"])
                memory_ids.append(mem.get("id", ""))
            elif "content" in mem:
                try:
                    emb = await self.get_embedding_fn(mem["content"])
                    if emb:
                        embeddings.append(emb)
                        memory_ids.append(mem.get("id", ""))
                except Exception:
                    pass

        if len(embeddings) < 5:
            return []

        # Simple k-means clustering
        embeddings_array = np.array(embeddings)
        n_clusters = min(10, len(embeddings) // 5)

        clusters = await self._kmeans_cluster(embeddings_array, n_clusters)

        # Build cluster objects
        result_clusters = []
        for cluster_id, indices in clusters.items():
            if len(indices) < 2:
                continue

            cluster = MemoryCluster(
                id=f"cluster_{datetime.now().strftime('%Y%m%d')}_{cluster_id}",
                memory_ids=[memory_ids[i] for i in indices],
                centroid=embeddings_array[indices].mean(axis=0).tolist(),
                coherence_score=self._calculate_cluster_coherence(
                    embeddings_array[indices]
                ),
            )

            # Generate label from common terms
            cluster_memories = [memories[i] for i in indices]
            cluster.label = self._generate_cluster_label(cluster_memories)

            result_clusters.append(cluster)
            self.clusters[cluster.id] = cluster

        return result_clusters

    async def _kmeans_cluster(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        max_iterations: int = 50,
    ) -> dict[int, list[int]]:
        """Simple k-means implementation."""
        n_samples = len(embeddings)

        # Initialize centroids randomly
        centroid_indices = random.sample(range(n_samples), n_clusters)
        centroids = embeddings[centroid_indices].copy()

        assignments = np.zeros(n_samples, dtype=int)

        for _ in range(max_iterations):
            # Assign points to nearest centroid
            distances = np.zeros((n_samples, n_clusters))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.linalg.norm(embeddings - centroid, axis=1)

            new_assignments = np.argmin(distances, axis=1)

            # Check for convergence
            if np.array_equal(assignments, new_assignments):
                break

            assignments = new_assignments

            # Update centroids
            for i in range(n_clusters):
                mask = assignments == i
                if mask.any():
                    centroids[i] = embeddings[mask].mean(axis=0)

        # Build cluster dict
        clusters = defaultdict(list)
        for idx, cluster_id in enumerate(assignments):
            clusters[int(cluster_id)].append(idx)

        return dict(clusters)

    def _calculate_cluster_coherence(self, embeddings: np.ndarray) -> float:
        """Calculate how coherent/tight a cluster is."""
        if len(embeddings) < 2:
            return 1.0

        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        avg_distance = distances.mean()

        # Normalize to 0-1 (lower distance = higher coherence)
        return 1.0 / (1.0 + avg_distance)

    def _generate_cluster_label(self, memories: list[dict]) -> str:
        """Generate a label for a cluster based on common terms."""
        # Simple term frequency approach
        word_counts = defaultdict(int)

        for mem in memories:
            content = mem.get("content", "")
            words = content.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] += 1

        # Get top terms
        top_terms = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        return " ".join(term for term, _ in top_terms)

    async def _generate_cluster_insights(
        self,
        cluster: MemoryCluster,
        all_memories: list[dict],
    ) -> list[str]:
        """Generate insights from a memory cluster."""
        insights = []

        # Find memories in this cluster
        cluster_memories = [
            m for m in all_memories
            if m.get("id") in cluster.memory_ids
        ]

        if len(cluster_memories) < 2:
            return []

        # Pattern: Repeated topics
        if cluster.coherence_score > 0.7:
            insights.append(
                f"Strong theme detected: '{cluster.label}' appears consistently "
                f"across {len(cluster_memories)} memories"
            )

        # Pattern: Temporal clustering
        timestamps = [
            datetime.fromisoformat(m.get("created_at", datetime.now().isoformat()))
            for m in cluster_memories
            if "created_at" in m
        ]
        if timestamps:
            time_span = max(timestamps) - min(timestamps)
            if time_span < timedelta(hours=1):
                insights.append(
                    f"Burst activity: {len(cluster_memories)} related memories "
                    f"created within a short period about '{cluster.label}'"
                )

        return insights

    async def _creative_recombination(self, memories: list[dict]) -> list[str]:
        """
        Novel: Generate creative insights by combining distant memories.

        This mimics the creative function of dreaming where unrelated
        concepts get connected in unexpected ways.
        """
        insights = []

        if len(memories) < 10:
            return []

        # Random sampling for creative combinations
        num_attempts = int(len(memories) * self.creativity_factor)

        for _ in range(num_attempts):
            # Pick two random memories
            mem1, mem2 = random.sample(memories, 2)

            content1 = mem1.get("content", "")[:100]
            content2 = mem2.get("content", "")[:100]

            # Extract key terms
            terms1 = set(w for w in content1.lower().split() if len(w) > 4)
            terms2 = set(w for w in content2.lower().split() if len(w) > 4)

            # Look for unexpected connections
            if terms1 and terms2:
                common = terms1 & terms2
                if common:
                    insights.append(
                        f"Connection found: '{list(common)[0]}' links different contexts"
                    )
                elif not common and random.random() < self.creativity_factor:
                    # Force a creative connection
                    t1 = random.choice(list(terms1)) if terms1 else "concept"
                    t2 = random.choice(list(terms2)) if terms2 else "idea"
                    insights.append(
                        f"Novel association: '{t1}' and '{t2}' might be related"
                    )

        return insights[:3]  # Limit creative insights

    async def _intelligent_forgetting(self, memories: list[dict]) -> int:
        """
        Intelligently forget low-utility memories.

        Factors:
        - Low access count
        - Old age
        - Low importance
        - Redundancy with other memories
        """
        if not self.delete_memory_fn:
            return 0

        forgotten = 0

        for mem in memories:
            utility = self._calculate_memory_utility(mem)

            if utility < self.forgetting_threshold:
                try:
                    await self.delete_memory_fn(mem.get("id"))
                    forgotten += 1
                except Exception as e:
                    logger.debug(f"Failed to forget memory: {e}")

        return forgotten

    def _calculate_memory_utility(self, memory: dict) -> float:
        """Calculate utility score for a memory."""
        utility = 0.5  # Base utility

        # Access count factor
        access_count = memory.get("access_count", 0)
        utility += min(0.3, access_count * 0.05)

        # Age factor (older = less utility, unless frequently accessed)
        created_at = memory.get("created_at")
        if created_at:
            try:
                age_days = (datetime.now() - datetime.fromisoformat(created_at)).days
                age_factor = 1.0 / (1.0 + age_days / 30)
                utility *= (0.5 + 0.5 * age_factor)
            except Exception:
                pass

        # Importance factor
        importance = memory.get("importance_score", 0.5)
        utility *= (0.5 + 0.5 * importance)

        return utility

    def _calculate_consolidation_score(
        self,
        processed: int,
        clusters: int,
        insights: int,
        forgotten: int,
    ) -> float:
        """Calculate overall consolidation effectiveness."""
        if processed == 0:
            return 0.0

        cluster_ratio = clusters / max(1, processed / 10)  # ~10 per cluster is ideal
        insight_ratio = min(1.0, insights / 5)  # 5 insights is good
        cleanup_ratio = min(1.0, forgotten / max(1, processed * 0.1))  # 10% cleanup

        return (cluster_ratio * 0.4 + insight_ratio * 0.4 + cleanup_ratio * 0.2)

    def get_stats(self) -> dict:
        """Get dreaming statistics."""
        return {
            "is_dreaming": self._is_dreaming,
            "last_dream": self.last_dream.isoformat() if self.last_dream else None,
            "total_dreams": len(self.dream_history),
            "total_clusters": len(self.clusters),
            "is_idle": self.is_idle(),
            "creativity_factor": self.creativity_factor,
            "recent_dreams": [
                {
                    "session_id": d.session_id,
                    "memories_processed": d.memories_processed,
                    "insights": len(d.insights_generated),
                    "score": d.consolidation_score,
                }
                for d in self.dream_history[-5:]
            ],
        }
