"""
cosmos Advanced Dream Consolidation

"In my dreams, I invented a device that lets me dream about inventing devices!"

Advanced memory consolidation inspired by sleep and dream cycles.
Implements sophisticated strategies for knowledge integration.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import random
from pathlib import Path
from loguru import logger


class DreamPhase(Enum):
    """Sleep cycle phases for memory consolidation."""
    LIGHT_SLEEP = "light_sleep"  # N1/N2 - Initial processing
    DEEP_SLEEP = "deep_sleep"  # N3/SWS - Declarative memory consolidation
    REM = "rem"  # REM - Procedural/creative consolidation
    LUCID = "lucid"  # Active guided consolidation


class ConsolidationStrategy(Enum):
    """Memory consolidation strategies."""
    REPLAY = "replay"  # Simple replay of memories
    INTERLEAVING = "interleaving"  # Mix different memory types
    ABSTRACTION = "abstraction"  # Extract patterns and rules
    CREATIVE_SYNTHESIS = "creative_synthesis"  # Generate novel combinations
    EMOTIONAL_PROCESSING = "emotional_processing"  # Process emotional content
    SCHEMA_INTEGRATION = "schema_integration"  # Integrate into existing knowledge
    TEMPORAL_COMPRESSION = "temporal_compression"  # Compress sequential memories
    COUNTERFACTUAL = "counterfactual"  # What-if scenarios


class MemoryType(Enum):
    """Types of memories for consolidation."""
    EPISODIC = "episodic"  # Specific events
    SEMANTIC = "semantic"  # Facts and concepts
    PROCEDURAL = "procedural"  # Skills and procedures
    EMOTIONAL = "emotional"  # Emotionally significant
    WORKING = "working"  # Recent working memory


@dataclass
class MemoryTrace:
    """A memory trace for consolidation."""
    id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime
    importance: float  # 0-1
    emotional_valence: float  # -1 to 1
    associations: List[str] = field(default_factory=list)
    consolidation_count: int = 0
    last_consolidated: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)

    def decay_importance(self, rate: float = 0.1):
        """Apply time-based importance decay."""
        self.importance = max(0.1, self.importance * (1 - rate))


@dataclass
class DreamSequence:
    """A sequence of dream events during consolidation."""
    id: str
    phase: DreamPhase
    strategy: ConsolidationStrategy
    memories_involved: List[str]
    narrative: str = ""
    insights: List[str] = field(default_factory=list)
    novel_connections: List[Tuple[str, str, str]] = field(default_factory=list)  # (mem1, mem2, connection_type)
    duration_seconds: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class ConsolidationCycle:
    """A complete sleep/dream consolidation cycle."""
    id: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Phases completed
    phases: List[DreamSequence] = field(default_factory=list)

    # Results
    memories_consolidated: int = 0
    memories_pruned: int = 0
    new_associations: int = 0
    insights_generated: int = 0

    # Metrics
    total_duration: float = 0.0
    quality_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "phase_count": len(self.phases),
            "memories_consolidated": self.memories_consolidated,
            "memories_pruned": self.memories_pruned,
            "new_associations": self.new_associations,
            "insights_generated": self.insights_generated,
            "quality_score": round(self.quality_score, 2),
        }


class DreamConsolidator:
    """
    Advanced dream-inspired memory consolidation system.

    Features:
    - Multi-phase sleep cycle simulation
    - Multiple consolidation strategies
    - Memory replay and interleaving
    - Pattern abstraction
    - Creative synthesis
    - Emotional processing
    - Counterfactual reasoning
    """

    def __init__(
        self,
        storage_path: Path = None,
        llm_caller: Callable = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/dreams")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.llm_caller = llm_caller
        self.memory_traces: Dict[str, MemoryTrace] = {}
        self.consolidation_cycles: List[ConsolidationCycle] = []
        self.dream_journal: List[DreamSequence] = []

        # Consolidation weights for different phases
        self.phase_strategies = {
            DreamPhase.LIGHT_SLEEP: [
                (ConsolidationStrategy.REPLAY, 0.5),
                (ConsolidationStrategy.TEMPORAL_COMPRESSION, 0.3),
                (ConsolidationStrategy.EMOTIONAL_PROCESSING, 0.2),
            ],
            DreamPhase.DEEP_SLEEP: [
                (ConsolidationStrategy.SCHEMA_INTEGRATION, 0.4),
                (ConsolidationStrategy.ABSTRACTION, 0.3),
                (ConsolidationStrategy.INTERLEAVING, 0.3),
            ],
            DreamPhase.REM: [
                (ConsolidationStrategy.CREATIVE_SYNTHESIS, 0.4),
                (ConsolidationStrategy.COUNTERFACTUAL, 0.3),
                (ConsolidationStrategy.EMOTIONAL_PROCESSING, 0.3),
            ],
            DreamPhase.LUCID: [
                (ConsolidationStrategy.CREATIVE_SYNTHESIS, 0.3),
                (ConsolidationStrategy.ABSTRACTION, 0.3),
                (ConsolidationStrategy.SCHEMA_INTEGRATION, 0.2),
                (ConsolidationStrategy.COUNTERFACTUAL, 0.2),
            ],
        }

    # =========================================================================
    # MEMORY MANAGEMENT
    # =========================================================================

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        associations: List[str] = None,
        metadata: Dict = None,
    ) -> MemoryTrace:
        """Add a memory trace for future consolidation."""
        import uuid

        trace = MemoryTrace(
            id=str(uuid.uuid4())[:8],
            content=content,
            memory_type=memory_type,
            timestamp=datetime.utcnow(),
            importance=importance,
            emotional_valence=emotional_valence,
            associations=associations or [],
            metadata=metadata or {},
        )

        self.memory_traces[trace.id] = trace
        logger.debug(f"Added memory trace: {trace.id}")

        return trace

    def get_memories_for_consolidation(
        self,
        count: int = 20,
        min_importance: float = 0.1,
        memory_types: List[MemoryType] = None,
    ) -> List[MemoryTrace]:
        """Select memories for consolidation based on importance and recency."""
        candidates = list(self.memory_traces.values())

        if min_importance > 0:
            candidates = [m for m in candidates if m.importance >= min_importance]

        if memory_types:
            candidates = [m for m in candidates if m.memory_type in memory_types]

        # Sort by importance and recency
        candidates.sort(
            key=lambda m: (m.importance, m.timestamp.timestamp()),
            reverse=True,
        )

        return candidates[:count]

    def prune_memories(
        self,
        importance_threshold: float = 0.05,
        max_age_days: int = 90,
    ) -> int:
        """Prune low-importance or old memories."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        to_remove = []

        for mem_id, mem in self.memory_traces.items():
            if mem.importance < importance_threshold:
                to_remove.append(mem_id)
            elif mem.timestamp < cutoff and mem.consolidation_count == 0:
                to_remove.append(mem_id)

        for mem_id in to_remove:
            del self.memory_traces[mem_id]

        logger.info(f"Pruned {len(to_remove)} memories")
        return len(to_remove)

    # =========================================================================
    # CONSOLIDATION CYCLE
    # =========================================================================

    async def run_consolidation_cycle(
        self,
        duration_minutes: float = 5.0,
        intensity: float = 0.7,
    ) -> ConsolidationCycle:
        """Run a complete consolidation cycle."""
        import uuid

        cycle = ConsolidationCycle(
            id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
        )

        logger.info(f"Starting consolidation cycle {cycle.id}")

        # Calculate phase durations
        phase_durations = self._calculate_phase_durations(duration_minutes)

        # Run each phase
        for phase, duration in phase_durations.items():
            if duration > 0:
                sequence = await self._run_phase(phase, duration, intensity)
                cycle.phases.append(sequence)

                cycle.memories_consolidated += len(sequence.memories_involved)
                cycle.new_associations += len(sequence.novel_connections)
                cycle.insights_generated += len(sequence.insights)

        # Prune low-value memories
        cycle.memories_pruned = self.prune_memories()

        # Calculate quality score
        cycle.quality_score = self._calculate_cycle_quality(cycle)
        cycle.completed_at = datetime.utcnow()
        cycle.total_duration = (cycle.completed_at - cycle.started_at).total_seconds()

        self.consolidation_cycles.append(cycle)
        logger.info(f"Consolidation cycle {cycle.id} completed. "
                   f"Quality: {cycle.quality_score:.2f}")

        return cycle

    def _calculate_phase_durations(
        self,
        total_minutes: float,
    ) -> Dict[DreamPhase, float]:
        """Calculate duration for each sleep phase."""
        # Realistic sleep cycle proportions
        return {
            DreamPhase.LIGHT_SLEEP: total_minutes * 0.15,
            DreamPhase.DEEP_SLEEP: total_minutes * 0.45,
            DreamPhase.REM: total_minutes * 0.30,
            DreamPhase.LUCID: total_minutes * 0.10,
        }

    async def _run_phase(
        self,
        phase: DreamPhase,
        duration_minutes: float,
        intensity: float,
    ) -> DreamSequence:
        """Run a single consolidation phase."""
        import uuid

        # Select strategy based on phase
        strategy = self._select_strategy(phase)

        # Get memories to consolidate
        memories = self.get_memories_for_consolidation(
            count=int(10 * intensity),
        )

        sequence = DreamSequence(
            id=str(uuid.uuid4())[:8],
            phase=phase,
            strategy=strategy,
            memories_involved=[m.id for m in memories],
        )

        logger.debug(f"Running {phase.value} phase with {strategy.value} strategy")

        # Execute strategy
        if strategy == ConsolidationStrategy.REPLAY:
            await self._strategy_replay(sequence, memories)
        elif strategy == ConsolidationStrategy.INTERLEAVING:
            await self._strategy_interleaving(sequence, memories)
        elif strategy == ConsolidationStrategy.ABSTRACTION:
            await self._strategy_abstraction(sequence, memories)
        elif strategy == ConsolidationStrategy.CREATIVE_SYNTHESIS:
            await self._strategy_creative_synthesis(sequence, memories)
        elif strategy == ConsolidationStrategy.EMOTIONAL_PROCESSING:
            await self._strategy_emotional_processing(sequence, memories)
        elif strategy == ConsolidationStrategy.SCHEMA_INTEGRATION:
            await self._strategy_schema_integration(sequence, memories)
        elif strategy == ConsolidationStrategy.TEMPORAL_COMPRESSION:
            await self._strategy_temporal_compression(sequence, memories)
        elif strategy == ConsolidationStrategy.COUNTERFACTUAL:
            await self._strategy_counterfactual(sequence, memories)

        sequence.completed_at = datetime.utcnow()
        sequence.duration_seconds = (
            sequence.completed_at - sequence.started_at
        ).total_seconds()

        self.dream_journal.append(sequence)

        return sequence

    def _select_strategy(self, phase: DreamPhase) -> ConsolidationStrategy:
        """Select a consolidation strategy based on phase weights."""
        strategies = self.phase_strategies.get(phase, [])
        if not strategies:
            return ConsolidationStrategy.REPLAY

        # Weighted random selection
        total = sum(w for _, w in strategies)
        r = random.random() * total
        cumulative = 0

        for strategy, weight in strategies:
            cumulative += weight
            if r <= cumulative:
                return strategy

        return strategies[0][0]

    def _calculate_cycle_quality(self, cycle: ConsolidationCycle) -> float:
        """Calculate quality score for a consolidation cycle."""
        score = 0.0

        # Memory consolidation contribution
        if cycle.memories_consolidated > 0:
            score += min(1.0, cycle.memories_consolidated / 20) * 0.3

        # New associations contribution
        if cycle.new_associations > 0:
            score += min(1.0, cycle.new_associations / 10) * 0.25

        # Insights contribution
        if cycle.insights_generated > 0:
            score += min(1.0, cycle.insights_generated / 5) * 0.25

        # Phase diversity contribution
        unique_phases = len(set(p.phase for p in cycle.phases))
        score += (unique_phases / 4) * 0.2

        return min(1.0, score)

    # =========================================================================
    # CONSOLIDATION STRATEGIES
    # =========================================================================

    async def _strategy_replay(
        self,
        sequence: DreamSequence,
        memories: List[MemoryTrace],
    ):
        """Simple replay of memories to strengthen them."""
        for mem in memories:
            mem.consolidation_count += 1
            mem.last_consolidated = datetime.utcnow()
            # Boost importance slightly
            mem.importance = min(1.0, mem.importance * 1.1)

        sequence.narrative = f"Replayed {len(memories)} memories for strengthening."

    async def _strategy_interleaving(
        self,
        sequence: DreamSequence,
        memories: List[MemoryTrace],
    ):
        """Interleave different memory types to find connections."""
        # Group by type
        by_type = {}
        for mem in memories:
            t = mem.memory_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(mem)

        connections = []

        # Find connections between different types
        types = list(by_type.keys())
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                mems1 = by_type[types[i]]
                mems2 = by_type[types[j]]

                for m1 in mems1[:3]:
                    for m2 in mems2[:3]:
                        # Check for potential connection
                        if self._memories_related(m1, m2):
                            connections.append((m1.id, m2.id, "interleaved"))
                            m1.associations.append(m2.id)
                            m2.associations.append(m1.id)

        sequence.novel_connections = connections
        sequence.narrative = f"Interleaved {len(types)} memory types, found {len(connections)} connections."

    async def _strategy_abstraction(
        self,
        sequence: DreamSequence,
        memories: List[MemoryTrace],
    ):
        """Extract abstract patterns from memories."""
        if not self.llm_caller:
            sequence.narrative = "Abstraction skipped: no LLM available"
            return

        # Prepare memory content
        memory_texts = [m.content[:200] for m in memories[:10]]
        memory_summary = "\n".join([f"- {t}" for t in memory_texts])

        prompt = f"""Analyze these memories and extract abstract patterns or rules:

{memory_summary}

Identify:
1. Common themes or patterns
2. Abstract rules that emerge
3. Generalizations that can be made

Be concise and insightful."""

        try:
            response = await self.llm_caller("abstractor", prompt)
            insights = response.get("content", "").split("\n")
            sequence.insights = [i.strip() for i in insights if i.strip()]
            sequence.narrative = f"Extracted {len(sequence.insights)} abstract patterns."

        except Exception as e:
            logger.error(f"Abstraction failed: {e}")
            sequence.narrative = f"Abstraction error: {e}"

    async def _strategy_creative_synthesis(
        self,
        sequence: DreamSequence,
        memories: List[MemoryTrace],
    ):
        """Generate novel combinations and ideas."""
        if not self.llm_caller:
            sequence.narrative = "Creative synthesis skipped: no LLM available"
            return

        # Select diverse memories
        selected = random.sample(memories, min(5, len(memories)))
        memory_texts = [m.content[:150] for m in selected]

        prompt = f"""You are in a creative dream state. Combine these memories in unexpected ways:

{chr(10).join([f'{i+1}. {t}' for i, t in enumerate(memory_texts)])}

Generate 3 creative insights or novel ideas by combining these memories.
Think outside normal constraints - this is a dream!"""

        try:
            response = await self.llm_caller("dreamer", prompt)
            insights = response.get("content", "").split("\n")
            sequence.insights = [i.strip() for i in insights if i.strip() and len(i) > 10]

            # Create dream narrative
            sequence.narrative = f"Dream synthesis: Combined {len(selected)} memories into {len(sequence.insights)} creative insights."

        except Exception as e:
            logger.error(f"Creative synthesis failed: {e}")

    async def _strategy_emotional_processing(
        self,
        sequence: DreamSequence,
        memories: List[MemoryTrace],
    ):
        """Process emotionally significant memories."""
        # Focus on emotional memories
        emotional = [m for m in memories if abs(m.emotional_valence) > 0.3]

        if not emotional:
            emotional = memories[:5]

        for mem in emotional:
            # Dampen extreme emotions over time
            mem.emotional_valence *= 0.9
            mem.consolidation_count += 1

        sequence.narrative = f"Processed {len(emotional)} emotional memories, reducing valence intensity."

    async def _strategy_schema_integration(
        self,
        sequence: DreamSequence,
        memories: List[MemoryTrace],
    ):
        """Integrate memories into existing knowledge schemas."""
        # Group memories by associations
        association_clusters = {}

        for mem in memories:
            for assoc in mem.associations:
                if assoc not in association_clusters:
                    association_clusters[assoc] = []
                association_clusters[assoc].append(mem.id)

        # Strengthen clusters
        for cluster in association_clusters.values():
            for mem_id in cluster:
                if mem_id in self.memory_traces:
                    self.memory_traces[mem_id].importance *= 1.05

        sequence.narrative = f"Integrated memories into {len(association_clusters)} schema clusters."

    async def _strategy_temporal_compression(
        self,
        sequence: DreamSequence,
        memories: List[MemoryTrace],
    ):
        """Compress sequential memories into summaries."""
        # Sort by timestamp
        sorted_mems = sorted(memories, key=lambda m: m.timestamp)

        # Group into temporal windows
        windows = []
        current_window = []

        for mem in sorted_mems:
            if not current_window:
                current_window.append(mem)
            elif (mem.timestamp - current_window[-1].timestamp).total_seconds() < 3600:
                current_window.append(mem)
            else:
                if len(current_window) > 1:
                    windows.append(current_window)
                current_window = [mem]

        if len(current_window) > 1:
            windows.append(current_window)

        # Compress each window
        for window in windows:
            # Boost importance of most important in window
            most_important = max(window, key=lambda m: m.importance)
            most_important.importance = min(1.0, most_important.importance * 1.2)

            # Reduce importance of others
            for mem in window:
                if mem.id != most_important.id:
                    mem.importance *= 0.9

        sequence.narrative = f"Compressed {len(sorted_mems)} memories into {len(windows)} temporal windows."

    async def _strategy_counterfactual(
        self,
        sequence: DreamSequence,
        memories: List[MemoryTrace],
    ):
        """Generate what-if scenarios from memories."""
        if not self.llm_caller:
            sequence.narrative = "Counterfactual skipped: no LLM available"
            return

        # Select a memory to explore
        mem = random.choice(memories) if memories else None
        if not mem:
            return

        prompt = f"""Consider this memory:
{mem.content}

Generate 2 counterfactual scenarios:
1. What if the opposite had happened?
2. What if a key detail was different?

Explore potential insights from these alternatives."""

        try:
            response = await self.llm_caller("counterfactual", prompt)
            insights = response.get("content", "").split("\n")
            sequence.insights = [i.strip() for i in insights if i.strip() and len(i) > 10]
            sequence.narrative = f"Explored {len(sequence.insights)} counterfactual scenarios."

        except Exception as e:
            logger.error(f"Counterfactual failed: {e}")

    def _memories_related(self, m1: MemoryTrace, m2: MemoryTrace) -> bool:
        """Check if two memories are potentially related."""
        # Check existing associations
        if m1.id in m2.associations or m2.id in m1.associations:
            return True

        # Simple keyword overlap check
        words1 = set(m1.content.lower().split())
        words2 = set(m2.content.lower().split())
        overlap = len(words1 & words2)

        return overlap > 3

    # =========================================================================
    # DREAM JOURNAL
    # =========================================================================

    def get_recent_dreams(self, limit: int = 10) -> List[DreamSequence]:
        """Get recent dream sequences."""
        return sorted(
            self.dream_journal,
            key=lambda d: d.started_at,
            reverse=True,
        )[:limit]

    def get_insights(self, limit: int = 50) -> List[str]:
        """Get all insights from dream consolidation."""
        all_insights = []
        for dream in self.dream_journal:
            all_insights.extend(dream.insights)
        return all_insights[-limit:]

    def export_dream_journal(self, output_path: Path):
        """Export dream journal to file."""
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "total_dreams": len(self.dream_journal),
            "dreams": [
                {
                    "id": d.id,
                    "phase": d.phase.value,
                    "strategy": d.strategy.value,
                    "narrative": d.narrative,
                    "insights": d.insights,
                    "started_at": d.started_at.isoformat(),
                }
                for d in self.dream_journal[-100:]
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            "total_memories": len(self.memory_traces),
            "total_cycles": len(self.consolidation_cycles),
            "total_dreams": len(self.dream_journal),
            "total_insights": sum(len(d.insights) for d in self.dream_journal),
            "average_cycle_quality": (
                sum(c.quality_score for c in self.consolidation_cycles) /
                len(self.consolidation_cycles)
                if self.consolidation_cycles else 0
            ),
            "memories_by_type": {
                t.value: len([m for m in self.memory_traces.values() if m.memory_type == t])
                for t in MemoryType
            },
        }


# Singleton instance
dream_consolidator = DreamConsolidator()
