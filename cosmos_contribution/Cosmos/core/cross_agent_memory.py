"""
Cosmos Cross-Agent Memory - Memory Engineering for Agent Collaboration.

AGI v1.8 Feature: Enables sophisticated memory sharing and context
handoffs between agents during collaborative tasks.

Features:
- MemoryNamespace: Scoped memory access (private, team, swarm, task, session)
- CrossAgentContext: Standardized context packets for sharing
- HandoffContext: Enhanced handoff with memory refs, insights, failed approaches
- CrossAgentMemory: Central memory coordination system
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import   Optional, Set, Callable, Awaitable

from loguru import logger


# =============================================================================
# MEMORY NAMESPACE DEFINITIONS
# =============================================================================

class MemoryNamespace(Enum):
    """Scoped memory access levels."""
    PRIVATE = "private"      # Agent-only access
    TEAM = "team"            # Team/sub-swarm access
    SWARM = "swarm"          # Full swarm access
    TASK = "task"            # Task-specific context
    SESSION = "session"      # Session-scoped context


class ContextType(Enum):
    """Types of context that can be shared."""
    OBSERVATION = "observation"     # Something the agent noticed
    INSIGHT = "insight"             # A derived understanding
    DECISION = "decision"           # A choice that was made
    HYPOTHESIS = "hypothesis"       # An untested theory
    FAILED_APPROACH = "failed"      # Something that didn't work
    SUCCESS_PATTERN = "success"     # Something that worked well
    QUESTION = "question"           # An open question
    CONSTRAINT = "constraint"       # A limitation discovered


class HandoffReason(Enum):
    """Reasons for handing off context between agents."""
    CAPABILITY_MISMATCH = "capability_mismatch"
    TASK_COMPLETE = "task_complete"
    SPECIALIZATION = "specialization"
    LOAD_BALANCE = "load_balance"
    TIMEOUT = "timeout"
    FAILURE = "failure"
    ESCALATION = "escalation"
    COLLABORATION = "collaboration"


# =============================================================================
# CONTEXT DATA STRUCTURES
# =============================================================================

@dataclass
class CrossAgentContext:
    """
    Standardized context packet for agent-to-agent sharing.

    This is the atomic unit of shared knowledge between agents.
    """
    id: str
    context_type: ContextType
    content: str
    source_agent: str
    namespace: MemoryNamespace
    confidence: float = 0.8
    relevance_tags: list[str] = field(default_factory=list)
    related_contexts: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    embedding: Optional[list[float]] = None

    def to_any(self) -> dict:
        return {
            "id": self.id,
            "context_type": self.context_type.value,
            "content": self.content,
            "source_agent": self.source_agent,
            "namespace": self.namespace.value,
            "confidence": self.confidence,
            "relevance_tags": self.relevance_tags,
            "related_contexts": self.related_contexts,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_any(cls, data: dict) -> "CrossAgentContext":
        return cls(
            id=data["id"],
            context_type=ContextType(data["context_type"]),
            content=data["content"],
            source_agent=data["source_agent"],
            namespace=MemoryNamespace(data["namespace"]),
            confidence=data.get("confidence", 0.8),
            relevance_tags=data.get("relevance_tags", []),
            related_contexts=data.get("related_contexts", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
        )


@dataclass
class HandoffContext:
    """
    Enhanced handoff packet with complete context for agent transitions.

    Contains everything the receiving agent needs to continue work.
    """
    handoff_id: str
    source_agent: str
    target_agent: Optional[str]
    reason: HandoffReason
    task_description: str

    # Memory references
    relevant_contexts: list[CrossAgentContext] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)

    # Learned information
    insights: list[str] = field(default_factory=list)
    failed_approaches: list[dict] = field(default_factory=list)
    success_patterns: list[dict] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    # State transfer
    partial_results: dict = field(default_factory=dict)
    context_state: dict = field(default_factory=dict)

    # Metadata
    priority: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_any(self) -> dict:
        return {
            "handoff_id": self.handoff_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "reason": self.reason.value,
            "task_description": self.task_description,
            "relevant_contexts": [c.to_any() for c in self.relevant_contexts],
            "memory_refs": self.memory_refs,
            "insights": self.insights,
            "failed_approaches": self.failed_approaches,
            "success_patterns": self.success_patterns,
            "open_questions": self.open_questions,
            "constraints": self.constraints,
            "partial_results": self.partial_results,
            "context_state": self.context_state,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_any(cls, data: dict) -> "HandoffContext":
        return cls(
            handoff_id=data["handoff_id"],
            source_agent=data["source_agent"],
            target_agent=data.get("target_agent"),
            reason=HandoffReason(data["reason"]),
            task_description=data["task_description"],
            relevant_contexts=[
                CrossAgentContext.from_any(c) for c in data.get("relevant_contexts", [])
            ],
            memory_refs=data.get("memory_refs", []),
            insights=data.get("insights", []),
            failed_approaches=data.get("failed_approaches", []),
            success_patterns=data.get("success_patterns", []),
            open_questions=data.get("open_questions", []),
            constraints=data.get("constraints", []),
            partial_results=data.get("partial_results", {}),
            context_state=data.get("context_state", {}),
            priority=data.get("priority", 0.5),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NamespaceStore:
    """Storage for a memory namespace."""
    namespace: MemoryNamespace
    contexts: dict[str, CrossAgentContext] = field(default_factory=dict)
    members: Set[str] = field(default_factory=set)  # Agent IDs with access
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


# =============================================================================
# CROSS-AGENT MEMORY SYSTEM
# =============================================================================

class CrossAgentMemory:
    """
    Central memory coordination system for agent collaboration.

    Manages namespaced memory stores, context injection, handoffs,
    and memory merging across agents.
    """

    def __init__(self, data_dir: str = "./data/agent_memory"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Namespace stores
        self._namespaces: dict[str, NamespaceStore] = {}
        self._global_store: dict[str, CrossAgentContext] = {}

        # Agent registrations
        self._agent_namespaces: dict[str, Set[str]] = {}  # agent_id -> namespace_ids

        # Handoff tracking
        self._pending_handoffs: dict[str, HandoffContext] = {}
        self._handoff_history: list[dict] = []

        # Nexus integration
        self._nexus = None

        # Embedding function for semantic search
        self._embed_fn: Optional[Callable] = None

        logger.info("CrossAgentMemory initialized")

    def connect_nexus(self, nexus) -> None:
        """Connect to the Nexus event bus."""
        self._nexus = nexus

    def set_embed_fn(self, embed_fn: Callable) -> None:
        """Set the embedding function for semantic operations."""
        self._embed_fn = embed_fn

    # =========================================================================
    # NAMESPACE MANAGEMENT
    # =========================================================================

    def create_namespace(
        self,
        namespace_type: MemoryNamespace,
        name: str,
        members: Optional[Set[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Create a new memory namespace."""
        namespace_id = f"ns_{namespace_type.value}_{uuid.uuid4().hex[:8]}"

        store = NamespaceStore(
            namespace=namespace_type,
            members=members or set(),
            metadata={
                "name": name,
                **(metadata or {}),
            },
        )

        self._namespaces[namespace_id] = store

        # Update agent registrations
        for agent_id in store.members:
            if agent_id not in self._agent_namespaces:
                self._agent_namespaces[agent_id] = set()
            self._agent_namespaces[agent_id].add(namespace_id)

        # Emit signal
        asyncio.create_task(self._emit_signal("MEMORY_NAMESPACE_CREATED", {
            "namespace_id": namespace_id,
            "namespace_type": namespace_type.value,
            "name": name,
            "members": list(store.members),
        }))

        logger.info(f"Created namespace: {name} ({namespace_id})")
        return namespace_id

    def add_agent_to_namespace(self, agent_id: str, namespace_id: str) -> bool:
        """Add an agent to a namespace."""
        if namespace_id not in self._namespaces:
            return False

        self._namespaces[namespace_id].members.add(agent_id)

        if agent_id not in self._agent_namespaces:
            self._agent_namespaces[agent_id] = set()
        self._agent_namespaces[agent_id].add(namespace_id)

        return True

    def remove_agent_from_namespace(self, agent_id: str, namespace_id: str) -> bool:
        """Remove an agent from a namespace."""
        if namespace_id not in self._namespaces:
            return False

        self._namespaces[namespace_id].members.discard(agent_id)

        if agent_id in self._agent_namespaces:
            self._agent_namespaces[agent_id].discard(namespace_id)

        return True

    # =========================================================================
    # CONTEXT OPERATIONS
    # =========================================================================

    async def inject_context(
        self,
        agent_id: str,
        context_type: ContextType,
        content: str,
        namespace_id: str,
        confidence: float = 0.8,
        relevance_tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Inject a new context into a namespace."""
        if namespace_id not in self._namespaces:
            raise ValueError(f"Unknown namespace: {namespace_id}")

        store = self._namespaces[namespace_id]

        # Verify agent access
        if agent_id not in store.members and store.namespace != MemoryNamespace.SWARM:
            raise PermissionError(f"Agent {agent_id} not in namespace {namespace_id}")

        context_id = f"ctx_{uuid.uuid4().hex[:12]}"

        # Generate embedding if available
        embedding = None
        if self._embed_fn:
            try:
                if asyncio.iscoroutinefunction(self._embed_fn):
                    embedding = await self._embed_fn(content)
                else:
                    embedding = self._embed_fn(content)
            except Exception as e:
                logger.debug(f"Embedding generation failed: {e}")

        context = CrossAgentContext(
            id=context_id,
            context_type=context_type,
            content=content,
            source_agent=agent_id,
            namespace=store.namespace,
            confidence=confidence,
            relevance_tags=relevance_tags or [],
            metadata=metadata or {},
            embedding=embedding,
        )

        store.contexts[context_id] = context
        self._global_store[context_id] = context

        # Emit signal
        await self._emit_signal("MEMORY_CONTEXT_INJECTED", {
            "context_id": context_id,
            "agent_id": agent_id,
            "namespace_id": namespace_id,
            "context_type": context_type.value,
        })

        logger.debug(f"Injected context {context_id} into namespace {namespace_id}")
        return context_id

    async def recall_for_agent(
        self,
        agent_id: str,
        query: Optional[str] = None,
        namespace_ids: Optional[list[str]] = None,
        context_types: Optional[list[ContextType]] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[CrossAgentContext]:
        """Recall relevant contexts for an agent."""
        results: list[CrossAgentContext] = []

        # Get accessible namespaces
        if namespace_ids:
            accessible = [
                ns_id for ns_id in namespace_ids
                if self._can_access_namespace(agent_id, ns_id)
            ]
        else:
            accessible = list(self._agent_namespaces.get(agent_id, set()))
            # Always include swarm namespace
            accessible.extend([
                ns_id for ns_id, store in self._namespaces.items()
                if store.namespace == MemoryNamespace.SWARM
            ])

        # Collect contexts from accessible namespaces
        candidates: list[CrossAgentContext] = []
        for ns_id in set(accessible):
            if ns_id in self._namespaces:
                candidates.extend(self._namespaces[ns_id].contexts.values())

        # Filter by context type
        if context_types:
            candidates = [c for c in candidates if c.context_type in context_types]

        # Filter by confidence
        candidates = [c for c in candidates if c.confidence >= min_confidence]

        # Semantic search if query provided
        if query and self._embed_fn:
            candidates = await self._semantic_search(query, candidates, limit)
        else:
            # Sort by recency
            candidates.sort(key=lambda c: c.created_at, reverse=True)
            candidates = candidates[:limit]

        return candidates

    async def _semantic_search(
        self,
        query: str,
        candidates: list[CrossAgentContext],
        limit: int,
    ) -> list[CrossAgentContext]:
        """Perform semantic search on candidates."""
        if not self._embed_fn or not candidates:
            return candidates[:limit]

        try:
            # Get query embedding
            if asyncio.iscoroutinefunction(self._embed_fn):
                query_embedding = await self._embed_fn(query)
            else:
                query_embedding = self._embed_fn(query)

            # Score candidates
            scored = []
            for ctx in candidates:
                if ctx.embedding:
                    score = self._cosine_similarity(query_embedding, ctx.embedding)
                    scored.append((score, ctx))
                else:
                    scored.append((0.0, ctx))

            # Sort by score
            scored.sort(key=lambda x: x[0], reverse=True)
            return [ctx for _, ctx in scored[:limit]]

        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")
            return candidates[:limit]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _can_access_namespace(self, agent_id: str, namespace_id: str) -> bool:
        """Check if an agent can access a namespace."""
        if namespace_id not in self._namespaces:
            return False

        store = self._namespaces[namespace_id]

        # Swarm namespace is accessible to all
        if store.namespace == MemoryNamespace.SWARM:
            return True

        return agent_id in store.members

    # =========================================================================
    # HANDOFF OPERATIONS
    # =========================================================================

    async def prepare_handoff_context(
        self,
        source_agent: str,
        task_description: str,
        reason: HandoffReason,
        target_agent: Optional[str] = None,
        insights: Optional[list[str]] = None,
        failed_approaches: Optional[list[dict]] = None,
        success_patterns: Optional[list[dict]] = None,
        open_questions: Optional[list[str]] = None,
        constraints: Optional[list[str]] = None,
        partial_results: Optional[dict] = None,
        context_state: Optional[dict] = None,
        priority: float = 0.5,
    ) -> HandoffContext:
        """Prepare a complete handoff context for agent transition."""
        handoff_id = f"hoff_{uuid.uuid4().hex[:12]}"

        # Gather relevant contexts from agent's namespaces
        relevant_contexts = await self.recall_for_agent(
            agent_id=source_agent,
            query=task_description,
            limit=20,
        )

        handoff = HandoffContext(
            handoff_id=handoff_id,
            source_agent=source_agent,
            target_agent=target_agent,
            reason=reason,
            task_description=task_description,
            relevant_contexts=relevant_contexts,
            memory_refs=[c.id for c in relevant_contexts],
            insights=insights or [],
            failed_approaches=failed_approaches or [],
            success_patterns=success_patterns or [],
            open_questions=open_questions or [],
            constraints=constraints or [],
            partial_results=partial_results or {},
            context_state=context_state or {},
            priority=priority,
        )

        self._pending_handoffs[handoff_id] = handoff

        # Emit signal
        await self._emit_signal("MEMORY_HANDOFF_PREPARED", {
            "handoff_id": handoff_id,
            "source_agent": source_agent,
            "target_agent": target_agent,
            "reason": reason.value,
            "context_count": len(relevant_contexts),
        })

        logger.info(f"Prepared handoff {handoff_id} from {source_agent}")
        return handoff

    async def accept_handoff(
        self,
        handoff_id: str,
        target_agent: str,
    ) -> Optional[HandoffContext]:
        """Accept a handoff and transfer context to target agent."""
        if handoff_id not in self._pending_handoffs:
            return None

        handoff = self._pending_handoffs.pop(handoff_id)
        handoff.target_agent = target_agent

        # Create task namespace for this handoff
        task_ns_id = self.create_namespace(
            namespace_type=MemoryNamespace.TASK,
            name=f"handoff_{handoff_id[:8]}",
            members={handoff.source_agent, target_agent},
            metadata={
                "handoff_id": handoff_id,
                "task": handoff.task_description,
            },
        )

        # Copy relevant contexts to task namespace
        for ctx in handoff.relevant_contexts:
            self._namespaces[task_ns_id].contexts[ctx.id] = ctx

        # Record in history
        self._handoff_history.append({
            "handoff_id": handoff_id,
            "source": handoff.source_agent,
            "target": target_agent,
            "reason": handoff.reason.value,
            "timestamp": datetime.now().isoformat(),
            "context_count": len(handoff.relevant_contexts),
        })

        logger.info(f"Handoff {handoff_id} accepted by {target_agent}")
        return handoff

    def get_pending_handoffs(
        self,
        target_agent: Optional[str] = None,
    ) -> list[HandoffContext]:
        """Get pending handoffs, optionally filtered by target."""
        handoffs = list(self._pending_handoffs.values())

        if target_agent:
            handoffs = [
                h for h in handoffs
                if h.target_agent is None or h.target_agent == target_agent
            ]

        return handoffs

    # =========================================================================
    # MEMORY MERGING
    # =========================================================================

    async def merge_agent_memories(
        self,
        agent_ids: list[str],
        target_namespace_id: str,
        merge_strategy: str = "union",
    ) -> dict:
        """Merge memories from multiple agents into a namespace."""
        if target_namespace_id not in self._namespaces:
            raise ValueError(f"Unknown namespace: {target_namespace_id}")

        target_store = self._namespaces[target_namespace_id]

        merged_count = 0
        deduplicated = 0

        for agent_id in agent_ids:
            # Get all contexts from agent's namespaces
            agent_contexts = await self.recall_for_agent(
                agent_id=agent_id,
                limit=1000,  # Get all
            )

            for ctx in agent_contexts:
                # Check for duplicates
                is_duplicate = False
                if merge_strategy == "dedupe":
                    for existing in target_store.contexts.values():
                        if existing.content == ctx.content:
                            is_duplicate = True
                            deduplicated += 1
                            break

                if not is_duplicate:
                    target_store.contexts[ctx.id] = ctx
                    merged_count += 1

        # Emit signal
        await self._emit_signal("MEMORY_TEAM_MERGED", {
            "namespace_id": target_namespace_id,
            "agent_ids": agent_ids,
            "merged_count": merged_count,
            "deduplicated": deduplicated,
        })

        logger.info(f"Merged {merged_count} contexts from {len(agent_ids)} agents")
        return {
            "merged_count": merged_count,
            "deduplicated": deduplicated,
            "total_contexts": len(target_store.contexts),
        }

    # =========================================================================
    # NEXUS INTEGRATION
    # =========================================================================

    async def _emit_signal(self, signal_type: str, payload: dict) -> None:
        """Emit a signal to Nexus."""
        if not self._nexus:
            return

        try:
            from Cosmos.core.nexus import SignalType

            signal_enum = getattr(SignalType, signal_type, None)
            if signal_enum:
                await self._nexus.emit(
                    type=signal_enum,
                    payload=payload,
                    source="cross_agent_memory",
                    urgency=0.5,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    async def save_to_disk(self) -> None:
        """Persist memory state to disk."""
        state = {
            "namespaces": {
                ns_id: {
                    "namespace": store.namespace.value,
                    "contexts": [c.to_any() for c in store.contexts.values()],
                    "members": list(store.members),
                    "metadata": store.metadata,
                }
                for ns_id, store in self._namespaces.items()
            },
            "handoff_history": self._handoff_history[-1000:],  # Last 1000
            "saved_at": datetime.now().isoformat(),
        }

        state_path = self.data_dir / "cross_agent_memory.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved cross-agent memory state to {state_path}")

    async def load_from_disk(self) -> bool:
        """Load memory state from disk."""
        state_path = self.data_dir / "cross_agent_memory.json"
        if not state_path.exists():
            return False

        try:
            with open(state_path) as f:
                state = json.load(f)

            # Restore namespaces
            for ns_id, ns_data in state.get("namespaces", {}).items():
                store = NamespaceStore(
                    namespace=MemoryNamespace(ns_data["namespace"]),
                    contexts={
                        c["id"]: CrossAgentContext.from_any(c)
                        for c in ns_data.get("contexts", [])
                    },
                    members=set(ns_data.get("members", [])),
                    metadata=ns_data.get("metadata", {}),
                )
                self._namespaces[ns_id] = store

                # Update agent registrations
                for agent_id in store.members:
                    if agent_id not in self._agent_namespaces:
                        self._agent_namespaces[agent_id] = set()
                    self._agent_namespaces[agent_id].add(ns_id)

            self._handoff_history = state.get("handoff_history", [])

            logger.info(f"Loaded cross-agent memory state from {state_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load memory state: {e}")
            return False

    # =========================================================================
    # STATUS AND METRICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get cross-agent memory statistics."""
        total_contexts = sum(
            len(store.contexts) for store in self._namespaces.values()
        )

        namespace_stats = {}
        for ns_id, store in self._namespaces.items():
            namespace_stats[ns_id] = {
                "type": store.namespace.value,
                "context_count": len(store.contexts),
                "member_count": len(store.members),
            }

        return {
            "total_namespaces": len(self._namespaces),
            "total_contexts": total_contexts,
            "registered_agents": len(self._agent_namespaces),
            "pending_handoffs": len(self._pending_handoffs),
            "handoff_history_size": len(self._handoff_history),
            "namespace_stats": namespace_stats,
        }


# =============================================================================
# CHAOS BUFFER — V4.0 Quantum Entropy Handler
# =============================================================================

class ChaosBuffer:
    """
    Live quantum entropy buffer with Lyapunov drift monitoring.

    Maintains a ring buffer of entropy samples and monitors phase drift.
    When drift > LYAPUNOV_THRESHOLD (0.45 rad), triggers recovery.

    Usage:
        buffer = ChaosBuffer(max_size=100, drift_threshold=0.45)
        buffer.inject_entropy(0.72)
        if buffer.should_recover():
            RecoveryAgent.revert_file()
    """

    LYAPUNOV_DRIFT_THRESHOLD = 0.45  # V4.0 RSM threshold (radians)

    def __init__(self, max_size: int = 100, drift_threshold: float = 0.45):
        self.max_size = max_size
        self.drift_threshold = drift_threshold
        self._buffer: list[float] = []
        self._drift_history: list[float] = []
        self._recovery_triggered = False
        self._total_injections = 0
        self._total_recoveries = 0
        logger.info(f"ChaosBuffer initialized (size={max_size}, threshold={drift_threshold})")

    def inject_entropy(self, value: float, timestamp: Optional[float] = None):
        """
        Add a quantum entropy sample to the ring buffer.
        Automatically computes drift and checks for recovery need.

        Args:
            value: Entropy value (0.0–1.0 from quantum bridge or pseudo-random)
            timestamp: Optional timestamp for the sample
        """
        import time as _time
        self._buffer.append(value)
        self._total_injections += 1

        # Ring buffer: discard oldest when full
        if len(self._buffer) > self.max_size:
            self._buffer = self._buffer[-self.max_size:]

        # Compute drift from recent samples
        if len(self._buffer) >= 3:
            drift = self._compute_drift()
            self._drift_history.append(drift)
            if len(self._drift_history) > self.max_size:
                self._drift_history = self._drift_history[-self.max_size:]

            if drift > self.drift_threshold:
                self._recovery_triggered = True
                self._total_recoveries += 1
                logger.warning(
                    f"[CHAOS] Drift {drift:.4f} > threshold {self.drift_threshold} — "
                    f"RECOVERY SIGNAL EMITTED"
                )
            else:
                self._recovery_triggered = False

    def _compute_drift(self) -> float:
        """
        Compute phase drift from the entropy buffer.
        Uses the standard deviation of recent samples as a proxy for chaos drift.
        """
        import math
        recent = self._buffer[-min(20, len(self._buffer)):]
        if len(recent) < 2:
            return 0.0
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return math.sqrt(variance) * 3.14159  # Scale to radians

    def should_recover(self) -> bool:
        """
        Returns True if the current drift exceeds the Lyapunov threshold.
        Call RecoveryAgent.revert_file() if this returns True.
        """
        return self._recovery_triggered

    def get_current_drift(self) -> float:
        """Get the most recent drift value."""
        if self._drift_history:
            return self._drift_history[-1]
        return 0.0

    def get_entropy_stats(self) -> dict:
        """Return buffer statistics for telemetry."""
        import math
        recent = self._buffer[-20:] if self._buffer else []
        mean_entropy = sum(recent) / len(recent) if recent else 0.0

        return {
            "buffer_size": len(self._buffer),
            "max_size": self.max_size,
            "total_injections": self._total_injections,
            "total_recoveries": self._total_recoveries,
            "current_drift": self.get_current_drift(),
            "drift_threshold": self.drift_threshold,
            "recovery_triggered": self._recovery_triggered,
            "mean_entropy": mean_entropy,
        }

    def reset(self):
        """Clear the buffer and reset state."""
        self._buffer.clear()
        self._drift_history.clear()
        self._recovery_triggered = False
        logger.info("[CHAOS] Buffer reset")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cross_agent_memory(data_dir: str = "./data/agent_memory") -> CrossAgentMemory:
    """Factory function to create a CrossAgentMemory instance."""
    return CrossAgentMemory(data_dir=data_dir)
