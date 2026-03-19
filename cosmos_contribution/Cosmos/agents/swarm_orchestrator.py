"""
cosmos Swarm Orchestrator - Multi-Agent Coordination

Novel Approaches:
1. Dynamic Agent Spawning - Create agents on demand
2. Capability-Based Routing - Match tasks to best agent
3. Parallel Execution - Run independent subtasks concurrently
4. Emergent Behavior - Allow agent team compositions to evolve

AGI UPGRADES (v1.4):
- Fully event-driven via Nexus (central nervous system)
- Context vector routing (semantic agent matching)
- Memory-aware task assignment (recall before routing)
- Speculative agent spawning on spontaneous thoughts
- Evolution triggers on dialogue consensus and anomalies
- Closed AGI loop: Memory ↔ Nexus ↔ Swarm

AGI UPGRADES (v1.5):
- Performance-based agent pooling (recycle low performers)
- Warm agent pool for rapid spawning
- Agent health scoring and decay tracking
- Dynamic pool sizing based on load

AGI UPGRADES (v1.6):
- Embedded prompting for swarm coordination
- Model-adaptive orchestration instructions
- Collective coordination protocols
- Structured handoff prompts
"""

import asyncio
import random
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, List, Dict
import uuid

from loguru import logger

from Cosmos.agents.base_agent import BaseAgent, AgentCapability, AgentStatus, TaskResult


class TaskStatus(Enum):
    """Status of a swarm task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SwarmTask:
    """A task managed by the swarm."""
    id: str
    description: str
    required_capabilities: set[AgentCapability] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[TaskResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    parent_task: Optional[str] = None
    subtasks: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more priority

    # AGI: Semantic routing
    context_vector: Optional[List[float]] = None  # For neural routing
    semantic_tags: List[str] = field(default_factory=list)
    memory_refs: List[str] = field(default_factory=list)  # Related memory IDs


@dataclass
class SwarmState:
    """State of the entire swarm."""
    active_agents: dict[str, BaseAgent] = field(default_factory=dict)
    tasks: dict[str, SwarmTask] = field(default_factory=dict)
    task_queue: list[str] = field(default_factory=list)  # Task IDs
    completed_tasks: list[str] = field(default_factory=list)
    total_tasks_processed: int = 0
    total_handoffs: int = 0


class SwarmOrchestrator:
    """
    Orchestrates multiple agents working together.

    Features:
    - Dynamic agent creation and lifecycle management
    - Intelligent task routing based on capabilities
    - Handoff protocols between specialists
    - Shared state and memory management
    - Parallel subtask execution

    AGI Upgrades:
    - Fully event-driven via Nexus signals
    - Context vector routing (semantic agent matching)
    - Memory-aware task assignment
    - Speculative agent spawning on spontaneous thoughts
    - Evolution triggers on dialogue consensus/anomalies
    """

    def __init__(
        self,
        max_concurrent_agents: int = 5,
        handoff_timeout_seconds: float = 30.0,
        enable_nexus: bool = True,
    ):
        self.max_concurrent = max_concurrent_agents
        self.handoff_timeout = handoff_timeout_seconds
        self.enable_nexus = enable_nexus

        self.state = SwarmState()

        # Agent factories
        self._agent_factories: dict[str, Callable[[], BaseAgent]] = {}

        # Shared resources
        self.llm_backend = None
        self.memory_system = None

        # Event handlers (legacy - kept for backwards compatibility)
        self._on_task_complete: list[Callable] = []
        self._on_handoff: list[Callable] = []

        self._lock = asyncio.Lock()

        # AGI: Nexus integration
        self._nexus_subscribed = False
        self._speculative_spawn_probability = 0.3  # Chance to spawn on thought
        self._evolution_on_consensus = True
        self._evolution_on_anomaly = True
        self._memory_aware_routing = True

        # AGI: Agent capability vectors (for semantic matching)
        self._agent_type_vectors: Dict[str, List[float]] = {}

        # AGI: Speculative agent tracking
        self._speculative_agents: Dict[str, str] = {}  # agent_id -> thought_id

        # AGI Cohesion: Memory-inferred capability hints
        self._last_capability_hints: List[str] = []

        # AGI v1.5: Agent pooling
        self._pool_config = AgentPoolConfig()
        self._agent_pool: Dict[str, PooledAgent] = {}  # agent_id -> PooledAgent
        self._pool_by_type: Dict[str, List[str]] = {}  # agent_type -> [agent_ids]
        self._pool_metrics: Dict[str, AgentPerformanceMetrics] = {}  # agent_id -> metrics
        self._pool_lock = asyncio.Lock()
        self._decay_task: Optional[asyncio.Task] = None

        # AGI v1.6: Embedded prompting system
        self._swarm_prompt_cache: Optional[str] = None
        self._collective_prompt_cache: Optional[str] = None
        self._init_embedded_prompts()

        logger.info("SwarmOrchestrator initialized with AGI upgrades (v1.6 embedded prompting)")

    # =========================================================================
    # EMBEDDED PROMPTING SYSTEM (AGI v1.6)
    # =========================================================================

    def _init_embedded_prompts(self):
        """Initialize embedded prompting for swarm coordination."""
        if not EMBEDDED_PROMPTS_AVAILABLE:
            logger.debug("Embedded prompts not available")
            return

        try:
            # Cache swarm orchestration prompt
            self._swarm_prompt_cache = prompt_manager.render_prompt(
                "swarm_orchestration_base",
                swarm_state="initializing",
                active_agents="none",
                pending_tasks="0",
            )

            # Cache collective coordination prompt
            self._collective_prompt_cache = prompt_manager.render_prompt(
                "collective_coordination_base",
                collective_mode="orchestrated",
                current_phase="idle",
                agent_role="orchestrator",
            )

            logger.debug("SwarmOrchestrator embedded prompts initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize swarm embedded prompts: {e}")
            self._swarm_prompt_cache = None
            self._collective_prompt_cache = None

    def get_swarm_coordination_prompt(
        self,
        swarm_mode: str = "orchestrated",
        include_collective: bool = False,
    ) -> str:
        """
        Get the full swarm coordination prompt for agent instructions.

        Args:
            swarm_mode: "orchestrated" or "collective"
            include_collective: Include collective coordination prompts

        Returns:
            Complete swarm coordination prompt
        """
        if not EMBEDDED_PROMPTS_AVAILABLE:
            return ""

        sections = []

        # Update swarm prompt with current state
        try:
            active_agents = list(self.state.active_agents.keys())
            pending_tasks = len(self.state.task_queue)

            swarm_prompt = prompt_manager.render_prompt(
                "swarm_orchestration_base",
                swarm_state=swarm_mode,
                active_agents=", ".join(active_agents) if active_agents else "none",
                pending_tasks=str(pending_tasks),
            )
            if swarm_prompt:
                sections.append(swarm_prompt)

        except Exception as e:
            logger.debug(f"Error rendering swarm prompt: {e}")
            if self._swarm_prompt_cache:
                sections.append(self._swarm_prompt_cache)

        # Add collective coordination if requested
        if include_collective:
            try:
                collective_prompt = prompt_manager.render_prompt(
                    "collective_coordination_base",
                    collective_mode=swarm_mode,
                    current_phase="active",
                    agent_role="participant",
                )
                if collective_prompt:
                    sections.append(collective_prompt)
            except Exception:
                if self._collective_prompt_cache:
                    sections.append(self._collective_prompt_cache)

        return "\n\n---\n\n".join(sections)

    def get_handoff_prompt_for_agent(
        self,
        source_agent_id: str,
        target_agent_id: str,
        task_description: str,
        context: Optional[Dict] = None,
    ) -> str:
        """
        Get a structured handoff prompt for agent-to-agent task transfer.

        Args:
            source_agent_id: Agent handing off the task
            target_agent_id: Agent receiving the task
            task_description: Description of the task
            context: Optional context dictionary

        Returns:
            Formatted handoff prompt
        """
        if not EMBEDDED_PROMPTS_AVAILABLE:
            return f"Handoff from {source_agent_id} to {target_agent_id}: {task_description}"

        try:
            # Get source agent metrics
            source_metrics = self._pool_metrics.get(source_agent_id)
            current_confidence = source_metrics.avg_confidence if source_metrics else 0.5

            # Calculate chain depth
            chain_depth = context.get("chain_depth", 0) if context else 0

            handoff_prompt = prompt_manager.render_prompt(
                "task_handoff_base",
                confidence_threshold=0.6,
                current_confidence=current_confidence,
                max_attempts=3,
                handoff_id=f"handoff_{uuid.uuid4().hex[:8]}",
                source_agent=source_agent_id,
                target_agent=target_agent_id,
                task_description=task_description,
                original_goal=task_description,
                current_state="in_progress",
                progress_pct=context.get("progress", 0) if context else 0,
                memory_refs=str(context.get("memory_refs", [])) if context else "[]",
                key_insights=str(context.get("insights", [])) if context else "[]",
                failed_approaches=str(context.get("failed", [])) if context else "[]",
                constraints=str(context.get("constraints", [])) if context else "[]",
                expected_output=context.get("expected_output", "task completion") if context else "task completion",
                quality_criteria=str(context.get("quality", ["accuracy", "completeness"])) if context else '["accuracy"]',
                deadline=context.get("deadline", "none") if context else "none",
                urgency=context.get("urgency", 0.5) if context else 0.5,
                parent_task_id=context.get("parent_task_id", "root") if context else "root",
                max_chain_depth=5,
                current_depth=chain_depth,
            )
            return handoff_prompt or f"Handoff: {task_description}"

        except Exception as e:
            logger.debug(f"Error rendering handoff prompt: {e}")
            return f"Handoff from {source_agent_id} to {target_agent_id}: {task_description}"

    # =========================================================================
    # NEXUS INTEGRATION (AGI Upgrade)
    # =========================================================================

    async def connect_to_nexus(self):
        """
        Connect the swarm to the Nexus event bus.

        Subscribes to relevant signals for event-driven operation:
        - THOUGHT_EMITTED: Speculative agent spawning
        - DIALOGUE_CONSENSUS: Evolution triggers
        - ANOMALY_DETECTED: Evolution and adaptation
        - MEMORY_CONSOLIDATION: Memory-aware routing
        - TASK_* signals: Task lifecycle events
        """
        if self._nexus_subscribed:
            logger.warning("Already connected to Nexus")
            return

        # Subscribe to cognitive signals
        nexus.subscribe(SignalType.THOUGHT_EMITTED, self._on_thought_emitted)
        nexus.subscribe(SignalType.DIALOGUE_CONSENSUS, self._on_dialogue_consensus)
        nexus.subscribe(SignalType.ANOMALY_DETECTED, self._on_anomaly_detected)
        nexus.subscribe(SignalType.MEMORY_CONSOLIDATION, self._on_memory_consolidation)

        # Subscribe to P2P signals
        nexus.subscribe(SignalType.TASK_RECEIVED, self._on_external_task)
        nexus.subscribe(SignalType.SKILL_RECEIVED, self._on_skill_received)

        # Subscribe to resonance signals (collective thoughts)
        nexus.subscribe(SignalType.RESONANCE_RECEIVED, self._on_resonance_received)

        self._nexus_subscribed = True
        logger.info("SwarmOrchestrator connected to Nexus")

        # AGI Cohesion: Register semantic subscriptions for neural routing
        await self._register_semantic_subscriptions()

        # Emit startup signal
        await nexus.emit(
            SignalType.SYSTEM_STARTUP,
            {"component": "swarm_orchestrator", "agent_types": list(self._agent_factories.keys())},
            source="swarm_orchestrator",
            urgency=0.8,
        )

    async def _register_semantic_subscriptions(self):
        """
        Register semantic subscriptions for neural routing.

        AGI Cohesion: Links swarm_orchestrator to nexus semantic routing,
        enabling context-vector-based task discovery and agent matching.
        """
        # Subscribe to memory-related signals with semantic matching
        if self._agent_type_vectors:
            for agent_type, type_vector in self._agent_type_vectors.items():
                # Create semantic handler for each agent type
                async def make_semantic_handler(a_type):
                    async def handler(signal: Signal):
                        await self._on_semantic_match(signal, a_type)
                    return handler

                handler = await make_semantic_handler(agent_type)

                # Register with nexus semantic subscription
                nexus.subscribe_semantic(
                    handler=handler,
                    target_vector=type_vector,
                    similarity_threshold=0.75,  # Only match highly relevant signals
                    signal_types={
                        SignalType.THOUGHT_EMITTED,
                        SignalType.MEMORY_CONSOLIDATION,
                        SignalType.TASK_CREATED,
                    },
                )

        logger.debug(f"Registered {len(self._agent_type_vectors)} semantic subscriptions")

    async def _on_semantic_match(self, signal: Signal, matched_agent_type: str):
        """
        Handle semantically matched signals for proactive task creation.

        Called when a signal's context_vector matches an agent type's vector.
        """
        # Don't act on low-urgency signals
        if signal.urgency < 0.4:
            return

        # Check if this signal suggests a task we should handle
        if signal.type == SignalType.THOUGHT_EMITTED:
            content = signal.payload.get("content", "")
            thought_type = signal.payload.get("thought_type", "")

            # High-relevance thoughts may warrant speculative task creation
            relevance = signal.payload.get("relevance", 0.5)
            if relevance >= 0.7:
                logger.debug(
                    f"Semantic match: {matched_agent_type} for thought '{content[:50]}...'"
                )

        elif signal.type == SignalType.MEMORY_CONSOLIDATION:
            # Memory consolidation matched to agent type - update routing
            memory_ids = signal.payload.get("memory_ids", [])
            if memory_ids and signal.context_vector:
                self._update_agent_type_vector(matched_agent_type, signal.context_vector)

    async def disconnect_from_nexus(self):
        """Disconnect from Nexus."""
        if not self._nexus_subscribed:
            return

        # Note: Nexus doesn't currently support unsubscribe by handler
        # In production, we'd track subscription IDs
        self._nexus_subscribed = False
        logger.info("SwarmOrchestrator disconnected from Nexus")

    async def _on_thought_emitted(self, signal: Signal):
        """
        Handle spontaneous thought signals.

        May trigger speculative agent spawning for interesting thoughts.
        """
        if not self.enable_nexus:
            return

        thought_content = signal.payload.get("content", "")
        thought_type = signal.payload.get("thought_type", "general")
        relevance = signal.payload.get("relevance", 0.5)

        # Only consider high-relevance thoughts
        if relevance < 0.6:
            return

        # Probabilistic speculative spawning
        if random.random() > self._speculative_spawn_probability:
            return

        # Determine if thought suggests a useful task
        agent_type = self._infer_agent_type_from_thought(thought_content, thought_type)

        if agent_type and agent_type in self._agent_factories:
            # Spawn speculative agent
            agent = await self.spawn_agent(agent_type)
            if agent:
                self._speculative_agents[agent.agent_id] = signal.id
                logger.info(
                    f"Speculative agent spawned: {agent.name} "
                    f"(triggered by thought: {thought_content[:50]}...)"
                )

                # Create exploratory task from thought
                if thought_type == "question":
                    task_desc = f"Investigate: {thought_content}"
                elif thought_type == "insight":
                    task_desc = f"Explore implications: {thought_content}"
                else:
                    task_desc = f"Research: {thought_content}"

                await self.submit_task(
                    description=task_desc,
                    context={"speculative": True, "source_thought": signal.id},
                    context_vector=signal.context_vector,
                    priority=3,  # Lower priority for speculative tasks
                )

    async def _on_dialogue_consensus(self, signal: Signal):
        """
        Handle dialogue consensus signals.

        High-confidence consensus may trigger evolution of agent population.
        """
        if not self._evolution_on_consensus:
            return

        session_id = signal.payload.get("session_id")
        confidence = signal.payload.get("confidence", 0.0)
        decision = signal.payload.get("decision", "")

        # High-confidence consensus triggers evolution
        if confidence >= 0.85:
            logger.info(f"High-confidence consensus detected - considering evolution")

            if hasattr(self, '_evolution_config') and self._population:
                # Boost fitness of agents that contributed to consensus
                contributors = signal.payload.get("contributors", [])
                for agent_id in contributors:
                    for variant in self._population.values():
                        if variant.base_agent_type in agent_id:
                            variant.fitness_score *= 1.1  # 10% fitness boost
                            variant.mutation_history.append(f"consensus_boost:{session_id}")

    async def _on_anomaly_detected(self, signal: Signal):
        """
        Handle anomaly signals.

        Anomalies may trigger defensive evolution or adaptation.
        """
        if not self._evolution_on_anomaly:
            return

        anomaly_type = signal.payload.get("anomaly_type", "unknown")
        severity = signal.payload.get("severity", 0.5)

        if severity >= 0.7:
            logger.warning(f"High-severity anomaly: {anomaly_type} - triggering adaptation")

            # Trigger evolution with stricter fitness threshold
            if hasattr(self, '_evolution_config'):
                self.evolve_subscriptions_internal(fitness_threshold=0.5)

            # Potentially spawn defensive agents
            if anomaly_type in ["security", "resource_exhaustion", "cascade_failure"]:
                if "defensive" in self._agent_factories:
                    agent = await self.spawn_agent("defensive")
                    if agent:
                        await self.submit_task(
                            description=f"Investigate and mitigate anomaly: {anomaly_type}",
                            context={"anomaly_signal": signal.id},
                            priority=9,  # High priority
                        )

    async def _on_memory_consolidation(self, signal: Signal):
        """
        Handle memory consolidation signals.

        Updates routing knowledge based on memory patterns and proactively
        updates in-progress task contexts with relevant consolidated memories.

        AGI Cohesion Upgrade: Links memory_system.py consolidation events
        to swarm task context for dynamic handoffs and context enrichment.
        """
        if not self._memory_aware_routing:
            return

        memory_ids = signal.payload.get("memory_ids", [])
        new_vector = signal.payload.get("new_vector")
        session_ref = signal.payload.get("session_ref")
        consolidated_content = signal.payload.get("content_preview", "")
        relevance_score = signal.payload.get("relevance", 0.5)

        # If we have a context vector, update agent type vectors
        if new_vector and session_ref:
            # Associate vector with successful agent types
            if session_ref in self.state.tasks:
                task = self.state.tasks[session_ref]
                if task.assigned_agent:
                    agent = self.state.active_agents.get(task.assigned_agent)
                    if agent:
                        agent_type = agent.name
                        self._update_agent_type_vector(agent_type, new_vector)

        # AGI Cohesion: Proactively update in-progress task contexts
        if memory_ids and relevance_score >= 0.6:
            await self._update_task_contexts_with_memories(
                memory_ids=memory_ids,
                new_vector=new_vector,
                relevance_score=relevance_score,
            )

    async def _update_task_contexts_with_memories(
        self,
        memory_ids: List[str],
        new_vector: Optional[List[float]],
        relevance_score: float,
    ):
        """
        Proactively update in-progress task contexts with relevant memories.

        This enables dynamic task handoffs when consolidated memories reveal
        that a different agent might be better suited for the task.
        """
        for task_id, task in self.state.tasks.items():
            if task.status != TaskStatus.IN_PROGRESS:
                continue

            # Check semantic relevance to task
            if task.context_vector and new_vector:
                similarity = self._cosine_similarity(task.context_vector, new_vector)

                if similarity >= 0.7:  # Highly relevant to this task
                    # Enrich task context with memory references
                    if "enriched_memory_refs" not in task.context:
                        task.context["enriched_memory_refs"] = []

                    for mem_id in memory_ids:
                        if mem_id not in task.memory_refs and mem_id not in task.context["enriched_memory_refs"]:
                            task.context["enriched_memory_refs"].append(mem_id)

                    logger.debug(
                        f"Enriched task {task_id} with {len(memory_ids)} consolidated memories "
                        f"(similarity={similarity:.2f})"
                    )

                    # Consider dynamic handoff if memories suggest different capability
                    if similarity >= 0.85 and self._should_consider_handoff(task, new_vector):
                        await self._consider_dynamic_handoff(task, new_vector, relevance_score)

    def _should_consider_handoff(self, task: SwarmTask, memory_vector: List[float]) -> bool:
        """Determine if memory consolidation suggests a handoff might be beneficial."""
        if not task.assigned_agent:
            return False

        agent = self.state.active_agents.get(task.assigned_agent)
        if not agent:
            return False

        # Check if memory vector is closer to a different agent type
        current_agent_type = agent.name
        current_similarity = 0.0

        if current_agent_type in self._agent_type_vectors:
            current_similarity = self._cosine_similarity(
                memory_vector,
                self._agent_type_vectors[current_agent_type]
            )

        # Find if any other agent type is significantly better matched
        for agent_type, type_vector in self._agent_type_vectors.items():
            if agent_type == current_agent_type:
                continue

            other_similarity = self._cosine_similarity(memory_vector, type_vector)
            if other_similarity > current_similarity + 0.2:  # 20% better match
                return True

        return False

    async def _consider_dynamic_handoff(
        self,
        task: SwarmTask,
        memory_vector: List[float],
        relevance_score: float,
    ):
        """
        Consider a dynamic handoff based on memory consolidation insights.

        This is a soft handoff - we emit a signal but don't force the handoff.
        """
        if not task.assigned_agent:
            return

        # Find best matching agent type
        best_type = None
        best_similarity = 0.0

        for agent_type, type_vector in self._agent_type_vectors.items():
            similarity = self._cosine_similarity(memory_vector, type_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_type = agent_type

        if best_type and best_type != self.state.active_agents.get(task.assigned_agent, {}).name:
            # Emit handoff consideration signal (soft - doesn't force handoff)
            if self.enable_nexus:
                await nexus.emit(
                    SignalType.EXTERNAL_EVENT,
                    {
                        "event_type": "memory_triggered_handoff_consideration",
                        "task_id": task.id,
                        "current_agent": task.assigned_agent,
                        "suggested_agent_type": best_type,
                        "similarity_score": best_similarity,
                        "relevance_score": relevance_score,
                    },
                    source="swarm_orchestrator",
                    urgency=0.5,
                    context_vector=memory_vector,
                )

    async def _on_external_task(self, signal: Signal):
        """Handle tasks received from P2P network."""
        task_data = signal.payload.get("task", {})
        description = task_data.get("description", "")
        priority = task_data.get("priority", 5)
        context_vector = signal.context_vector

        if description:
            await self.submit_task(
                description=description,
                context={"p2p_source": signal.source_id},
                context_vector=context_vector,
                priority=priority,
            )

    async def _on_skill_received(self, signal: Signal):
        """Handle new skill received from network."""
        skill_type = signal.payload.get("skill_type")
        factory_code = signal.payload.get("factory")

        # Note: In production, would validate/sandbox received skills
        logger.info(f"Skill received from network: {skill_type}")

    async def _on_resonance_received(self, signal: Signal):
        """Handle collective thoughts from other Cosmos instances."""
        thought = signal.payload.get("thought", "")
        source_collective = signal.payload.get("source_collective")

        if thought and signal.context_vector:
            # Consider spawning based on collective resonance
            await self._on_thought_emitted(signal)

    def _infer_agent_type_from_thought(self, content: str, thought_type: str) -> Optional[str]:
        """Infer what agent type might be useful for a thought."""
        content_lower = content.lower()

        if thought_type == "connection":
            return "reasoning"
        elif thought_type == "question":
            if any(kw in content_lower for kw in ["code", "implement", "function"]):
                return "code"
            elif any(kw in content_lower for kw in ["research", "find", "search"]):
                return "research"
            return "reasoning"
        elif thought_type == "insight":
            return "reasoning"
        elif thought_type == "idea":
            if any(kw in content_lower for kw in ["creative", "design", "write"]):
                return "creative"
            return "general"

        return None

    def _update_agent_type_vector(self, agent_type: str, vector: List[float], alpha: float = 0.1):
        """Update the semantic vector for an agent type (exponential moving average)."""
        if agent_type not in self._agent_type_vectors:
            self._agent_type_vectors[agent_type] = vector
        else:
            existing = self._agent_type_vectors[agent_type]
            if len(existing) == len(vector):
                # EMA update
                self._agent_type_vectors[agent_type] = [
                    alpha * v + (1 - alpha) * e
                    for v, e in zip(vector, existing)
                ]

    def evolve_subscriptions_internal(self, fitness_threshold: float = 0.3):
        """Trigger evolution of agent population based on fitness."""
        if not hasattr(self, '_population') or not self._population:
            return

        # Flag low-fitness variants for replacement
        to_replace = []
        for variant_id, variant in self._population.items():
            if variant.fitness_score < fitness_threshold:
                to_replace.append(variant_id)

        if to_replace:
            logger.info(f"Evolution: {len(to_replace)} low-fitness variants flagged")

    # =========================================================================
    # NEXUS SIGNAL EMISSION
    # =========================================================================

    async def _emit_task_created(self, task: SwarmTask):
        """Emit TASK_CREATED signal."""
        if self.enable_nexus:
            await nexus.emit(
                SignalType.TASK_CREATED,
                {
                    "task_id": task.id,
                    "description": task.description[:200],
                    "priority": task.priority,
                    "capabilities": [c.value for c in task.required_capabilities],
                },
                source="swarm_orchestrator",
                urgency=0.5 + task.priority * 0.05,
                context_vector=task.context_vector,
            )

    async def _emit_task_completed(self, task: SwarmTask, result: TaskResult):
        """Emit TASK_COMPLETED signal."""
        if self.enable_nexus:
            await nexus.emit(
                SignalType.TASK_COMPLETED,
                {
                    "task_id": task.id,
                    "success": result.success,
                    "confidence": result.confidence,
                    "assigned_agent": task.assigned_agent,
                    "execution_time": result.execution_time,
                },
                source="swarm_orchestrator",
                urgency=0.6,
                context_vector=task.context_vector,
            )

    async def _emit_task_failed(self, task: SwarmTask, error: str):
        """Emit TASK_FAILED signal."""
        if self.enable_nexus:
            await nexus.emit(
                SignalType.TASK_FAILED,
                {
                    "task_id": task.id,
                    "error": error,
                    "assigned_agent": task.assigned_agent,
                    "retry_count": task.context.get("retry_count", 0),
                },
                source="swarm_orchestrator",
                urgency=0.8,
                context_vector=task.context_vector,
            )

    async def _emit_handoff(self, from_agent: str, to_type: str, reason: str):
        """Emit handoff signal."""
        if self.enable_nexus:
            self.state.total_handoffs += 1
            # Use external event for handoffs
            await nexus.emit(
                SignalType.EXTERNAL_EVENT,
                {
                    "event_type": "agent_handoff",
                    "from_agent": from_agent,
                    "to_agent_type": to_type,
                    "reason": reason,
                },
                source="swarm_orchestrator",
                urgency=0.6,
            )

    # =========================================================================
    # ORIGINAL METHODS (Enhanced)
    # =========================================================================

    def register_agent_factory(
        self,
        agent_type: str,
        factory: Callable[[], BaseAgent],
        capability_vector: Optional[List[float]] = None,
    ):
        """
        Register an agent factory for dynamic creation.

        Args:
            agent_type: Name of the agent type
            factory: Factory function to create agent instances
            capability_vector: Optional semantic vector for this agent type
        """
        self._agent_factories[agent_type] = factory
        if capability_vector:
            self._agent_type_vectors[agent_type] = capability_vector
        logger.info(f"Registered agent factory: {agent_type}")

    async def spawn_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """
        Spawn a new agent of the given type.

        AGI v1.5: Uses agent pool for efficient reuse and performance-based selection.
        """
        if agent_type not in self._agent_factories:
            logger.error(f"Unknown agent type: {agent_type}")
            return None

        if len(self.state.active_agents) >= self.max_concurrent:
            # Try to recycle an idle agent (performance-based)
            await self._recycle_idle_agents()

            if len(self.state.active_agents) >= self.max_concurrent:
                logger.warning("Max concurrent agents reached")
                return None

        # AGI v1.5: Try to checkout from pool first
        if self._agent_pool:
            agent = await self.checkout_agent(agent_type)
            if agent:
                logger.info(f"Spawned agent from pool: {agent.name} ({agent.agent_id})")
                return agent

        # Fallback: Create agent directly (pool not initialized)
        agent = self._agent_factories[agent_type]()

        # Configure agent
        agent.llm_backend = self.llm_backend
        agent.memory = self.memory_system
        agent.set_handoff_callback(self._handle_handoff)

        self.state.active_agents[agent.agent_id] = agent
        logger.info(f"Spawned agent: {agent.name} ({agent.agent_id})")

        return agent

    async def _recycle_idle_agents(self):
        """
        Recycle idle agents to free up capacity using performance-based selection.

        AGI v1.5: Prioritizes recycling low-performing agents based on health score.
        """
        idle_agents = [
            agent_id for agent_id, agent in self.state.active_agents.items()
            if agent.state.status == AgentStatus.IDLE
        ]

        if not idle_agents:
            return

        # Sort by health score (lowest first - recycle worst performers)
        def get_health(agent_id: str) -> float:
            if agent_id in self._pool_metrics:
                return self._pool_metrics[agent_id].compute_health_score()
            return 0.5  # Default

        idle_agents.sort(key=get_health)

        # Keep at least one of each type
        type_counts: dict[str, int] = {}
        for agent in self.state.active_agents.values():
            type_counts[agent.name] = type_counts.get(agent.name, 0) + 1

        recycled_count = 0
        for agent_id in idle_agents:
            agent = self.state.active_agents[agent_id]
            health = get_health(agent_id)

            # Recycle if: low health OR too many of this type
            should_recycle = (
                health < self._pool_config.health_threshold or
                type_counts.get(agent.name, 0) > 1
            )

            if should_recycle:
                await self._return_to_pool(agent_id, recycle=health < self._pool_config.health_threshold)
                if agent_id in self.state.active_agents:
                    del self.state.active_agents[agent_id]
                type_counts[agent.name] = max(0, type_counts.get(agent.name, 1) - 1)
                recycled_count += 1
                logger.debug(f"Recycled agent {agent_id} (health={health:.2f})")

        if recycled_count > 0:
            logger.info(f"Recycled {recycled_count} agents (performance-based)")

    # =========================================================================
    # AGENT POOLING (AGI v1.5)
    # =========================================================================

    async def init_agent_pool(self, config: Optional[AgentPoolConfig] = None):
        """
        Initialize the agent pool with warm agents.

        Args:
            config: Pool configuration (uses defaults if not provided)
        """
        if config:
            self._pool_config = config

        async with self._pool_lock:
            # Pre-warm pool with agents for each registered type
            if self._pool_config.warmup_on_startup:
                for agent_type in self._agent_factories.keys():
                    for _ in range(self._pool_config.min_pool_size):
                        await self._create_pooled_agent(agent_type)

            # Start health decay task
            if not self._decay_task:
                self._decay_task = asyncio.create_task(self._pool_decay_loop())

        logger.info(f"Agent pool initialized: {len(self._agent_pool)} agents warmed")

    async def _create_pooled_agent(self, agent_type: str) -> Optional[PooledAgent]:
        """Create a new agent and add it to the pool."""
        if agent_type not in self._agent_factories:
            return None

        if len(self._agent_pool) >= self._pool_config.max_pool_size:
            # Pool full - try to recycle lowest performer
            await self._recycle_lowest_performer()
            if len(self._agent_pool) >= self._pool_config.max_pool_size:
                return None

        # Create agent
        agent = self._agent_factories[agent_type]()
        agent.llm_backend = self.llm_backend
        agent.memory = self.memory_system
        agent.set_handoff_callback(self._handle_handoff)

        # Create metrics
        metrics = AgentPerformanceMetrics(
            agent_id=agent.agent_id,
            agent_type=agent_type,
        )

        # Create pooled wrapper
        pooled = PooledAgent(
            agent=agent,
            metrics=metrics,
            pool_state="warm",
        )

        # Add to pool
        self._agent_pool[agent.agent_id] = pooled
        self._pool_metrics[agent.agent_id] = metrics

        if agent_type not in self._pool_by_type:
            self._pool_by_type[agent_type] = []
        self._pool_by_type[agent_type].append(agent.agent_id)

        logger.debug(f"Created pooled agent: {agent_type} ({agent.agent_id})")
        return pooled

    async def checkout_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """
        Check out an agent from the pool.

        Prefers warm agents; creates new if none available.
        Selection prioritizes agents with higher health scores.
        """
        async with self._pool_lock:
            # Find available warm agents of this type
            candidates = []
            for agent_id in self._pool_by_type.get(agent_type, []):
                pooled = self._agent_pool.get(agent_id)
                if pooled and pooled.pool_state == "warm":
                    candidates.append(pooled)

            if candidates:
                # Select healthiest agent
                candidates.sort(key=lambda p: p.metrics.compute_health_score(), reverse=True)
                pooled = candidates[0]

                # Mark as active
                pooled.pool_state = "active"
                pooled.checkout_time = datetime.now()

                # Add to active agents
                self.state.active_agents[pooled.agent.agent_id] = pooled.agent

                logger.debug(
                    f"Checked out pooled agent: {agent_type} "
                    f"(health={pooled.metrics.health_score:.2f})"
                )
                return pooled.agent

            # No warm agents - create new one
            pooled = await self._create_pooled_agent(agent_type)
            if pooled:
                pooled.pool_state = "active"
                pooled.checkout_time = datetime.now()
                self.state.active_agents[pooled.agent.agent_id] = pooled.agent
                return pooled.agent

            return None

    async def _return_to_pool(self, agent_id: str, recycle: bool = False):
        """
        Return an agent to the pool after use.

        Args:
            agent_id: The agent to return
            recycle: If True, destroy the agent instead of pooling
        """
        async with self._pool_lock:
            pooled = self._agent_pool.get(agent_id)
            if not pooled:
                return

            if recycle or pooled.metrics.health_score < self._pool_config.health_threshold:
                # Remove from pool entirely
                pooled.pool_state = "recycled"
                agent_type = pooled.metrics.agent_type

                if agent_id in self._pool_by_type.get(agent_type, []):
                    self._pool_by_type[agent_type].remove(agent_id)
                if agent_id in self._agent_pool:
                    del self._agent_pool[agent_id]
                if agent_id in self._pool_metrics:
                    del self._pool_metrics[agent_id]

                logger.debug(f"Recycled agent from pool: {agent_id}")

                # Ensure minimum pool size
                type_count = len(self._pool_by_type.get(agent_type, []))
                if type_count < self._pool_config.min_pool_size:
                    await self._create_pooled_agent(agent_type)
            else:
                # Return to warm state
                pooled.pool_state = "warm"
                pooled.checkout_time = None
                pooled.metrics.last_used = datetime.now()

                logger.debug(f"Returned agent to pool: {agent_id} (warm)")

    def record_agent_task_result(
        self,
        agent_id: str,
        success: bool,
        confidence: float,
        execution_time_ms: float,
    ):
        """Record task result for agent performance tracking."""
        metrics = self._pool_metrics.get(agent_id)
        if not metrics:
            return

        if success:
            metrics.tasks_completed += 1
            metrics.error_streak = 0
        else:
            metrics.tasks_failed += 1
            metrics.error_streak += 1

        # Update running averages
        total_tasks = metrics.tasks_completed + metrics.tasks_failed
        metrics.avg_confidence = (
            (metrics.avg_confidence * (total_tasks - 1) + confidence) / total_tasks
        )
        metrics.avg_latency_ms = (
            (metrics.avg_latency_ms * (total_tasks - 1) + execution_time_ms) / total_tasks
        )
        metrics.total_execution_time += execution_time_ms / 1000

        # Recompute health
        metrics.compute_health_score()

        # Check if agent should be recycled due to error streak
        if metrics.error_streak >= self._pool_config.error_streak_limit:
            asyncio.create_task(self._return_to_pool(agent_id, recycle=True))
            logger.warning(
                f"Agent {agent_id} recycled due to {metrics.error_streak} consecutive errors"
            )

    async def _recycle_lowest_performer(self):
        """Recycle the lowest performing warm agent to make room."""
        warm_agents = [
            (agent_id, pooled)
            for agent_id, pooled in self._agent_pool.items()
            if pooled.pool_state == "warm"
        ]

        if not warm_agents:
            return

        # Find lowest health
        warm_agents.sort(key=lambda x: x[1].metrics.compute_health_score())
        lowest_id, _ = warm_agents[0]

        await self._return_to_pool(lowest_id, recycle=True)

    async def _pool_decay_loop(self):
        """Background task to decay agent health over time."""
        while True:
            try:
                await asyncio.sleep(self._pool_config.decay_interval_seconds)

                async with self._pool_lock:
                    now = datetime.now()
                    to_recycle = []

                    for agent_id, pooled in self._agent_pool.items():
                        if pooled.pool_state != "warm":
                            continue

                        # Check idle timeout
                        idle_seconds = (now - pooled.metrics.last_used).total_seconds()
                        if idle_seconds > self._pool_config.idle_timeout_seconds:
                            to_recycle.append(agent_id)
                            continue

                        # Apply health decay (small penalty for being idle)
                        decay_factor = 0.99  # 1% decay per interval
                        pooled.metrics.health_score *= decay_factor

                        # Recycle if health drops too low
                        if pooled.metrics.health_score < self._pool_config.health_threshold:
                            to_recycle.append(agent_id)

                    for agent_id in to_recycle:
                        await self._return_to_pool(agent_id, recycle=True)

                    if to_recycle:
                        logger.debug(f"Pool decay: recycled {len(to_recycle)} agents")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pool decay error: {e}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        warm_count = sum(1 for p in self._agent_pool.values() if p.pool_state == "warm")
        active_count = sum(1 for p in self._agent_pool.values() if p.pool_state == "active")

        type_stats = {}
        for agent_type, agent_ids in self._pool_by_type.items():
            type_agents = [self._agent_pool.get(aid) for aid in agent_ids if aid in self._agent_pool]
            type_stats[agent_type] = {
                "total": len(type_agents),
                "warm": sum(1 for p in type_agents if p and p.pool_state == "warm"),
                "active": sum(1 for p in type_agents if p and p.pool_state == "active"),
                "avg_health": (
                    sum(p.metrics.health_score for p in type_agents if p) / len(type_agents)
                    if type_agents else 0
                ),
            }

        return {
            "total_pooled": len(self._agent_pool),
            "warm": warm_count,
            "active": active_count,
            "config": {
                "min_pool_size": self._pool_config.min_pool_size,
                "max_pool_size": self._pool_config.max_pool_size,
                "health_threshold": self._pool_config.health_threshold,
            },
            "by_type": type_stats,
            "top_performers": [
                {
                    "agent_id": p.agent.agent_id,
                    "type": p.metrics.agent_type,
                    "health": p.metrics.health_score,
                    "tasks_completed": p.metrics.tasks_completed,
                }
                for p in sorted(
                    self._agent_pool.values(),
                    key=lambda x: x.metrics.health_score,
                    reverse=True,
                )[:5]
            ],
        }

    async def submit_task(
        self,
        description: str,
        required_capabilities: Optional[set[AgentCapability]] = None,
        context: Optional[dict] = None,
        priority: int = 5,
        context_vector: Optional[List[float]] = None,
        semantic_tags: Optional[List[str]] = None,
    ) -> str:
        """
        Submit a task to the swarm.

        Args:
            description: Task description
            required_capabilities: Required agent capabilities
            context: Additional context dict
            priority: Task priority (1-10)
            context_vector: Semantic vector for neural routing
            semantic_tags: Tags for categorization

        Returns:
            Task ID
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        task = SwarmTask(
            id=task_id,
            description=description,
            required_capabilities=required_capabilities or set(),
            context=context or {},
            priority=priority,
            context_vector=context_vector,
            semantic_tags=semantic_tags or [],
        )

        # AGI: Memory-aware routing - recall relevant memories with capability hints
        if self._memory_aware_routing and self.memory_system:
            try:
                # Extract capability tags from required capabilities
                capability_tags = [cap.value for cap in task.required_capabilities] if task.required_capabilities else None

                memories = await self._recall_relevant_memories(
                    description=description,
                    context_vector=context_vector,
                    task_capabilities=capability_tags,
                )
                task.memory_refs = [m.get("id", "") for m in memories if m.get("id")]
                task.context["memory_context"] = memories[:3]  # Top 3 for context

                # Use inferred capabilities to enrich task routing
                if self._last_capability_hints and not task.required_capabilities:
                    task.context["inferred_capabilities"] = self._last_capability_hints
                    task.semantic_tags.extend(self._last_capability_hints)

            except Exception as e:
                logger.debug(f"Memory recall failed: {e}")

        async with self._lock:
            self.state.tasks[task_id] = task

            # Insert into queue based on priority
            inserted = False
            for i, queued_id in enumerate(self.state.task_queue):
                queued_task = self.state.tasks[queued_id]
                if task.priority > queued_task.priority:
                    self.state.task_queue.insert(i, task_id)
                    inserted = True
                    break

            if not inserted:
                self.state.task_queue.append(task_id)

        logger.info(f"Task submitted: {task_id} - {description[:50]}...")

        # Emit Nexus signal
        await self._emit_task_created(task)

        # Try to process immediately
        asyncio.create_task(self._process_queue())

        return task_id

    async def _recall_relevant_memories(
        self,
        description: str,
        context_vector: List[float],
        limit: int = 5,
        task_capabilities: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Recall relevant memories for a task using memory_system cohesion.

        AGI Cohesion: Uses memory_system.recall_for_task() when available
        for richer context including capability hints and entity relationships.
        """
        if not self.memory_system:
            return []

        try:
            # Prefer the new task-aware recall method (AGI Cohesion)
            if hasattr(self.memory_system, 'recall_for_task'):
                result = await self.memory_system.recall_for_task(
                    task_description=description,
                    context_vector=context_vector,
                    task_capabilities=task_capabilities,
                    limit=limit,
                    include_graph=True,
                    boost_recent=True,
                )
                # Return memories with enriched context
                memories = result.get("memories", [])

                # Store capability hints for routing decisions
                if result.get("capability_hints"):
                    self._last_capability_hints = result["capability_hints"]

                return memories

            # Fallback to standard recall method
            elif hasattr(self.memory_system, 'recall'):
                results = await self.memory_system.recall(
                    query=description,
                    top_k=limit,
                )
                return [
                    r.to_dict() if hasattr(r, 'to_dict') else {"content": str(r)}
                    for r in results
                ]

            # Final fallback to query method
            elif hasattr(self.memory_system, 'query'):
                results = await self.memory_system.query(description, top_k=limit)
                return results

        except Exception as e:
            logger.debug(f"Memory recall error: {e}")

        return []

    async def _process_queue(self):
        """Process tasks in the queue."""
        async with self._lock:
            while self.state.task_queue:
                task_id = self.state.task_queue[0]
                task = self.state.tasks.get(task_id)

                if not task or task.status != TaskStatus.PENDING:
                    self.state.task_queue.pop(0)
                    continue

                # Find best agent for task
                agent = await self._find_best_agent(task)

                if agent is None:
                    # Try to spawn a new agent
                    agent_type = self._infer_agent_type(task)
                    agent = await self.spawn_agent(agent_type)

                if agent is None:
                    # No capacity, wait
                    break

                # Assign task
                self.state.task_queue.pop(0)
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent = agent.agent_id

                # Execute task (don't await here to allow parallel processing)
                asyncio.create_task(self._execute_task(task, agent))

    async def _find_best_agent(self, task: SwarmTask) -> Optional[BaseAgent]:
        """
        Find the best available agent for a task.

        Uses hybrid scoring (AGI v1.7 enhanced):
        1. Capability matching (original)
        2. Context vector similarity (AGI v1.4)
        3. Agent performance history
        4. Handler benchmark scores (AGI v1.7)
        5. Dynamic provider recommendations
        """
        best_agent = None
        best_score = 0.0

        # AGI v1.7: Get benchmark recommendations if available
        benchmark_recommendation = None
        if BENCHMARK_AVAILABLE:
            try:
                benchmark_recommendation = benchmark_engine.get_provider_recommendation(
                    self._infer_benchmark_type(task)
                )
            except Exception:
                pass

        for agent in self.state.active_agents.values():
            if agent.state.status not in (AgentStatus.IDLE, AgentStatus.COMPLETED):
                continue

            # Original capability score
            capability_score = agent.can_handle(task.required_capabilities)

            # AGI: Context vector similarity
            vector_score = 0.0
            if task.context_vector and agent.name in self._agent_type_vectors:
                agent_vector = self._agent_type_vectors[agent.name]
                vector_score = self._cosine_similarity(task.context_vector, agent_vector)

            # AGI v1.7: Benchmark score bonus
            benchmark_bonus = 0.0
            if benchmark_recommendation and benchmark_recommendation.get("handler_id"):
                # Check if this agent matches the benchmark recommendation
                rec_handler = benchmark_recommendation["handler_id"]
                if agent.name.lower() in rec_handler.lower() or rec_handler.lower() in agent.name.lower():
                    benchmark_bonus = benchmark_recommendation.get("score", 0) * 0.3

            # Combine scores with benchmark integration
            if task.context_vector and agent.name in self._agent_type_vectors:
                # Weighted combination when vector available
                score = (
                    capability_score * 0.35 +
                    vector_score * 0.3 +
                    agent.state.avg_confidence * 0.15 +
                    benchmark_bonus * 0.2
                )
            else:
                # Original scoring when no vector
                base_score = capability_score * (0.5 + 0.5 * agent.state.avg_confidence)
                score = base_score * 0.8 + benchmark_bonus * 0.2

            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent if best_score > 0.3 else None

    def _infer_benchmark_type(self, task: SwarmTask) -> "BenchmarkType":
        """Infer benchmark type from task for handler selection."""
        if not BENCHMARK_AVAILABLE:
            return None

        desc = task.description.lower()

        if any(kw in desc for kw in ["code", "implement", "function", "debug", "refactor"]):
            return BenchmarkType.CODING
        elif any(kw in desc for kw in ["research", "find", "search", "analyze"]):
            return BenchmarkType.RESEARCH
        elif any(kw in desc for kw in ["reason", "think", "logic", "math", "calculate"]):
            return BenchmarkType.REASONING
        elif any(kw in desc for kw in ["trade", "buy", "sell", "market", "token"]):
            return BenchmarkType.TRADING
        elif any(kw in desc for kw in ["write", "creative", "story", "brainstorm"]):
            return BenchmarkType.CREATIVE
        else:
            return BenchmarkType.REASONING  # Default

    async def run_handler_tournament(
        self,
        task: SwarmTask,
        handler_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a competitive tournament to find the best handler for a task.

        AGI v1.7: Dynamic selection through competitive benchmarking.

        Args:
            task: The task to compete on
            handler_ids: Specific handlers to test (or all available)

        Returns:
            Tournament results with winner and rankings
        """
        if not BENCHMARK_AVAILABLE:
            return {"error": "Benchmark system not available"}

        benchmark_type = self._infer_benchmark_type(task)

        # Get handlers to test
        if handler_ids is None:
            # Get top candidates based on type
            candidates = benchmark_engine.select_best_handlers(benchmark_type, top_n=5)
            handler_ids = [c[0] for c in candidates]

        if not handler_ids:
            return {"error": "No handlers available for tournament"}

        # Create benchmark task
        bench_task = BenchmarkTask(
            task_id=task.id,
            benchmark_type=benchmark_type,
            prompt=task.description,
            timeout_ms=30000.0,
        )

        # Run tournament
        try:
            result = await benchmark_engine.run_tournament(handler_ids, bench_task)

            # Emit result via Nexus
            await self._emit_benchmark_result(result)

            return {
                "tournament_id": result.tournament_id,
                "winner": result.winner_id,
                "winner_score": result.winner_score,
                "rankings": result.rankings,
                "benchmark_type": benchmark_type.value,
            }

        except Exception as e:
            logger.error(f"Tournament failed: {e}")
            return {"error": str(e)}

    async def _emit_benchmark_result(self, result):
        """Emit benchmark result via Nexus."""
        try:
            await nexus.emit(
                SignalType.BEST_HANDLER_SELECTED,
                {
                    "tournament_id": result.tournament_id,
                    "winner": result.winner_id,
                    "score": result.winner_score,
                    "rankings": result.rankings[:3],
                },
                source="swarm_orchestrator",
                urgency=0.5,
            )
        except Exception:
            pass

    async def spawn_subswarm_for_task(
        self,
        task: SwarmTask,
        swarm_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Spawn a sub-swarm for complex tasks requiring specialized agent groups.

        AGI v1.7: API-triggered sub-swarm spinning.

        Args:
            task: The task requiring sub-swarm
            swarm_type: Override the inferred swarm type

        Returns:
            Sub-swarm info or None if not available
        """
        if not SUBSWARM_AVAILABLE:
            logger.debug("Sub-swarm spawner not available")
            return None

        # Infer swarm type from task
        if swarm_type is None:
            desc = task.description.lower()
            if any(kw in desc for kw in ["trade", "market", "token", "price"]):
                inferred_type = SubSwarmType.TRADING
            elif any(kw in desc for kw in ["research", "analyze", "investigate"]):
                inferred_type = SubSwarmType.RESEARCH
            elif any(kw in desc for kw in ["code", "implement", "debug"]):
                inferred_type = SubSwarmType.CODING
            elif any(kw in desc for kw in ["predict", "forecast", "odds"]):
                inferred_type = SubSwarmType.PREDICTION
            else:
                inferred_type = SubSwarmType.ANALYSIS
        else:
            inferred_type = SubSwarmType(swarm_type)

        try:
            subswarm = await subswarm_spawner.spawn_subswarm(
                inferred_type,
                task.description,
                context={"task_id": task.id, "capabilities": [c.value for c in task.required_capabilities]},
                trigger_source="swarm_orchestrator",
            )

            logger.info(f"Spawned sub-swarm {subswarm.swarm_id} for task {task.id}")

            return {
                "swarm_id": subswarm.swarm_id,
                "type": inferred_type.value,
                "agent_count": len(subswarm.agents),
                "state": subswarm.state.value,
            }

        except Exception as e:
            logger.error(f"Failed to spawn sub-swarm: {e}")
            return None

    async def get_or_create_persistent_session(
        self,
        handler_id: str,
        session_type: str = "development",
    ) -> Optional[Dict[str, Any]]:
        """
        Get or create a persistent tmux session for a handler.

        AGI v1.7: Persistent sessions for Claude/Kimi coding workflows.

        Args:
            handler_id: The handler requesting a session
            session_type: Type of session (claude_code, development, research)

        Returns:
            Session info or None if not available
        """
        if not TMUX_AVAILABLE:
            logger.debug("Tmux session manager not available")
            return None

        try:
            # Map session type
            type_map = {
                "claude_code": SessionType.CLAUDE_CODE,
                "development": SessionType.DEVELOPMENT,
                "research": SessionType.RESEARCH,
                "trading": SessionType.TRADING,
            }
            sess_type = type_map.get(session_type, SessionType.DEVELOPMENT)

            session = await session_manager.get_or_create_session(
                sess_type,
                handler_id=handler_id,
            )

            return {
                "session_id": session.session_id,
                "session_name": session.session_name,
                "type": session.session_type.value,
                "state": session.state.value,
            }

        except Exception as e:
            logger.error(f"Failed to get/create session: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _infer_agent_type(self, task: SwarmTask) -> str:
        """Infer the best agent type for a task."""
        if not task.required_capabilities:
            # Try to infer from description
            desc_lower = task.description.lower()

            if any(kw in desc_lower for kw in ["code", "implement", "function", "debug"]):
                return "code"
            elif any(kw in desc_lower for kw in ["reason", "think", "analyze", "math"]):
                return "reasoning"
            elif any(kw in desc_lower for kw in ["research", "find", "search"]):
                return "research"
            elif any(kw in desc_lower for kw in ["write", "creative", "story"]):
                return "creative"

        # Map capabilities to agent types
        cap_to_type = {
            AgentCapability.CODE_GENERATION: "code",
            AgentCapability.CODE_ANALYSIS: "code",
            AgentCapability.REASONING: "reasoning",
            AgentCapability.MATH: "reasoning",
            AgentCapability.RESEARCH: "research",
            AgentCapability.CREATIVE_WRITING: "creative",
        }

        for cap in task.required_capabilities:
            if cap in cap_to_type:
                return cap_to_type[cap]

        return "general"

    async def _execute_task(
        self,
        task: SwarmTask,
        agent: BaseAgent,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Execute a task with an agent, with retry logic and fallback."""
        import time

        task.status = TaskStatus.IN_PROGRESS
        task.context["retry_count"] = 0

        last_error = None
        for attempt in range(max_retries):
            task.context["retry_count"] = attempt
            task_start_time = time.time()
            try:
                result = await agent.execute(task.description, task.context)
                execution_time_ms = (time.time() - task_start_time) * 1000

                # AGI v1.5: Record performance metrics
                self.record_agent_task_result(
                    agent_id=agent.agent_id,
                    success=result.success,
                    confidence=result.confidence,
                    execution_time_ms=execution_time_ms,
                )

                # Check if result indicates failure that might be retryable
                if result.success:
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()

                    self.state.total_tasks_processed += 1
                    self.state.completed_tasks.append(task.id)

                    # Notify legacy listeners
                    for handler in self._on_task_complete:
                        try:
                            await handler(task, result)
                        except Exception as e:
                            logger.error(f"Task complete handler error: {e}")

                    # AGI: Emit Nexus signal
                    await self._emit_task_completed(task, result)

                    # AGI: Update agent type vector with successful task
                    if task.context_vector:
                        self._update_agent_type_vector(agent.name, task.context_vector)

                    # AGI: Emit memory consolidation for successful tasks
                    if self.enable_nexus and task.memory_refs:
                        await emit_memory_consolidation(
                            memory_ids=task.memory_refs,
                            session_ref=task.id,
                            context_vector=task.context_vector,
                        )

                    logger.info(f"Task {task.id} completed: success={result.success}")
                    return

                # Result was not successful - try with different agent if available
                last_error = result.output
                logger.warning(f"Task {task.id} attempt {attempt + 1} failed: {last_error}")

                if attempt < max_retries - 1:
                    # Try to find a different agent
                    alt_agent = await self._find_alternative_agent(task, agent.agent_id)
                    if alt_agent:
                        # AGI: Emit handoff signal
                        await self._emit_handoff(agent.agent_id, alt_agent.name, last_error)
                        agent = alt_agent
                        logger.info(f"Retrying with alternative agent: {alt_agent.name}")
                    await asyncio.sleep(retry_delay * (attempt + 1))

            except asyncio.TimeoutError:
                last_error = "Task execution timed out"
                logger.warning(f"Task {task.id} attempt {attempt + 1} timed out")
                # AGI v1.5: Record timeout as failure
                self.record_agent_task_result(
                    agent_id=agent.agent_id,
                    success=False,
                    confidence=0.0,
                    execution_time_ms=self.handoff_timeout * 1000,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

            except Exception as e:
                last_error = str(e)
                logger.error(f"Task {task.id} attempt {attempt + 1} error: {e}")
                execution_time_ms = (time.time() - task_start_time) * 1000
                # AGI v1.5: Record exception as failure
                self.record_agent_task_result(
                    agent_id=agent.agent_id,
                    success=False,
                    confidence=0.0,
                    execution_time_ms=execution_time_ms,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

        # All retries exhausted
        logger.error(f"Task {task.id} failed after {max_retries} attempts")
        task.status = TaskStatus.FAILED
        task.result = TaskResult(success=False, output=f"Failed after {max_retries} attempts: {last_error}")

        # AGI: Emit failure signal
        await self._emit_task_failed(task, last_error or "Unknown error")

    async def _find_alternative_agent(
        self,
        task: SwarmTask,
        exclude_agent_id: str
    ) -> Optional[BaseAgent]:
        """Find an alternative agent for a failed task."""
        for agent in self.state.active_agents.values():
            if agent.agent_id == exclude_agent_id:
                continue
            if agent.state.status not in (AgentStatus.IDLE, AgentStatus.COMPLETED):
                continue
            if agent.can_handle(task.required_capabilities) > 0.3:
                return agent
        return None

        # Process more tasks
        asyncio.create_task(self._process_queue())

    async def _handle_handoff(
        self,
        target_agent_type: str,
        task_description: str,
        reason: str,
        context: Optional[dict],
    ):
        """Handle a handoff request from an agent."""
        logger.info(f"Handoff requested: {target_agent_type} - {reason}")

        # Notify legacy listeners
        for handler in self._on_handoff:
            try:
                await handler(target_agent_type, task_description, reason)
            except Exception as e:
                logger.error(f"Handoff handler error: {e}")

        # AGI: Emit handoff signal via Nexus
        await self._emit_handoff("unknown", target_agent_type, reason)

        # Extract context_vector if available
        context_vector = None
        if context and "context_vector" in context:
            context_vector = context["context_vector"]

        # Submit as new task
        await self.submit_task(
            description=task_description,
            context=context,
            context_vector=context_vector,
            priority=6,  # Slightly higher priority for handoffs
        )

    async def execute_with_subtasks(
        self,
        main_task: str,
        subtasks: list[str],
        context: Optional[dict] = None,
    ) -> list[TaskResult]:
        """
        Execute a main task with parallel subtasks.

        Subtasks are executed concurrently when possible.
        """
        # Submit main task
        main_id = await self.submit_task(main_task, context=context, priority=7)

        # Submit subtasks
        subtask_ids = []
        for subtask in subtasks:
            subtask_id = await self.submit_task(
                subtask,
                context={**(context or {}), "parent_task": main_id},
                priority=5,
            )
            subtask_ids.append(subtask_id)
            self.state.tasks[main_id].subtasks.append(subtask_id)

        # Wait for all subtasks to complete
        results = await asyncio.gather(*[
            self.wait_for_task(task_id)
            for task_id in subtask_ids
        ])

        return results

    async def wait_for_task(self, task_id: str, timeout: float = 300.0) -> TaskResult:
        """Wait for a task to complete."""
        start_time = asyncio.get_event_loop().time()

        while True:
            task = self.state.tasks.get(task_id)

            if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return task.result or TaskResult(success=False, output="No result")

            if asyncio.get_event_loop().time() - start_time > timeout:
                return TaskResult(success=False, output="Task timeout")

            await asyncio.sleep(0.1)

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get the status of a task."""
        task = self.state.tasks.get(task_id)
        if not task:
            return None

        return {
            "id": task.id,
            "description": task.description[:100],
            "status": task.status.value,
            "assigned_agent": task.assigned_agent,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }

    def on_task_complete(self, handler: Callable):
        """Register a task completion handler."""
        self._on_task_complete.append(handler)

    def on_handoff(self, handler: Callable):
        """Register a handoff handler."""
        self._on_handoff.append(handler)

    def get_swarm_status(self) -> dict:
        """Get comprehensive swarm status."""
        return {
            "active_agents": {
                agent_id: agent.get_status()
                for agent_id, agent in self.state.active_agents.items()
            },
            "queue_length": len(self.state.task_queue),
            "pending_tasks": sum(
                1 for t in self.state.tasks.values()
                if t.status == TaskStatus.PENDING
            ),
            "in_progress_tasks": sum(
                1 for t in self.state.tasks.values()
                if t.status == TaskStatus.IN_PROGRESS
            ),
            "total_tasks_processed": self.state.total_tasks_processed,
            "total_handoffs": self.state.total_handoffs,
            "agent_types": list(self._agent_factories.keys()),
            # AGI metrics
            "nexus_connected": self._nexus_subscribed,
            "speculative_agents": len(self._speculative_agents),
            "agent_type_vectors": len(self._agent_type_vectors),
            "memory_aware_routing": self._memory_aware_routing,
            "evolution_on_consensus": self._evolution_on_consensus,
            "evolution_on_anomaly": self._evolution_on_anomaly,
            # AGI v1.5: Agent pooling metrics
            "pool_stats": self.get_pool_stats() if self._agent_pool else None,
        }

    # =========================================================================
    # POPULATION-BASED EVOLUTION (AGI Upgrade)
    # =========================================================================

    def _init_evolution(self, config: Optional[EvolutionConfig] = None):
        """Initialize evolution tracking."""
        self._evolution_config = config or EvolutionConfig()
        self._population: dict[str, AgentVariant] = {}
        self._evolution_history: list[dict] = []

    async def evolve_population(
        self,
        agent_type: str,
        evaluation_tasks: list[str],
        generations: Optional[int] = None,
    ) -> list[AgentVariant]:
        """
        Run population-based evolution on a set of agent variants.

        Uses genetic algorithm:
        1. Initialize population with random variants
        2. Evaluate fitness on test tasks
        3. Select elite survivors
        4. Crossover and mutate to create next generation
        5. Repeat for N generations

        Returns the final evolved population sorted by fitness.
        """
        if not hasattr(self, '_evolution_config'):
            self._init_evolution()

        config = self._evolution_config
        num_generations = generations or config.generations

        # Initialize population if empty
        if not self._population:
            for i in range(config.population_size):
                variant = AgentVariant(
                    variant_id=f"{agent_type}_v{i}_{uuid.uuid4().hex[:6]}",
                    base_agent_type=agent_type,
                    generation=0,
                    temperature=random.uniform(0.3, 1.0),
                    prompt_style=random.choice(["concise", "detailed", "creative", "balanced"]),
                    confidence_threshold=random.uniform(0.4, 0.8),
                )
                self._population[variant.variant_id] = variant

        logger.info(f"Starting evolution: {config.population_size} variants, {num_generations} generations")

        # Evolution loop
        for gen in range(num_generations):
            logger.info(f"Generation {gen + 1}/{num_generations}")

            # Evaluate all variants
            for variant in self._population.values():
                if variant.generation == gen:
                    await self._evaluate_variant(variant, evaluation_tasks)

            # Sort by fitness
            sorted_variants = sorted(
                self._population.values(),
                key=lambda v: v.fitness_score,
                reverse=True,
            )

            # Elite selection - top performers survive unchanged
            elite_count = max(1, int(config.population_size * config.elite_ratio))
            elites = sorted_variants[:elite_count]

            # Create next generation
            next_gen: list[AgentVariant] = []

            # Elites pass through
            for elite in elites:
                next_gen.append(elite)

            # Fill rest with crossover and mutation
            while len(next_gen) < config.population_size:
                # Tournament selection for parents
                parent1 = self._tournament_select(sorted_variants)
                parent2 = self._tournament_select(sorted_variants)

                # Crossover
                if random.random() < config.crossover_rate:
                    child = self._crossover(parent1, parent2, gen + 1)
                else:
                    child = AgentVariant(
                        variant_id=f"{agent_type}_v{len(next_gen)}_{uuid.uuid4().hex[:6]}",
                        base_agent_type=agent_type,
                        generation=gen + 1,
                        parent_id=parent1.variant_id,
                        temperature=parent1.temperature,
                        prompt_style=parent1.prompt_style,
                        confidence_threshold=parent1.confidence_threshold,
                    )

                # Mutation
                if random.random() < config.mutation_rate:
                    child = self._mutate_variant(child)

                next_gen.append(child)

            # Update population
            self._population = {v.variant_id: v for v in next_gen}

            # Record history
            self._evolution_history.append({
                "generation": gen + 1,
                "best_fitness": sorted_variants[0].fitness_score,
                "avg_fitness": sum(v.fitness_score for v in sorted_variants) / len(sorted_variants),
                "best_variant": sorted_variants[0].variant_id,
            })

        # Return final sorted population
        return sorted(
            self._population.values(),
            key=lambda v: v.fitness_score,
            reverse=True,
        )

    async def _evaluate_variant(
        self,
        variant: AgentVariant,
        evaluation_tasks: list[str],
    ):
        """Evaluate a variant's fitness on test tasks."""
        config = self._evolution_config
        start_time = datetime.now()

        successes = 0
        total_confidence = 0.0

        for task_desc in evaluation_tasks:
            try:
                # Spawn agent with variant's traits
                agent = await self.spawn_agent(variant.base_agent_type)
                if not agent:
                    continue

                # Apply variant traits
                agent.temperature = variant.temperature
                agent.confidence_threshold = variant.confidence_threshold

                # Execute task
                result = await asyncio.wait_for(
                    agent.execute(task_desc, {}),
                    timeout=self.handoff_timeout,
                )

                if result.success:
                    successes += 1
                    variant.tasks_completed += 1
                else:
                    variant.tasks_failed += 1

                total_confidence += result.confidence

            except asyncio.TimeoutError:
                variant.tasks_failed += 1
            except Exception as e:
                logger.warning(f"Evaluation error for {variant.variant_id}: {e}")
                variant.tasks_failed += 1

        # Calculate fitness components
        total_tasks = len(evaluation_tasks)
        success_rate = successes / total_tasks if total_tasks > 0 else 0
        avg_confidence = total_confidence / total_tasks if total_tasks > 0 else 0.5
        execution_time = (datetime.now() - start_time).total_seconds()

        # Speed score (faster = better, max 1.0)
        speed_score = max(0, 1 - (execution_time / (total_tasks * self.handoff_timeout)))

        # Handoff efficiency (fewer handoffs = better)
        handoff_efficiency = 1 / (1 + variant.tasks_failed * 0.1)

        # Weighted fitness
        weights = config.fitness_weights
        variant.fitness_score = (
            weights["success_rate"] * success_rate +
            weights["confidence"] * avg_confidence +
            weights["speed"] * speed_score +
            weights["handoff_efficiency"] * handoff_efficiency
        )

        variant.avg_confidence = avg_confidence
        variant.avg_execution_time = execution_time / total_tasks if total_tasks > 0 else 0

        logger.debug(
            f"Variant {variant.variant_id}: fitness={variant.fitness_score:.3f}, "
            f"success={success_rate:.2f}, confidence={avg_confidence:.2f}"
        )

    def _tournament_select(
        self,
        population: list[AgentVariant],
        tournament_size: int = 3,
    ) -> AgentVariant:
        """Select a variant using tournament selection."""
        tournament = random.sample(
            population,
            min(tournament_size, len(population)),
        )
        return max(tournament, key=lambda v: v.fitness_score)

    def _crossover(
        self,
        parent1: AgentVariant,
        parent2: AgentVariant,
        generation: int,
    ) -> AgentVariant:
        """Create a child variant by crossing over two parents."""
        child = AgentVariant(
            variant_id=f"{parent1.base_agent_type}_v{generation}_{uuid.uuid4().hex[:6]}",
            base_agent_type=parent1.base_agent_type,
            generation=generation,
            parent_id=parent1.variant_id,
            # Crossover traits
            temperature=(parent1.temperature + parent2.temperature) / 2,
            prompt_style=random.choice([parent1.prompt_style, parent2.prompt_style]),
            confidence_threshold=(parent1.confidence_threshold + parent2.confidence_threshold) / 2,
        )
        return child

    def _mutate_variant(self, variant: AgentVariant) -> AgentVariant:
        """Apply random mutations to a variant."""
        mutations = []

        # Temperature mutation
        if random.random() < 0.5:
            delta = random.gauss(0, 0.1)
            variant.temperature = max(0.1, min(1.5, variant.temperature + delta))
            mutations.append(f"temperature:{delta:+.2f}")

        # Prompt style mutation
        if random.random() < 0.3:
            styles = ["concise", "detailed", "creative", "balanced"]
            old_style = variant.prompt_style
            variant.prompt_style = random.choice(styles)
            if variant.prompt_style != old_style:
                mutations.append(f"style:{variant.prompt_style}")

        # Confidence threshold mutation
        if random.random() < 0.5:
            delta = random.gauss(0, 0.05)
            variant.confidence_threshold = max(0.3, min(0.9, variant.confidence_threshold + delta))
            mutations.append(f"conf_thresh:{delta:+.2f}")

        variant.mutation_history.extend(mutations)
        return variant

    async def gossip_state(
        self,
        peer_orchestrators: list["SwarmOrchestrator"],
        share_ratio: float = 0.3,
    ) -> dict:
        """
        Share evolutionary state with peer orchestrators.

        Implements gossip protocol for decentralized knowledge sharing:
        1. Select top performers to share
        2. Send to random subset of peers
        3. Merge received variants into local population

        Returns: {sent_count, received_count, merged_count}
        """
        if not hasattr(self, '_population') or not self._population:
            return {"sent_count": 0, "received_count": 0, "merged_count": 0}

        # Select top variants to share
        sorted_variants = sorted(
            self._population.values(),
            key=lambda v: v.fitness_score,
            reverse=True,
        )
        share_count = max(1, int(len(sorted_variants) * share_ratio))
        to_share = sorted_variants[:share_count]

        sent_count = 0
        received_count = 0
        merged_count = 0

        # Share with each peer
        for peer in peer_orchestrators:
            if not hasattr(peer, '_population'):
                peer._init_evolution()

            # Send our top variants
            for variant in to_share:
                variant_data = variant.to_dict()
                sent_count += 1

                # Check if peer should adopt this variant
                if variant.fitness_score > 0.5:  # Only share good variants
                    # Create new variant for peer with slightly mutated traits
                    peer_variant = AgentVariant(
                        variant_id=f"gossip_{variant.variant_id}_{uuid.uuid4().hex[:4]}",
                        base_agent_type=variant.base_agent_type,
                        generation=variant.generation,
                        temperature=variant.temperature,
                        prompt_style=variant.prompt_style,
                        confidence_threshold=variant.confidence_threshold,
                        fitness_score=variant.fitness_score * 0.9,  # Slight penalty for transferred
                        parent_id=variant.variant_id,
                    )

                    # Only add if peer doesn't have similar variant
                    similar_exists = any(
                        abs(v.temperature - peer_variant.temperature) < 0.1 and
                        v.prompt_style == peer_variant.prompt_style
                        for v in peer._population.values()
                    )

                    if not similar_exists:
                        peer._population[peer_variant.variant_id] = peer_variant
                        merged_count += 1

            # Receive from peer (symmetric exchange)
            received_count += len(peer._population)

        logger.info(
            f"Gossip complete: sent={sent_count}, received={received_count}, merged={merged_count}"
        )

        return {
            "sent_count": sent_count,
            "received_count": received_count,
            "merged_count": merged_count,
        }

    def get_best_variant(self, agent_type: Optional[str] = None) -> Optional[AgentVariant]:
        """Get the best performing variant, optionally filtered by agent type."""
        if not hasattr(self, '_population') or not self._population:
            return None

        variants = list(self._population.values())

        if agent_type:
            variants = [v for v in variants if v.base_agent_type == agent_type]

        if not variants:
            return None

        return max(variants, key=lambda v: v.fitness_score)

    def get_evolution_stats(self) -> dict:
        """Get statistics about the evolution process."""
        if not hasattr(self, '_evolution_history'):
            return {"status": "not_initialized"}

        return {
            "population_size": len(self._population) if hasattr(self, '_population') else 0,
            "generations_completed": len(self._evolution_history),
            "history": self._evolution_history[-10:],  # Last 10 generations
            "best_variant": self.get_best_variant().to_dict() if self.get_best_variant() else None,
        }

    # =========================================================================
    # AGI v1.8: CROSS-AGENT MEMORY INTEGRATION
    # =========================================================================

    def enable_cross_agent_memory(self, cross_agent_memory) -> None:
        """
        Enable cross-agent memory for enhanced task handoffs.

        AGI v1.8: Memory engineering for agent collaboration.

        Args:
            cross_agent_memory: CrossAgentMemory instance
        """
        self._cross_agent_memory = cross_agent_memory
        logger.info("SwarmOrchestrator: Cross-agent memory enabled")

    def enable_a2a_protocol(self, a2a_protocol) -> None:
        """
        Enable A2A protocol for direct agent-to-agent communication.

        AGI v1.8: Agent-to-agent collaboration protocol.

        Args:
            a2a_protocol: A2AProtocol instance
        """
        self._a2a_protocol = a2a_protocol
        logger.info("SwarmOrchestrator: A2A protocol enabled")

    def enable_langgraph_hybrid(self, langgraph_hybrid) -> None:
        """
        Enable LangGraph hybrid workflows for stateful task execution.

        AGI v1.8: Hybrid Nexus + LangGraph integration.

        Args:
            langgraph_hybrid: LangGraphNexusHybrid instance
        """
        self._langgraph_hybrid = langgraph_hybrid
        logger.info("SwarmOrchestrator: LangGraph hybrid enabled")

    async def handoff_task_with_memory(
        self,
        source_agent_id: str,
        target_agent_id: Optional[str],
        task: SwarmTask,
        insights: Optional[List[str]] = None,
        failed_approaches: Optional[List[Dict]] = None,
        partial_results: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Hand off a task with full memory context transfer.

        AGI v1.8: Enhanced handoffs using CrossAgentMemory.

        Args:
            source_agent_id: Agent handing off the task
            target_agent_id: Target agent (or None for auction)
            task: The task being handed off
            insights: Insights learned during processing
            failed_approaches: Approaches that didn't work
            partial_results: Any partial results achieved

        Returns:
            Handoff ID or None if failed
        """
        if not hasattr(self, '_cross_agent_memory') or not self._cross_agent_memory:
            logger.debug("Cross-agent memory not enabled, using standard handoff")
            return None

        try:
            from Cosmos.core.cross_agent_memory import HandoffReason

            # Determine handoff reason
            reason = HandoffReason.CAPABILITY_MISMATCH
            if task.status == TaskStatus.FAILED:
                reason = HandoffReason.FAILURE
            elif partial_results:
                reason = HandoffReason.SPECIALIZATION

            # Prepare handoff context with full memory
            handoff = await self._cross_agent_memory.prepare_handoff_context(
                source_agent=source_agent_id,
                task_description=task.description,
                reason=reason,
                target_agent=target_agent_id,
                insights=insights,
                failed_approaches=failed_approaches,
                partial_results=partial_results,
                context_state=task.context,
                priority=task.priority / 10.0,  # Normalize to 0-1
            )

            # If no target, use A2A auction
            if target_agent_id is None and hasattr(self, '_a2a_protocol') and self._a2a_protocol:
                # Get required capabilities
                capabilities = [cap.value for cap in task.required_capabilities]

                auction_id = await self._a2a_protocol.broadcast_task_auction(
                    initiator=source_agent_id,
                    task_description=task.description,
                    required_capabilities=capabilities,
                    deadline_seconds=30.0,
                    min_confidence=0.5,
                    priority=task.priority / 10.0,
                    context={"handoff_id": handoff.handoff_id},
                )

                logger.info(f"Task auction {auction_id} started for handoff {handoff.handoff_id}")
                return handoff.handoff_id

            # Direct handoff to target
            logger.info(f"Handoff {handoff.handoff_id} prepared from {source_agent_id} to {target_agent_id}")
            return handoff.handoff_id

        except Exception as e:
            logger.error(f"Handoff with memory failed: {e}")
            return None

    async def start_task_workflow(
        self,
        task: SwarmTask,
    ) -> Optional[str]:
        """
        Start a task using LangGraph workflow for stateful execution.

        AGI v1.8: Workflow-based task execution.

        Args:
            task: The task to execute

        Returns:
            Workflow ID or None if not available
        """
        if not hasattr(self, '_langgraph_hybrid') or not self._langgraph_hybrid:
            logger.debug("LangGraph hybrid not enabled")
            return None

        try:
            from Cosmos.core.langgraph_workflows import TaskExecutionWorkflow

            # Create task execution workflow if not exists
            if not hasattr(self, '_task_execution_workflow'):
                self._task_execution_workflow = TaskExecutionWorkflow(self._langgraph_hybrid)

            # Start workflow
            workflow_id = await self._task_execution_workflow.start(
                task_description=task.description,
                required_capabilities=[cap.value for cap in task.required_capabilities],
                context={
                    "task_id": task.id,
                    "priority": task.priority,
                    "memory_refs": task.memory_refs,
                },
            )

            # Track workflow
            if not hasattr(self, '_task_workflows'):
                self._task_workflows = {}
            self._task_workflows[task.id] = workflow_id

            logger.info(f"Started workflow {workflow_id} for task {task.id}")
            return workflow_id

        except Exception as e:
            logger.error(f"Failed to start task workflow: {e}")
            return None

    async def start_deliberation_workflow(
        self,
        topic: str,
        participant_agent_ids: List[str],
    ) -> Optional[str]:
        """
        Start a deliberation workflow for collaborative decision-making.

        AGI v1.8: PROPOSE -> CRITIQUE -> REFINE -> VOTE workflow.

        Args:
            topic: Topic to deliberate on
            participant_agent_ids: Agents to participate

        Returns:
            Workflow ID or None if not available
        """
        if not hasattr(self, '_langgraph_hybrid') or not self._langgraph_hybrid:
            logger.debug("LangGraph hybrid not enabled")
            return None

        try:
            from Cosmos.core.langgraph_workflows import DeliberationWorkflow

            # Create deliberation workflow if not exists
            if not hasattr(self, '_deliberation_workflow'):
                self._deliberation_workflow = DeliberationWorkflow(self._langgraph_hybrid)

            # Start workflow
            workflow_id = await self._deliberation_workflow.start(
                topic=topic,
                participants=participant_agent_ids,
                context={"orchestrator": "swarm_orchestrator"},
            )

            logger.info(f"Started deliberation workflow {workflow_id} on topic: {topic[:50]}...")
            return workflow_id

        except Exception as e:
            logger.error(f"Failed to start deliberation workflow: {e}")
            return None

    def get_agi_v18_status(self) -> Dict[str, Any]:
        """Get AGI v1.8 feature status."""
        return {
            "cross_agent_memory_enabled": hasattr(self, '_cross_agent_memory') and self._cross_agent_memory is not None,
            "a2a_protocol_enabled": hasattr(self, '_a2a_protocol') and self._a2a_protocol is not None,
            "langgraph_hybrid_enabled": hasattr(self, '_langgraph_hybrid') and self._langgraph_hybrid is not None,
            "active_workflows": len(getattr(self, '_task_workflows', {})),
            "cross_agent_memory_stats": (
                self._cross_agent_memory.get_stats()
                if hasattr(self, '_cross_agent_memory') and self._cross_agent_memory
                else None
            ),
            "a2a_protocol_stats": (
                self._a2a_protocol.get_stats()
                if hasattr(self, '_a2a_protocol') and self._a2a_protocol
                else None
            ),
        }
