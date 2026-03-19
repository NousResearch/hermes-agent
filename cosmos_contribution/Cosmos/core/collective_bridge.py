"""
Cosmos Collective Bridge - Collective to Swarm Communication.

AGI v1.8.4 Feature: Bridge between the deliberation collective and
individual shadow agents in the swarm.

Enables:
- Dispatching consensus decisions to appropriate agents
- Agents escalating issues to the collective
- Cross-collective coordination and memory sync

Architecture:
    ┌──────────────────────┐
    │  Deliberation Room   │
    │  (PROPOSE/CRITIQUE/  │
    │   REFINE/VOTE)       │
    └──────────┬───────────┘
               │
         COLLECTIVE_DISPATCH
               │
    ┌──────────▼───────────┐
    │   CollectiveBridge   │
    │   (routing/escala.)  │
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │   Shadow Agents      │
    │   (grok, gemini...)  │
    └──────────────────────┘
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import  Callable, Optional, Set, Awaitable
from pathlib import Path

from loguru import logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DispatchPriority(Enum):
    """Priority levels for collective dispatches."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationReason(Enum):
    """Reasons an agent might escalate to the collective."""
    UNCERTAINTY = "uncertainty"           # Agent unsure how to proceed
    CONFLICT = "conflict"                 # Conflicting information
    RESOURCE_LIMIT = "resource_limit"     # Need more resources/tokens
    ETHICAL_CONCERN = "ethical_concern"   # Potential ethical issue
    KNOWLEDGE_GAP = "knowledge_gap"       # Lacks needed knowledge
    TASK_TOO_COMPLEX = "task_too_complex" # Task beyond capabilities


@dataclass
class ConsensusResult:
    """Result from a deliberation collective."""
    deliberation_id: str
    prompt: str
    final_response: str
    winning_agent: str
    vote_breakdown: dict[str, float]
    consensus_reached: bool
    participating_agents: list[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "deliberation_id": self.deliberation_id,
            "prompt": self.prompt,
            "final_response": self.final_response,
            "winning_agent": self.winning_agent,
            "vote_breakdown": self.vote_breakdown,
            "consensus_reached": self.consensus_reached,
            "participating_agents": self.participating_agents,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DispatchRequest:
    """A request to dispatch work to an agent."""
    dispatch_id: str
    source: str  # "collective" or specific collective_id
    consensus: ConsensusResult
    target_agents: list[str]
    priority: DispatchPriority = DispatchPriority.NORMAL
    context: dict = field(default_factory=dict)
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "dispatch_id": self.dispatch_id,
            "source": self.source,
            "deliberation_id": self.consensus.deliberation_id,
            "target_agents": self.target_agents,
            "priority": self.priority.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EscalationRequest:
    """A request from an agent to escalate to the collective."""
    escalation_id: str
    agent_id: str
    reason: EscalationReason
    issue: dict
    context: dict = field(default_factory=dict)
    suggested_agents: list[str] = field(default_factory=list)
    urgency: float = 0.5  # 0-1
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "escalation_id": self.escalation_id,
            "agent_id": self.agent_id,
            "reason": self.reason.value,
            "issue": self.issue,
            "suggested_agents": self.suggested_agents,
            "urgency": self.urgency,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class VoteRequest:
    """A request from an agent for a collective vote."""
    vote_id: str
    agent_id: str
    question: str
    options: list[dict]
    default_option: Optional[int] = None
    timeout_seconds: float = 300.0
    context: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    votes: dict[str, int] = field(default_factory=dict)  # agent_id -> option_index
    result: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "vote_id": self.vote_id,
            "agent_id": self.agent_id,
            "question": self.question,
            "options": self.options,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "vote_count": len(self.votes),
            "result": self.result,
        }


@dataclass
class CollectiveState:
    """State of a collective for syncing."""
    collective_id: str
    name: str
    members: list[str]
    active_deliberations: int
    memory_hash: str  # Hash of collective memory state
    last_consensus: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# =============================================================================
# COLLECTIVE BRIDGE
# =============================================================================

class CollectiveBridge:
    """
    Bridge between deliberation collective and shadow agent swarm.

    Enables:
    - Dispatching consensus decisions to individual agents
    - Agents escalating issues to the collective
    - Cross-collective coordination
    - Memory synchronization between collectives
    """

    def __init__(self, data_dir: str = "./data/collective_bridge"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Pending dispatches
        self._dispatch_queue: asyncio.Queue[DispatchRequest] = asyncio.Queue()
        self._active_dispatches: dict[str, DispatchRequest] = {}
        self._dispatch_results: dict[str[str]] = {}

        # Escalations
        self._escalation_queue: asyncio.Queue[EscalationRequest] = asyncio.Queue()
        self._active_escalations: dict[str, EscalationRequest] = {}

        # Vote requests
        self._vote_requests: dict[str, VoteRequest] = {}

        # Collective registry
        self._collectives: dict[str, CollectiveState] = {}

        # Handlers
        self._dispatch_handlers: dict[str, Callable[[DispatchRequest], Awaitable[dict]]] = {}
        self._escalation_handlers: list[Callable[[EscalationRequest], Awaitable[None]]] = []

        # Nexus integration
        self._nexus = None

        # Deliberation room reference
        self._deliberation_room = None

        # A2A Mesh integration
        self._a2a_mesh = None

        # Worker tasks
        self._dispatch_worker: Optional[asyncio.Task] = None
        self._escalation_worker: Optional[asyncio.Task] = None

        logger.info("CollectiveBridge initialized")

    def connect_nexus(self, nexus) -> None:
        """Connect to the Nexus event bus."""
        self._nexus = nexus

    def connect_deliberation_room(self, room) -> None:
        """Connect to the deliberation room."""
        self._deliberation_room = room

    def connect_a2a_mesh(self, mesh) -> None:
        """Connect to the A2A mesh."""
        self._a2a_mesh = mesh

    # =========================================================================
    # COLLECTIVE -> SWARM: DISPATCH
    # =========================================================================

    async def dispatch_consensus(
        self,
        consensus: ConsensusResult,
        target_agents: list[str],
        priority: DispatchPriority = DispatchPriority.NORMAL,
        context: Optional[dict] = None,
        deadline_seconds: Optional[float] = None,
    ) -> str:
        """
        Dispatch a consensus decision to specific agents.

        Args:
            consensus: The consensus result from deliberation
            target_agents: Agents to receive the dispatch
            priority: Priority level
            context: Additional context for execution
            deadline_seconds: Optional deadline for completion

        Returns:
            Dispatch ID for tracking
        """
        dispatch_id = f"dispatch_{uuid.uuid4().hex[:12]}"

        deadline = None
        if deadline_seconds:
            deadline = datetime.now() + timedelta(seconds=deadline_seconds)

        dispatch = DispatchRequest(
            dispatch_id=dispatch_id,
            source="collective",
            consensus=consensus,
            target_agents=target_agents,
            priority=priority,
            context=context or {},
            deadline=deadline,
        )

        self._active_dispatches[dispatch_id] = dispatch
        await self._dispatch_queue.put(dispatch)

        # Emit signal
        await self._emit_signal("COLLECTIVE_DISPATCH", {
            "dispatch_id": dispatch_id,
            "deliberation_id": consensus.deliberation_id,
            "target_agents": target_agents,
            "priority": priority.value,
            "winning_agent": consensus.winning_agent,
        })

        logger.info(
            f"Dispatching consensus {consensus.deliberation_id} to "
            f"{len(target_agents)} agents as {dispatch_id}"
        )
        return dispatch_id

    async def assign_from_deliberation(
        self,
        task: dict,
        winning_agent: str,
        context: Optional[dict] = None,
    ) -> bool:
        """
        Assign a task directly to the winning agent from deliberation.

        Args:
            task: Task details
            winning_agent: Agent that won the deliberation
            context: Additional context

        Returns:
            Whether assignment was successful
        """
        # Send via A2A mesh if available
        if self._a2a_mesh:
            await self._a2a_mesh.send_direct(
                source="collective_bridge",
                target=winning_agent,
                message_type="task.assign",
                payload={
                    "task": task,
                    "context": context or {},
                },
            )

        # Emit signal
        await self._emit_signal("A2A_TASK_ASSIGNED", {
            "agent": winning_agent,
            "task": task.get("description", "")[:100],
        })

        logger.info(f"Assigned task to {winning_agent}")
        return True

    def register_dispatch_handler(
        self,
        agent_id: str,
        handler: Callable[[DispatchRequest], Awaitable[dict]],
    ) -> None:
        """Register a handler for an agent to receive dispatches."""
        self._dispatch_handlers[agent_id] = handler
        logger.debug(f"Registered dispatch handler for {agent_id}")

    async def get_dispatch_result(
        self,
        dispatch_id: str,
        timeout: float = 60.0,
    ) -> Optional[dict]:
        """Wait for and get the result of a dispatch."""
        start = datetime.now()
        while (datetime.now() - start).total_seconds() < timeout:
            if dispatch_id in self._dispatch_results:
                return self._dispatch_results.pop(dispatch_id)
            await asyncio.sleep(0.5)
        return None

    # =========================================================================
    # SWARM -> COLLECTIVE: ESCALATION
    # =========================================================================

    async def escalate_to_collective(
        self,
        agent_id: str,
        reason: EscalationReason,
        issue: dict,
        context: Optional[dict] = None,
        suggested_agents: Optional[list[str]] = None,
        urgency: float = 0.5,
    ) -> str:
        """
        Escalate an issue from an agent to the collective.

        Args:
            agent_id: Agent making the escalation
            reason: Why the escalation is needed
            issue: Details of the issue
            context: Additional context
            suggested_agents: Suggested agents to involve
            urgency: How urgent (0-1)

        Returns:
            Escalation ID for tracking
        """
        escalation_id = f"esc_{uuid.uuid4().hex[:12]}"

        escalation = EscalationRequest(
            escalation_id=escalation_id,
            agent_id=agent_id,
            reason=reason,
            issue=issue,
            context=context or {},
            suggested_agents=suggested_agents or [],
            urgency=urgency,
        )

        self._active_escalations[escalation_id] = escalation
        await self._escalation_queue.put(escalation)

        # Emit signal
        await self._emit_signal("COLLECTIVE_ESCALATE", {
            "escalation_id": escalation_id,
            "agent_id": agent_id,
            "reason": reason.value,
            "urgency": urgency,
        })

        logger.info(
            f"Escalation {escalation_id} from {agent_id}: "
            f"{reason.value} (urgency: {urgency})"
        )
        return escalation_id

    def register_escalation_handler(
        self,
        handler: Callable[[EscalationRequest], Awaitable[None]],
    ) -> None:
        """Register a handler for processing escalations."""
        self._escalation_handlers.append(handler)

    async def _process_escalation(self, escalation: EscalationRequest) -> None:
        """Process an escalation request."""
        # Trigger deliberation if room is connected
        if self._deliberation_room:
            try:
                # Build prompt from escalation
                prompt = self._build_escalation_prompt(escalation)

                # Determine which agents to involve
                agents = escalation.suggested_agents
                if not agents:
                    # Default to standard deliberation participants
                    agents = ["grok", "claude", "gemini", "deepseek"]

                # Trigger deliberation
                result = await self._deliberation_room.deliberate(
                    prompt=prompt,
                    agents=agents,
                )

                # Store result
                self._active_escalations[escalation.escalation_id] = escalation

                logger.info(
                    f"Escalation {escalation.escalation_id} resolved via "
                    f"deliberation: {result.deliberation_id}"
                )

            except Exception as e:
                logger.error(f"Escalation processing error: {e}")

        # Notify registered handlers
        for handler in self._escalation_handlers:
            try:
                await handler(escalation)
            except Exception as e:
                logger.error(f"Escalation handler error: {e}")

    def _build_escalation_prompt(self, escalation: EscalationRequest) -> str:
        """Build a deliberation prompt from an escalation."""
        reason_descriptions = {
            EscalationReason.UNCERTAINTY: "needs guidance on how to proceed",
            EscalationReason.CONFLICT: "has encountered conflicting information",
            EscalationReason.RESOURCE_LIMIT: "needs additional resources",
            EscalationReason.ETHICAL_CONCERN: "has identified a potential ethical concern",
            EscalationReason.KNOWLEDGE_GAP: "lacks necessary knowledge",
            EscalationReason.TASK_TOO_COMPLEX: "needs help with a complex task",
        }

        reason_text = reason_descriptions.get(
            escalation.reason,
            "needs collective input"
        )

        prompt_parts = [
            f"ESCALATION from {escalation.agent_id}:",
            f"The agent {reason_text}.",
            "",
            "Issue:",
            str(escalation.issue.get("description", escalation.issue)),
        ]

        if escalation.context:
            prompt_parts.extend([
                "",
                "Context:",
                str(escalation.context),
            ])

        prompt_parts.extend([
            "",
            "Please deliberate on the best course of action.",
        ])

        return "\n".join(prompt_parts)

    # =========================================================================
    # COLLECTIVE VOTING
    # =========================================================================

    async def request_collective_vote(
        self,
        agent_id: str,
        question: str,
        options: list[dict],
        default_option: Optional[int] = None,
        timeout_seconds: float = 300.0,
        context: Optional[dict] = None,
    ) -> str:
        """
        Request a collective vote on options.

        Args:
            agent_id: Agent requesting the vote
            question: Question to vote on
            options: list of options (each has "label" and "description")
            default_option: Default option index if no quorum
            timeout_seconds: Time limit for voting
            context: Additional context

        Returns:
            Vote ID for tracking
        """
        vote_id = f"vote_{uuid.uuid4().hex[:12]}"

        vote_request = VoteRequest(
            vote_id=vote_id,
            agent_id=agent_id,
            question=question,
            options=options,
            default_option=default_option,
            timeout_seconds=timeout_seconds,
            context=context or {},
        )

        self._vote_requests[vote_id] = vote_request

        # Emit signal
        await self._emit_signal("COLLECTIVE_VOTE_REQUEST", {
            "vote_id": vote_id,
            "agent_id": agent_id,
            "question": question,
            "option_count": len(options),
        })

        # Schedule vote closure
        asyncio.create_task(self._close_vote_after_timeout(vote_id, timeout_seconds))

        logger.info(f"Vote {vote_id} requested by {agent_id}: {question}")
        return vote_id

    async def submit_vote(
        self,
        vote_id: str,
        voter_agent_id: str,
        option_index: int,
    ) -> bool:
        """Submit a vote for a pending vote request."""
        if vote_id not in self._vote_requests:
            return False

        vote = self._vote_requests[vote_id]
        if option_index < 0 or option_index >= len(vote.options):
            return False

        vote.votes[voter_agent_id] = option_index
        logger.debug(f"Vote recorded: {voter_agent_id} -> option {option_index}")
        return True

    async def get_vote_result(self, vote_id: str) -> Optional[int]:
        """Get the result of a vote (or None if not yet decided)."""
        if vote_id not in self._vote_requests:
            return None
        return self._vote_requests[vote_id].result

    async def _close_vote_after_timeout(
        self,
        vote_id: str,
        timeout: float,
    ) -> None:
        """Close a vote after timeout."""
        await asyncio.sleep(timeout)

        if vote_id not in self._vote_requests:
            return

        vote = self._vote_requests[vote_id]
        if vote.result is not None:
            return  # Already decided

        # Count votes
        vote_counts: dict[int, int] = {}
        for option_idx in vote.votes.values():
            vote_counts[option_idx] = vote_counts.get(option_idx, 0) + 1

        # Find winner
        if vote_counts:
            winner = max(vote_counts.items(), key=lambda x: x[1])[0]
            vote.result = winner
        elif vote.default_option is not None:
            vote.result = vote.default_option
        else:
            vote.result = 0  # Default to first option

        logger.info(
            f"Vote {vote_id} closed with result: option {vote.result} "
            f"({len(vote.votes)} votes)"
        )

    # =========================================================================
    # CROSS-COLLECTIVE COORDINATION
    # =========================================================================

    def register_collective(
        self,
        collective_id: str,
        name: str,
        members: list[str],
        metadata: Optional[dict] = None,
    ) -> CollectiveState:
        """Register a collective for coordination."""
        state = CollectiveState(
            collective_id=collective_id,
            name=name,
            members=members,
            active_deliberations=0,
            memory_hash="",
            metadata=metadata or {},
        )

        self._collectives[collective_id] = state
        logger.info(f"Registered collective: {name} ({collective_id})")
        return state

    async def sync_collectives(
        self,
        collective_a: str,
        collective_b: str,
    ) -> dict:
        """
        Synchronize state between two collectives.

        Args:
            collective_a: First collective ID
            collective_b: Second collective ID

        Returns:
            Sync result with changes applied
        """
        if collective_a not in self._collectives or collective_b not in self._collectives:
            return {"error": "Collective not found"}

        state_a = self._collectives[collective_a]
        state_b = self._collectives[collective_b]

        # Merge metadata
        merged_meta = {**state_a.metadata, **state_b.metadata}

        # Update both
        state_a.metadata = merged_meta
        state_b.metadata = merged_meta

        # Emit signal
        await self._emit_signal("COLLECTIVE_SYNC", {
            "collective_a": collective_a,
            "collective_b": collective_b,
        })

        logger.info(f"Synced collectives: {collective_a} <-> {collective_b}")
        return {
            "synced": True,
            "collective_a": collective_a,
            "collective_b": collective_b,
            "merged_keys": list(merged_meta.keys()),
        }

    async def merge_collective_memory(
        self,
        sources: list[str],
        target: str,
    ) -> dict:
        """
        Merge memory from multiple collectives into a target.

        Args:
            sources: Source collective IDs
            target: Target collective ID

        Returns:
            Merge result
        """
        if target not in self._collectives:
            return {"error": "Target collective not found"}

        target_state = self._collectives[target]
        merged_count = 0

        for source_id in sources:
            if source_id not in self._collectives:
                continue

            source_state = self._collectives[source_id]

            # Merge metadata
            for key, value in source_state.metadata.items():
                if key not in target_state.metadata:
                    target_state.metadata[key] = value
                    merged_count += 1

        logger.info(
            f"Merged memory from {len(sources)} collectives into {target}: "
            f"{merged_count} items"
        )

        return {
            "merged": True,
            "sources": sources,
            "target": target,
            "items_merged": merged_count,
        }

    # =========================================================================
    # WORKER TASKS
    # =========================================================================

    async def start_workers(self) -> None:
        """Start background worker tasks."""
        if self._dispatch_worker is None:
            self._dispatch_worker = asyncio.create_task(self._dispatch_worker_loop())
        if self._escalation_worker is None:
            self._escalation_worker = asyncio.create_task(self._escalation_worker_loop())
        logger.info("CollectiveBridge workers started")

    async def stop_workers(self) -> None:
        """Stop background worker tasks."""
        for worker in [self._dispatch_worker, self._escalation_worker]:
            if worker:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

        self._dispatch_worker = None
        self._escalation_worker = None
        logger.info("CollectiveBridge workers stopped")

    async def _dispatch_worker_loop(self) -> None:
        """Worker loop for processing dispatches."""
        while True:
            try:
                dispatch = await self._dispatch_queue.get()

                # Process each target agent
                for agent_id in dispatch.target_agents:
                    if agent_id in self._dispatch_handlers:
                        try:
                            result = await self._dispatch_handlers[agent_id](dispatch)
                            self._dispatch_results[dispatch.dispatch_id] = result
                        except Exception as e:
                            logger.error(f"Dispatch handler error for {agent_id}: {e}")
                    elif self._a2a_mesh:
                        # Forward via mesh
                        await self._a2a_mesh.send_direct(
                            source="collective_bridge",
                            target=agent_id,
                            message_type="collective.dispatch",
                            payload=dispatch.to_dict(),
                        )

                self._dispatch_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatch worker error: {e}")

    async def _escalation_worker_loop(self) -> None:
        """Worker loop for processing escalations."""
        while True:
            try:
                escalation = await self._escalation_queue.get()
                await self._process_escalation(escalation)
                self._escalation_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Escalation worker error: {e}")

    # =========================================================================
    # INTERNAL HELPERS
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
                    source="collective_bridge",
                    urgency=0.6,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")

    # =========================================================================
    # STATUS AND METRICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get bridge statistics."""
        return {
            "dispatch_queue_size": self._dispatch_queue.qsize(),
            "escalation_queue_size": self._escalation_queue.qsize(),
            "active_dispatches": len(self._active_dispatches),
            "active_escalations": len(self._active_escalations),
            "pending_votes": sum(1 for v in self._vote_requests.values() if v.result is None),
            "registered_collectives": len(self._collectives),
            "dispatch_handlers": len(self._dispatch_handlers),
            "escalation_handlers": len(self._escalation_handlers),
        }

    def get_collective_list(self) -> list[dict]:
        """Get list of registered collectives."""
        return [
            {
                "collective_id": c.collective_id,
                "name": c.name,
                "members": c.members,
                "active_deliberations": c.active_deliberations,
            }
            for c in self._collectives.values()
        ]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_collective_bridge(data_dir: str = "./data/collective_bridge") -> CollectiveBridge:
    """Factory function to create a CollectiveBridge instance."""
    return CollectiveBridge(data_dir=data_dir)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_bridge_instance: Optional[CollectiveBridge] = None


def get_bridge() -> CollectiveBridge:
    """Get the global CollectiveBridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = create_collective_bridge()
    return _bridge_instance
