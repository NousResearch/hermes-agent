"""
Farnsworth A2A Mesh - Model-to-Model Full Mesh Connectivity.

AGI v1.8.4 Feature: Enables direct communication between all models/agents
without requiring a central coordinator.

Features:
- Peer discovery and registration
- Direct model-to-model messaging
- Sub-swarm formation and merging
- Knowledge sharing across the mesh
- Collaborative problem solving

Architecture:
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   Grok   │◄──►│  Claude  │◄──►│  Gemini  │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                   FULL MESH CONNECTIVITY
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Awaitable
from pathlib import Path

from loguru import logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PeerStatus(Enum):
    """Status of a peer in the mesh."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"


@dataclass
class AgentPeer:
    """Represents a peer agent in the mesh."""
    agent_id: str
    capabilities: List[str]
    status: PeerStatus = PeerStatus.IDLE
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    total_interactions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata,
            "response_time_ms": self.response_time_ms,
            "success_rate": self.success_rate,
            "total_interactions": self.total_interactions,
        }

    def is_stale(self, timeout_seconds: float = 60.0) -> bool:
        """Check if peer hasn't been seen recently."""
        age = (datetime.now() - self.last_seen).total_seconds()
        return age > timeout_seconds


@dataclass
class A2AMessage:
    """Message for model-to-model communication."""
    message_id: str
    source: str
    target: str  # Can be specific agent or "*" for broadcast
    message_type: str
    payload: Dict[str, Any]
    ttl: int = 30  # seconds
    created_at: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    requires_response: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "source": self.source,
            "target": self.target,
            "message_type": self.message_type,
            "payload": self.payload,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "correlation_id": self.correlation_id,
            "requires_response": self.requires_response,
        }

    def is_expired(self) -> bool:
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl


@dataclass
class SubSwarm:
    """A dynamically formed team of agents."""
    swarm_id: str
    purpose: str
    members: Set[str]
    leader: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"
    shared_context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "swarm_id": self.swarm_id,
            "purpose": self.purpose,
            "members": list(self.members),
            "leader": self.leader,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "member_count": len(self.members),
        }


@dataclass
class SharedInsight:
    """An insight shared across the mesh."""
    insight_id: str
    source: str
    content: str
    insight_type: str  # "connection", "pattern", "solution", "warning"
    relevance_score: float
    visibility: str = "public"  # "public", "team", "private"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_by: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "source": self.source,
            "content": self.content,
            "insight_type": self.insight_type,
            "relevance_score": self.relevance_score,
            "visibility": self.visibility,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "ack_count": len(self.acknowledged_by),
        }


# =============================================================================
# A2A MESH
# =============================================================================

class A2AMesh:
    """
    Full mesh connectivity between all models/agents.

    Enables direct model-to-model communication without a central coordinator.
    Supports peer discovery, direct messaging, sub-swarm formation, and
    knowledge sharing.
    """

    def __init__(self, data_dir: str = "./data/a2a_mesh"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Peer registry
        self._peers: Dict[str, AgentPeer] = {}

        # Message handling
        self._message_handlers: Dict[str, Callable[[A2AMessage], Awaitable[Any]]] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._message_history: List[A2AMessage] = []
        self._max_history: int = 1000

        # Sub-swarms
        self._sub_swarms: Dict[str, SubSwarm] = {}
        self._agent_swarms: Dict[str, Set[str]] = {}  # agent_id -> swarm_ids

        # Shared knowledge
        self._insights: Dict[str, SharedInsight] = {}
        self._insight_subscribers: Dict[str, Callable[[SharedInsight], Awaitable[None]]] = {}

        # Nexus integration
        self._nexus = None

        # Heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: float = 30.0

        logger.info("A2AMesh initialized")

    def connect_nexus(self, nexus) -> None:
        """Connect to the Nexus event bus."""
        self._nexus = nexus

    # =========================================================================
    # PEER DISCOVERY
    # =========================================================================

    async def register_peer(
        self,
        agent_id: str,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentPeer:
        """
        Register a peer in the mesh.

        Args:
            agent_id: Unique identifier for the agent
            capabilities: List of capabilities (e.g., ["code", "reasoning", "vision"])
            metadata: Optional additional metadata

        Returns:
            The registered AgentPeer
        """
        peer = AgentPeer(
            agent_id=agent_id,
            capabilities=capabilities,
            status=PeerStatus.ONLINE,
            metadata=metadata or {},
        )

        self._peers[agent_id] = peer

        # Initialize swarm membership tracking
        if agent_id not in self._agent_swarms:
            self._agent_swarms[agent_id] = set()

        # Emit signal
        await self._emit_signal("MESH_PEER_ANNOUNCE", {
            "agent_id": agent_id,
            "capabilities": capabilities,
            "action": "register",
        })

        # Broadcast to other peers
        await self.broadcast_to_peers(
            agent_id,
            A2AMessage(
                message_id=f"announce_{uuid.uuid4().hex[:8]}",
                source=agent_id,
                target="*",
                message_type="peer.announce",
                payload={"peer": peer.to_dict()},
            )
        )

        logger.info(f"Registered peer {agent_id} with capabilities: {capabilities}")
        return peer

    async def unregister_peer(self, agent_id: str) -> bool:
        """Remove a peer from the mesh."""
        if agent_id not in self._peers:
            return False

        # Leave all sub-swarms
        swarm_ids = list(self._agent_swarms.get(agent_id, set()))
        for swarm_id in swarm_ids:
            await self.leave_sub_swarm(agent_id, swarm_id)

        del self._peers[agent_id]
        self._agent_swarms.pop(agent_id, None)

        # Emit signal
        await self._emit_signal("MESH_PEER_ANNOUNCE", {
            "agent_id": agent_id,
            "action": "unregister",
        })

        logger.info(f"Unregistered peer {agent_id}")
        return True

    async def discover_peers(
        self,
        capabilities: Optional[List[str]] = None,
        status: Optional[PeerStatus] = None,
    ) -> List[AgentPeer]:
        """
        Discover peers in the mesh with optional filtering.

        Args:
            capabilities: Filter by required capabilities
            status: Filter by peer status

        Returns:
            List of matching peers
        """
        peers = list(self._peers.values())

        # Filter by capabilities
        if capabilities:
            peers = [
                p for p in peers
                if any(cap in p.capabilities for cap in capabilities)
            ]

        # Filter by status
        if status:
            peers = [p for p in peers if p.status == status]

        # Emit signal
        await self._emit_signal("MESH_PEER_DISCOVER", {
            "filter_capabilities": capabilities,
            "filter_status": status.value if status else None,
            "found_count": len(peers),
        })

        return peers

    def get_peer(self, agent_id: str) -> Optional[AgentPeer]:
        """Get a specific peer by ID."""
        return self._peers.get(agent_id)

    async def update_peer_status(self, agent_id: str, status: PeerStatus) -> bool:
        """Update a peer's status."""
        if agent_id not in self._peers:
            return False

        self._peers[agent_id].status = status
        self._peers[agent_id].last_seen = datetime.now()
        return True

    # =========================================================================
    # DIRECT MESSAGING
    # =========================================================================

    async def send_direct(
        self,
        source: str,
        target: str,
        message_type: str,
        payload: Dict[str, Any],
        requires_response: bool = False,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a direct message to another agent.

        Args:
            source: Source agent ID
            target: Target agent ID
            message_type: Type of message (e.g., "m2m.query", "m2m.insight")
            payload: Message payload
            requires_response: Whether to wait for a response
            timeout: Timeout for response (if required)

        Returns:
            Response payload if requires_response, else None
        """
        message = A2AMessage(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            source=source,
            target=target,
            message_type=message_type,
            payload=payload,
            requires_response=requires_response,
        )

        # Store in history
        self._store_message(message)

        # Update source peer's last_seen
        if source in self._peers:
            self._peers[source].last_seen = datetime.now()

        # Check if target exists
        if target not in self._peers:
            logger.warning(f"Target peer {target} not found in mesh")
            return None

        # If response required, create future
        if requires_response:
            future: asyncio.Future = asyncio.Future()
            self._pending_responses[message.message_id] = future

        # Route message to handler
        if message_type in self._message_handlers:
            try:
                result = await self._message_handlers[message_type](message)

                # If response required, complete the future
                if requires_response:
                    future.set_result(result)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                if requires_response:
                    future.set_exception(e)

        # Emit signal
        await self._emit_signal("M2M_QUERY" if message_type.startswith("m2m") else "A2A_TASK_AUCTIONED", {
            "message_id": message.message_id,
            "source": source,
            "target": target,
            "message_type": message_type,
        })

        # Wait for response if required
        if requires_response:
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Response timeout for message {message.message_id}")
                return None
            finally:
                self._pending_responses.pop(message.message_id, None)

        return None

    async def broadcast_to_peers(
        self,
        source: str,
        message: A2AMessage,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """
        Broadcast a message to all peers.

        Args:
            source: Source agent ID
            message: Message to broadcast
            exclude: Set of agent IDs to exclude

        Returns:
            Number of peers messaged
        """
        exclude = exclude or set()
        exclude.add(source)  # Don't send to self

        count = 0
        for agent_id, peer in self._peers.items():
            if agent_id in exclude:
                continue
            if peer.is_stale():
                continue

            # Clone message with new target
            peer_message = A2AMessage(
                message_id=f"{message.message_id}_{agent_id[:4]}",
                source=source,
                target=agent_id,
                message_type=message.message_type,
                payload=message.payload,
                ttl=message.ttl,
                correlation_id=message.message_id,
            )

            # Route to handler
            if message.message_type in self._message_handlers:
                try:
                    await self._message_handlers[message.message_type](peer_message)
                    count += 1
                except Exception as e:
                    logger.error(f"Broadcast handler error for {agent_id}: {e}")

        return count

    def register_message_handler(
        self,
        message_type: str,
        handler: Callable[[A2AMessage], Awaitable[Any]],
    ) -> None:
        """Register a handler for a message type."""
        self._message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")

    # =========================================================================
    # COLLABORATIVE SESSIONS (SUB-SWARMS)
    # =========================================================================

    async def form_sub_swarm(
        self,
        agents: List[str],
        purpose: str,
        leader: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SubSwarm:
        """
        Form a sub-swarm from a group of agents.

        Args:
            agents: List of agent IDs to include
            purpose: Description of the swarm's purpose
            leader: Optional leader agent ID
            context: Shared context for the swarm

        Returns:
            The formed SubSwarm
        """
        swarm_id = f"swarm_{uuid.uuid4().hex[:12]}"

        # Validate agents exist
        valid_agents = [a for a in agents if a in self._peers]
        if not valid_agents:
            raise ValueError("No valid agents for sub-swarm")

        # Select leader if not specified
        if not leader:
            leader = valid_agents[0]

        swarm = SubSwarm(
            swarm_id=swarm_id,
            purpose=purpose,
            members=set(valid_agents),
            leader=leader,
            shared_context=context or {},
        )

        self._sub_swarms[swarm_id] = swarm

        # Track membership
        for agent_id in valid_agents:
            self._agent_swarms.setdefault(agent_id, set()).add(swarm_id)

        # Emit signal
        await self._emit_signal("SUBSWARM_FORM", {
            "swarm_id": swarm_id,
            "members": valid_agents,
            "purpose": purpose,
            "leader": leader,
        })

        # Notify members
        for agent_id in valid_agents:
            await self.send_direct(
                source="mesh",
                target=agent_id,
                message_type="subswarm.form",
                payload={
                    "swarm_id": swarm_id,
                    "purpose": purpose,
                    "members": valid_agents,
                    "leader": leader,
                    "context": context,
                },
            )

        logger.info(f"Formed sub-swarm {swarm_id} with {len(valid_agents)} members")
        return swarm

    async def join_sub_swarm(self, agent_id: str, swarm_id: str) -> bool:
        """Have an agent join an existing sub-swarm."""
        if swarm_id not in self._sub_swarms:
            return False
        if agent_id not in self._peers:
            return False

        swarm = self._sub_swarms[swarm_id]
        swarm.members.add(agent_id)
        self._agent_swarms.setdefault(agent_id, set()).add(swarm_id)

        # Emit signal
        await self._emit_signal("SUBSWARM_JOIN", {
            "swarm_id": swarm_id,
            "agent_id": agent_id,
            "member_count": len(swarm.members),
        })

        logger.info(f"Agent {agent_id} joined sub-swarm {swarm_id}")
        return True

    async def leave_sub_swarm(self, agent_id: str, swarm_id: str) -> bool:
        """Have an agent leave a sub-swarm."""
        if swarm_id not in self._sub_swarms:
            return False

        swarm = self._sub_swarms[swarm_id]
        swarm.members.discard(agent_id)

        if agent_id in self._agent_swarms:
            self._agent_swarms[agent_id].discard(swarm_id)

        # If swarm is empty, remove it
        if not swarm.members:
            swarm.status = "disbanded"
            del self._sub_swarms[swarm_id]
        elif swarm.leader == agent_id:
            # Elect new leader
            swarm.leader = next(iter(swarm.members), None)

        # Emit signal
        await self._emit_signal("SUBSWARM_LEAVE", {
            "swarm_id": swarm_id,
            "agent_id": agent_id,
            "disbanded": swarm.status == "disbanded",
        })

        return True

    async def merge_sub_swarms(
        self,
        swarm_a_id: str,
        swarm_b_id: str,
        new_purpose: Optional[str] = None,
    ) -> Optional[SubSwarm]:
        """
        Merge two sub-swarms into one.

        Args:
            swarm_a_id: First swarm ID
            swarm_b_id: Second swarm ID
            new_purpose: Optional new purpose (defaults to combined)

        Returns:
            The merged SubSwarm or None if merge failed
        """
        if swarm_a_id not in self._sub_swarms or swarm_b_id not in self._sub_swarms:
            return None

        swarm_a = self._sub_swarms[swarm_a_id]
        swarm_b = self._sub_swarms[swarm_b_id]

        # Combine members
        combined_members = swarm_a.members | swarm_b.members

        # Combine context
        combined_context = {
            **swarm_a.shared_context,
            **swarm_b.shared_context,
        }

        # Create new swarm
        merged_swarm = await self.form_sub_swarm(
            agents=list(combined_members),
            purpose=new_purpose or f"{swarm_a.purpose} + {swarm_b.purpose}",
            leader=swarm_a.leader,
            context=combined_context,
        )

        # Remove old swarms
        for swarm_id in [swarm_a_id, swarm_b_id]:
            swarm = self._sub_swarms[swarm_id]
            for agent_id in list(swarm.members):
                await self.leave_sub_swarm(agent_id, swarm_id)

        # Emit signal
        await self._emit_signal("SUBSWARM_MERGE", {
            "source_swarms": [swarm_a_id, swarm_b_id],
            "merged_swarm_id": merged_swarm.swarm_id,
            "member_count": len(merged_swarm.members),
        })

        return merged_swarm

    def get_sub_swarm(self, swarm_id: str) -> Optional[SubSwarm]:
        """Get a sub-swarm by ID."""
        return self._sub_swarms.get(swarm_id)

    def get_agent_swarms(self, agent_id: str) -> List[SubSwarm]:
        """Get all sub-swarms an agent is part of."""
        swarm_ids = self._agent_swarms.get(agent_id, set())
        return [self._sub_swarms[sid] for sid in swarm_ids if sid in self._sub_swarms]

    # =========================================================================
    # KNOWLEDGE SHARING
    # =========================================================================

    async def share_insight(
        self,
        source: str,
        content: str,
        insight_type: str = "general",
        visibility: str = "public",
        tags: Optional[List[str]] = None,
        relevance_score: float = 0.5,
    ) -> SharedInsight:
        """
        Share an insight with the mesh.

        Args:
            source: Agent sharing the insight
            content: The insight content
            insight_type: Type of insight ("connection", "pattern", "solution", "warning")
            visibility: Who can see it ("public", "team", "private")
            tags: Optional tags for categorization
            relevance_score: How relevant/important the insight is (0-1)

        Returns:
            The shared insight
        """
        insight_id = f"insight_{uuid.uuid4().hex[:12]}"

        insight = SharedInsight(
            insight_id=insight_id,
            source=source,
            content=content,
            insight_type=insight_type,
            relevance_score=relevance_score,
            visibility=visibility,
            tags=tags or [],
        )

        self._insights[insight_id] = insight

        # Emit signal
        await self._emit_signal("M2M_INSIGHT", {
            "insight_id": insight_id,
            "source": source,
            "type": insight_type,
            "visibility": visibility,
            "relevance": relevance_score,
        })

        # Notify subscribers
        for subscriber_id, callback in self._insight_subscribers.items():
            # Respect visibility
            if visibility == "public" or subscriber_id == source:
                try:
                    await callback(insight)
                except Exception as e:
                    logger.error(f"Insight subscriber error: {e}")

        logger.debug(f"Shared insight {insight_id} from {source}")
        return insight

    async def request_knowledge(
        self,
        source: str,
        query: str,
        peers: Optional[List[str]] = None,
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Request knowledge from peers.

        Args:
            source: Requesting agent ID
            query: The knowledge query
            peers: Optional specific peers to query (all if None)
            timeout: Response timeout

        Returns:
            List of responses from peers
        """
        if peers is None:
            peers = [p.agent_id for p in self._peers.values() if p.agent_id != source]

        responses = []

        async def query_peer(peer_id: str) -> Optional[Dict[str, Any]]:
            try:
                response = await self.send_direct(
                    source=source,
                    target=peer_id,
                    message_type="m2m.query",
                    payload={"query": query},
                    requires_response=True,
                    timeout=timeout,
                )
                if response:
                    return {"peer": peer_id, "response": response}
            except Exception as e:
                logger.debug(f"Query to {peer_id} failed: {e}")
            return None

        # Query all peers in parallel
        tasks = [query_peer(peer_id) for peer_id in peers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict):
                responses.append(result)

        return responses

    def subscribe_to_insights(
        self,
        subscriber_id: str,
        callback: Callable[[SharedInsight], Awaitable[None]],
    ) -> None:
        """Subscribe to receive new insights."""
        self._insight_subscribers[subscriber_id] = callback

    def unsubscribe_from_insights(self, subscriber_id: str) -> bool:
        """Unsubscribe from insights."""
        return self._insight_subscribers.pop(subscriber_id, None) is not None

    def get_insights(
        self,
        insight_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_relevance: float = 0.0,
        limit: int = 100,
    ) -> List[SharedInsight]:
        """
        Get insights with optional filtering.

        Args:
            insight_type: Filter by type
            tags: Filter by tags (any match)
            min_relevance: Minimum relevance score
            limit: Maximum results

        Returns:
            List of matching insights
        """
        insights = list(self._insights.values())

        # Filter by type
        if insight_type:
            insights = [i for i in insights if i.insight_type == insight_type]

        # Filter by tags
        if tags:
            insights = [i for i in insights if any(t in i.tags for t in tags)]

        # Filter by relevance
        insights = [i for i in insights if i.relevance_score >= min_relevance]

        # Sort by relevance and recency
        insights.sort(key=lambda x: (x.relevance_score, x.created_at), reverse=True)

        return insights[:limit]

    # =========================================================================
    # HEARTBEAT AND HEALTH
    # =========================================================================

    async def start_heartbeat(self) -> None:
        """Start the heartbeat loop for peer health monitoring."""
        if self._heartbeat_task is not None:
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Mesh heartbeat started")

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat loop."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Background loop to check peer health."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                stale_peers = []
                for agent_id, peer in self._peers.items():
                    if peer.is_stale(timeout_seconds=self._heartbeat_interval * 3):
                        stale_peers.append(agent_id)
                        peer.status = PeerStatus.OFFLINE

                if stale_peers:
                    # Emit signal for stale peers
                    await self._emit_signal("MESH_PEER_HEARTBEAT", {
                        "stale_peers": stale_peers,
                        "active_peers": len(self._peers) - len(stale_peers),
                    })
                    logger.debug(f"Detected {len(stale_peers)} stale peers")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _store_message(self, message: A2AMessage) -> None:
        """Store message in history."""
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history:]

    async def _emit_signal(self, signal_type: str, payload: Dict[str, Any]) -> None:
        """Emit a signal to Nexus."""
        if not self._nexus:
            return

        try:
            from farnsworth.core.nexus import SignalType

            signal_enum = getattr(SignalType, signal_type, None)
            if signal_enum:
                await self._nexus.emit(
                    type=signal_enum,
                    payload=payload,
                    source="a2a_mesh",
                    urgency=0.5,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")

    # =========================================================================
    # STATUS AND METRICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get mesh statistics."""
        online_peers = sum(1 for p in self._peers.values() if p.status == PeerStatus.ONLINE)

        return {
            "total_peers": len(self._peers),
            "online_peers": online_peers,
            "sub_swarms": len(self._sub_swarms),
            "active_swarm_members": sum(len(s.members) for s in self._sub_swarms.values()),
            "total_insights": len(self._insights),
            "message_handlers": len(self._message_handlers),
            "message_history_size": len(self._message_history),
            "insight_subscribers": len(self._insight_subscribers),
        }

    def get_peer_list(self) -> List[Dict[str, Any]]:
        """Get list of all peers."""
        return [peer.to_dict() for peer in self._peers.values()]

    def get_swarm_list(self) -> List[Dict[str, Any]]:
        """Get list of all sub-swarms."""
        return [swarm.to_dict() for swarm in self._sub_swarms.values()]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_a2a_mesh(data_dir: str = "./data/a2a_mesh") -> A2AMesh:
    """Factory function to create an A2AMesh instance."""
    return A2AMesh(data_dir=data_dir)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_mesh_instance: Optional[A2AMesh] = None


def get_mesh() -> A2AMesh:
    """Get the global A2AMesh instance."""
    global _mesh_instance
    if _mesh_instance is None:
        _mesh_instance = create_a2a_mesh()
    return _mesh_instance
