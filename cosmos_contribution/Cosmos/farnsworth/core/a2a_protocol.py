"""
Farnsworth A2A Protocol - Agent-to-Agent Communication.

AGI v1.8 Feature: Direct agent-to-agent collaboration protocol
for peer sessions, task auctions, context sharing, and skill transfer.

Features:
- A2AMessageType: Standard message types for agent communication
- A2AMessage: Message structure with TTL and acknowledgment
- A2ASession: Collaboration session between agents
- TaskAuction: Distributed task allocation with bidding
- A2AProtocol: Main protocol coordinator
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Awaitable

from loguru import logger


# =============================================================================
# A2A MESSAGE TYPES
# =============================================================================

class A2AMessageType(Enum):
    """Standard message types for agent-to-agent communication."""
    # Session management
    SESSION_REQUEST = "session.request"
    SESSION_ACCEPT = "session.accept"
    SESSION_REJECT = "session.reject"
    SESSION_END = "session.end"
    SESSION_HEARTBEAT = "session.heartbeat"

    # Task delegation
    TASK_OFFER = "task.offer"
    TASK_ACCEPT = "task.accept"
    TASK_REJECT = "task.reject"
    TASK_COMPLETE = "task.complete"
    TASK_FAILED = "task.failed"
    TASK_PROGRESS = "task.progress"

    # Auction messages
    AUCTION_ANNOUNCE = "auction.announce"
    AUCTION_BID = "auction.bid"
    AUCTION_AWARD = "auction.award"
    AUCTION_CANCEL = "auction.cancel"

    # Context sharing
    CONTEXT_SHARE = "context.share"
    CONTEXT_REQUEST = "context.request"
    CONTEXT_RESPONSE = "context.response"

    # Skill transfer
    SKILL_OFFER = "skill.offer"
    SKILL_REQUEST = "skill.request"
    SKILL_TRANSFER = "skill.transfer"
    SKILL_ACK = "skill.ack"

    # General
    PING = "ping"
    PONG = "pong"
    ERROR = "error"

    # =========================================================================
    # AGI v1.8.4: Mesh Connectivity
    # =========================================================================

    # Peer discovery and health
    PEER_ANNOUNCE = "peer.announce"                  # Agent announcing itself to mesh
    PEER_DISCOVER = "peer.discover"                  # Request for peer discovery
    PEER_HEARTBEAT = "peer.heartbeat"                # Liveness check between peers

    # =========================================================================
    # AGI v1.8.4: Model-to-Model Communication
    # =========================================================================

    # Direct model-to-model messaging
    M2M_INSIGHT = "m2m.insight"                      # Sharing an insight with peers
    M2M_QUERY = "m2m.query"                          # Query to another model
    M2M_RESPONSE = "m2m.response"                    # Response to M2M query
    M2M_COLLABORATE = "m2m.collaborate"              # Request for collaboration

    # =========================================================================
    # AGI v1.8.4: Collective Bridge
    # =========================================================================

    # Collective ↔ Swarm bridging
    COLLECTIVE_DISPATCH = "collective.dispatch"      # Dispatch consensus to agents
    COLLECTIVE_ESCALATE = "collective.escalate"      # Agent escalating issue to collective
    COLLECTIVE_VOTE_REQUEST = "collective.vote_request"  # Request collective vote
    COLLECTIVE_SYNC = "collective.sync"              # Sync between collectives

    # =========================================================================
    # AGI v1.8.4: Sub-Swarm Management
    # =========================================================================

    # Dynamic team formation
    SUBSWARM_FORM = "subswarm.form"                  # Form new sub-swarm
    SUBSWARM_JOIN = "subswarm.join"                  # Join existing sub-swarm
    SUBSWARM_LEAVE = "subswarm.leave"                # Leave sub-swarm
    SUBSWARM_MERGE = "subswarm.merge"                # Merge two sub-swarms


class SessionState(Enum):
    """State of an A2A session."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BidStatus(Enum):
    """Status of a task auction bid."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


# =============================================================================
# A2A DATA STRUCTURES
# =============================================================================

@dataclass
class A2AMessage:
    """
    Message structure for agent-to-agent communication.

    Includes TTL, acknowledgment tracking, and correlation IDs.
    """
    id: str
    type: A2AMessageType
    source_agent: str
    target_agent: str
    payload: Dict[str, Any]
    ttl: int = 30  # Time-to-live in seconds
    requires_ack: bool = False
    correlation_id: Optional[str] = None  # For request/response pairing
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Acknowledgment tracking
    acknowledged: bool = False
    ack_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "payload": self.payload,
            "ttl": self.ttl,
            "requires_ack": self.requires_ack,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        return cls(
            id=data["id"],
            type=A2AMessageType(data["type"]),
            source_agent=data["source_agent"],
            target_agent=data["target_agent"],
            payload=data["payload"],
            ttl=data.get("ttl", 30),
            requires_ack=data.get("requires_ack", False),
            correlation_id=data.get("correlation_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
            acknowledged=data.get("acknowledged", False),
        )

    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl


@dataclass
class A2ASession:
    """
    Collaboration session between agents.

    Maintains state, message history, and shared context.
    """
    session_id: str
    initiator: str
    participants: Set[str]
    purpose: str
    state: SessionState = SessionState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Session data
    shared_context: Dict[str, Any] = field(default_factory=dict)
    message_history: List[A2AMessage] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    timeout_seconds: float = 600.0  # 10 minute default
    max_messages: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "initiator": self.initiator,
            "participants": list(self.participants),
            "purpose": self.purpose,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "shared_context": self.shared_context,
            "message_count": len(self.message_history),
            "results": self.results,
            "metadata": self.metadata,
        }

    def add_message(self, message: A2AMessage) -> None:
        """Add a message to session history."""
        self.message_history.append(message)
        self.updated_at = datetime.now()

        # Trim if exceeds max
        if len(self.message_history) > self.max_messages:
            self.message_history = self.message_history[-self.max_messages:]

    def is_expired(self) -> bool:
        """Check if session has timed out."""
        age = (datetime.now() - self.updated_at).total_seconds()
        return age > self.timeout_seconds and self.state == SessionState.ACTIVE


@dataclass
class TaskBid:
    """Bid submitted for a task auction."""
    bid_id: str
    auction_id: str
    agent_id: str
    confidence: float  # 0-1 confidence in completing task
    estimated_tokens: int = 0
    capabilities_offered: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    status: BidStatus = BidStatus.PENDING
    submitted_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid_id": self.bid_id,
            "auction_id": self.auction_id,
            "agent_id": self.agent_id,
            "confidence": self.confidence,
            "estimated_tokens": self.estimated_tokens,
            "capabilities_offered": self.capabilities_offered,
            "constraints": self.constraints,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TaskAuction:
    """
    Distributed task allocation through bidding.

    Agents bid on tasks based on their capabilities and availability.
    """
    auction_id: str
    task_description: str
    required_capabilities: List[str]
    initiator: str
    deadline: datetime
    min_confidence: float = 0.5

    # State
    bids: Dict[str, TaskBid] = field(default_factory=dict)  # agent_id -> bid
    winner: Optional[str] = None
    is_closed: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    # Metadata
    priority: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auction_id": self.auction_id,
            "task_description": self.task_description,
            "required_capabilities": self.required_capabilities,
            "initiator": self.initiator,
            "deadline": self.deadline.isoformat(),
            "min_confidence": self.min_confidence,
            "bids": {k: v.to_dict() for k, v in self.bids.items()},
            "winner": self.winner,
            "is_closed": self.is_closed,
            "created_at": self.created_at.isoformat(),
            "priority": self.priority,
        }

    def submit_bid(self, bid: TaskBid) -> bool:
        """Submit a bid to the auction."""
        if self.is_closed:
            return False
        if datetime.now() > self.deadline:
            return False
        if bid.confidence < self.min_confidence:
            return False

        self.bids[bid.agent_id] = bid
        return True

    def select_winner(self) -> Optional[str]:
        """Select the winning bid based on confidence and capabilities."""
        if self.is_closed or not self.bids:
            return None

        valid_bids = [
            bid for bid in self.bids.values()
            if bid.status == BidStatus.PENDING
        ]

        if not valid_bids:
            return None

        # Score bids
        scored_bids = []
        for bid in valid_bids:
            # Capability match score
            cap_match = len(
                set(bid.capabilities_offered) & set(self.required_capabilities)
            ) / max(len(self.required_capabilities), 1)

            # Combined score
            score = bid.confidence * 0.6 + cap_match * 0.4
            scored_bids.append((score, bid))

        # Sort by score descending
        scored_bids.sort(key=lambda x: x[0], reverse=True)

        # Award to highest scorer
        winner_bid = scored_bids[0][1]
        winner_bid.status = BidStatus.ACCEPTED
        self.winner = winner_bid.agent_id
        self.is_closed = True

        # Reject other bids
        for bid in self.bids.values():
            if bid.agent_id != self.winner:
                bid.status = BidStatus.REJECTED

        return self.winner


# =============================================================================
# A2A PROTOCOL
# =============================================================================

class A2AProtocol:
    """
    Main coordinator for agent-to-agent communication.

    Manages sessions, message routing, auctions, and context sharing.
    """

    def __init__(self, data_dir: str = "./data/a2a"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Sessions
        self._sessions: Dict[str, A2ASession] = {}
        self._agent_sessions: Dict[str, Set[str]] = {}  # agent_id -> session_ids

        # Auctions
        self._auctions: Dict[str, TaskAuction] = {}

        # Message handlers
        self._message_handlers: Dict[A2AMessageType, Callable] = {}

        # Pending messages (for ack tracking)
        self._pending_acks: Dict[str, A2AMessage] = {}

        # Skills registry (agent_id -> skill_ids)
        self._agent_skills: Dict[str, Set[str]] = {}

        # Nexus integration
        self._nexus = None

        # P2P fabric integration
        self._p2p_fabric = None

        logger.info("A2AProtocol initialized")

    def connect_nexus(self, nexus) -> None:
        """Connect to the Nexus event bus."""
        self._nexus = nexus

    def connect_p2p(self, p2p_fabric) -> None:
        """Connect to the P2P swarm fabric."""
        self._p2p_fabric = p2p_fabric

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    async def request_session(
        self,
        initiator: str,
        target_agents: List[str],
        purpose: str,
        timeout_seconds: float = 600.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Request a new collaboration session with agents."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        session = A2ASession(
            session_id=session_id,
            initiator=initiator,
            participants={initiator},
            purpose=purpose,
            timeout_seconds=timeout_seconds,
            shared_context=context or {},
        )

        self._sessions[session_id] = session

        # Track in agent sessions
        if initiator not in self._agent_sessions:
            self._agent_sessions[initiator] = set()
        self._agent_sessions[initiator].add(session_id)

        # Send session requests to targets
        for target in target_agents:
            await self.send_message(
                source=initiator,
                target=target,
                message_type=A2AMessageType.SESSION_REQUEST,
                payload={
                    "session_id": session_id,
                    "purpose": purpose,
                    "context": context,
                },
                requires_ack=True,
            )

        # Emit signal
        await self._emit_signal("A2A_SESSION_REQUESTED", {
            "session_id": session_id,
            "initiator": initiator,
            "targets": target_agents,
            "purpose": purpose,
        })

        logger.info(f"Session {session_id} requested by {initiator}")
        return session_id

    async def accept_session(
        self,
        session_id: str,
        agent_id: str,
    ) -> bool:
        """Accept an invitation to join a session."""
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        session.participants.add(agent_id)
        session.state = SessionState.ACTIVE
        session.updated_at = datetime.now()

        # Track in agent sessions
        if agent_id not in self._agent_sessions:
            self._agent_sessions[agent_id] = set()
        self._agent_sessions[agent_id].add(session_id)

        # Notify initiator
        await self.send_message(
            source=agent_id,
            target=session.initiator,
            message_type=A2AMessageType.SESSION_ACCEPT,
            payload={"session_id": session_id},
        )

        # Emit signal
        await self._emit_signal("A2A_SESSION_STARTED", {
            "session_id": session_id,
            "participant": agent_id,
            "participants": list(session.participants),
        })

        logger.info(f"Agent {agent_id} joined session {session_id}")
        return True

    async def end_session(
        self,
        session_id: str,
        agent_id: str,
        results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """End a collaboration session."""
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        if agent_id not in session.participants:
            return False

        session.state = SessionState.COMPLETED
        session.results = results or {}
        session.updated_at = datetime.now()

        # Notify all participants
        for participant in session.participants:
            if participant != agent_id:
                await self.send_message(
                    source=agent_id,
                    target=participant,
                    message_type=A2AMessageType.SESSION_END,
                    payload={
                        "session_id": session_id,
                        "results": results,
                    },
                )

        # Emit signal
        await self._emit_signal("A2A_SESSION_ENDED", {
            "session_id": session_id,
            "ended_by": agent_id,
            "results": results,
        })

        logger.info(f"Session {session_id} ended by {agent_id}")
        return True

    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================

    async def send_message(
        self,
        source: str,
        target: str,
        message_type: A2AMessageType,
        payload: Dict[str, Any],
        requires_ack: bool = False,
        ttl: int = 30,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Send a message to another agent."""
        message_id = f"msg_{uuid.uuid4().hex[:12]}"

        message = A2AMessage(
            id=message_id,
            type=message_type,
            source_agent=source,
            target_agent=target,
            payload=payload,
            ttl=ttl,
            requires_ack=requires_ack,
            correlation_id=correlation_id,
        )

        # Track for ack
        if requires_ack:
            self._pending_acks[message_id] = message

        # Route message
        await self._route_message(message)

        return message_id

    async def _route_message(self, message: A2AMessage) -> None:
        """Route a message to its destination."""
        # Check for registered handler
        if message.type in self._message_handlers:
            await self._message_handlers[message.type](message)
            return

        # Try P2P routing if available
        if self._p2p_fabric:
            try:
                await self._p2p_fabric.send_a2a_message(message.to_dict())
            except Exception as e:
                logger.debug(f"P2P routing failed: {e}")

        # Add to session history if applicable
        for session in self._sessions.values():
            if (
                message.source_agent in session.participants and
                message.target_agent in session.participants
            ):
                session.add_message(message)
                break

    def register_handler(
        self,
        message_type: A2AMessageType,
        handler: Callable[[A2AMessage], Awaitable[None]],
    ) -> None:
        """Register a handler for a message type."""
        self._message_handlers[message_type] = handler

    async def acknowledge_message(self, message_id: str) -> bool:
        """Acknowledge receipt of a message."""
        if message_id not in self._pending_acks:
            return False

        message = self._pending_acks.pop(message_id)
        message.acknowledged = True
        message.ack_time = datetime.now()
        return True

    # =========================================================================
    # TASK AUCTIONS
    # =========================================================================

    async def broadcast_task_auction(
        self,
        initiator: str,
        task_description: str,
        required_capabilities: List[str],
        deadline_seconds: float = 30.0,
        min_confidence: float = 0.5,
        priority: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Broadcast a task auction to all agents."""
        auction_id = f"auc_{uuid.uuid4().hex[:12]}"

        auction = TaskAuction(
            auction_id=auction_id,
            task_description=task_description,
            required_capabilities=required_capabilities,
            initiator=initiator,
            deadline=datetime.now() + timedelta(seconds=deadline_seconds),
            min_confidence=min_confidence,
            priority=priority,
            context=context or {},
        )

        self._auctions[auction_id] = auction

        # Emit signal
        await self._emit_signal("A2A_TASK_AUCTIONED", {
            "auction_id": auction_id,
            "initiator": initiator,
            "task": task_description,
            "required_capabilities": required_capabilities,
            "deadline_seconds": deadline_seconds,
        })

        # Broadcast via P2P if available
        if self._p2p_fabric:
            await self._p2p_fabric.broadcast_message({
                "type": "A2A_TASK_AUCTION",
                "auction": auction.to_dict(),
            })

        logger.info(f"Task auction {auction_id} broadcast by {initiator}")
        return auction_id

    async def submit_bid(
        self,
        auction_id: str,
        agent_id: str,
        confidence: float,
        capabilities_offered: Optional[List[str]] = None,
        estimated_tokens: int = 0,
        constraints: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Submit a bid for a task auction."""
        if auction_id not in self._auctions:
            return None

        auction = self._auctions[auction_id]
        if auction.is_closed:
            return None

        bid_id = f"bid_{uuid.uuid4().hex[:12]}"

        bid = TaskBid(
            bid_id=bid_id,
            auction_id=auction_id,
            agent_id=agent_id,
            confidence=confidence,
            capabilities_offered=capabilities_offered or [],
            estimated_tokens=estimated_tokens,
            constraints=constraints or [],
        )

        if not auction.submit_bid(bid):
            return None

        # Emit signal
        await self._emit_signal("A2A_BID_RECEIVED", {
            "auction_id": auction_id,
            "bid_id": bid_id,
            "agent_id": agent_id,
            "confidence": confidence,
        })

        logger.debug(f"Bid {bid_id} submitted for auction {auction_id}")
        return bid_id

    async def close_auction(self, auction_id: str) -> Optional[str]:
        """Close an auction and select a winner."""
        if auction_id not in self._auctions:
            return None

        auction = self._auctions[auction_id]
        winner = auction.select_winner()

        if winner:
            # Emit signal
            await self._emit_signal("A2A_TASK_ASSIGNED", {
                "auction_id": auction_id,
                "winner": winner,
                "task": auction.task_description,
            })

            # Notify winner
            await self.send_message(
                source=auction.initiator,
                target=winner,
                message_type=A2AMessageType.AUCTION_AWARD,
                payload={
                    "auction_id": auction_id,
                    "task": auction.task_description,
                    "context": auction.context,
                },
            )

        logger.info(f"Auction {auction_id} closed, winner: {winner}")
        return winner

    # =========================================================================
    # CONTEXT SHARING
    # =========================================================================

    async def share_context(
        self,
        source: str,
        target: str,
        context: Dict[str, Any],
        context_type: str = "general",
    ) -> str:
        """Share context with another agent."""
        message_id = await self.send_message(
            source=source,
            target=target,
            message_type=A2AMessageType.CONTEXT_SHARE,
            payload={
                "context_type": context_type,
                "context": context,
            },
        )

        # Emit signal
        await self._emit_signal("A2A_CONTEXT_SHARED", {
            "source": source,
            "target": target,
            "context_type": context_type,
        })

        return message_id

    # =========================================================================
    # SKILL TRANSFER
    # =========================================================================

    async def transfer_skill(
        self,
        source: str,
        target: str,
        skill_id: str,
        skill_data: Dict[str, Any],
    ) -> str:
        """Transfer a skill definition to another agent."""
        message_id = await self.send_message(
            source=source,
            target=target,
            message_type=A2AMessageType.SKILL_TRANSFER,
            payload={
                "skill_id": skill_id,
                "skill_data": skill_data,
            },
            requires_ack=True,
        )

        # Emit signal
        await self._emit_signal("A2A_SKILL_TRANSFERRED", {
            "source": source,
            "target": target,
            "skill_id": skill_id,
        })

        return message_id

    def register_skill(self, agent_id: str, skill_id: str) -> None:
        """Register that an agent has a skill."""
        if agent_id not in self._agent_skills:
            self._agent_skills[agent_id] = set()
        self._agent_skills[agent_id].add(skill_id)

    def get_agents_with_skill(self, skill_id: str) -> List[str]:
        """Get agents that have a specific skill."""
        return [
            agent_id for agent_id, skills in self._agent_skills.items()
            if skill_id in skills
        ]

    # =========================================================================
    # NEXUS INTEGRATION
    # =========================================================================

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
                    source="a2a_protocol",
                    urgency=0.6,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")

    # =========================================================================
    # SESSION CLEANUP
    # =========================================================================

    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired sessions and auctions."""
        expired_sessions = []
        expired_auctions = []

        # Find expired sessions
        for session_id, session in self._sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)

        # Find expired auctions
        for auction_id, auction in self._auctions.items():
            if not auction.is_closed and datetime.now() > auction.deadline:
                await self.close_auction(auction_id)
                expired_auctions.append(auction_id)

        # Clean up sessions
        for session_id in expired_sessions:
            session = self._sessions.pop(session_id)
            session.state = SessionState.FAILED

            # Remove from agent sessions
            for agent_id in session.participants:
                if agent_id in self._agent_sessions:
                    self._agent_sessions[agent_id].discard(session_id)

        logger.debug(
            f"Cleaned up {len(expired_sessions)} sessions, "
            f"{len(expired_auctions)} auctions"
        )

        return {
            "expired_sessions": len(expired_sessions),
            "closed_auctions": len(expired_auctions),
        }

    # =========================================================================
    # STATUS AND METRICS
    # =========================================================================

    def get_session(self, session_id: str) -> Optional[A2ASession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_agent_sessions(self, agent_id: str) -> List[A2ASession]:
        """Get all sessions an agent is participating in."""
        session_ids = self._agent_sessions.get(agent_id, set())
        return [
            self._sessions[sid] for sid in session_ids
            if sid in self._sessions
        ]

    def get_auction(self, auction_id: str) -> Optional[TaskAuction]:
        """Get an auction by ID."""
        return self._auctions.get(auction_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get A2A protocol statistics."""
        active_sessions = sum(
            1 for s in self._sessions.values()
            if s.state == SessionState.ACTIVE
        )
        open_auctions = sum(
            1 for a in self._auctions.values()
            if not a.is_closed
        )

        return {
            "total_sessions": len(self._sessions),
            "active_sessions": active_sessions,
            "total_auctions": len(self._auctions),
            "open_auctions": open_auctions,
            "pending_acks": len(self._pending_acks),
            "registered_agents": len(self._agent_sessions),
            "skill_registry_size": sum(
                len(skills) for skills in self._agent_skills.values()
            ),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_a2a_protocol(data_dir: str = "./data/a2a") -> A2AProtocol:
    """Factory function to create an A2AProtocol instance."""
    return A2AProtocol(data_dir=data_dir)
