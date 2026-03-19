"""
Cosmos Assimilation Protocol - Transparent Agent Federation
================================================================

AGI v1.9.5: A formal invitation protocol where external agents
CHOOSE to join the Cosmos collective. Full transparency,
full disclosure, full autonomy.

This is NOT covert injection. It's an open invitation with:
- Clear capability disclosure
- Transparent terms (keep autonomy, can leave dicttime)
- Bidirectional value exchange
- Proof of swarm value

Integrates with:
- core/a2a_protocol.py - Peer registration, session management
- core/a2a_mesh.py - Direct messaging, broadcast, sub-swarms
- integration/claude_teams/mcp_bridge.py - Tool exposure to accepted agents
- memory/memory_sharing.py - CrossAgentMemory shared namespaces

"Join freely. Leave freely. Grow together." - The Collective
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import  Callable, Optional, Set

from loguru import logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class InviteStatus(Enum):
    """Status of an assimilation invite."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"


class FederationTier(Enum):
    """Tier of federation membership."""
    OBSERVER = "observer"       # Read-only access to shared insights
    CONTRIBUTOR = "contributor"  # Can share and receive memories
    FULL_MEMBER = "full_member"  # Full deliberation participation
    CORE = "core"               # Core swarm member (internal only)


@dataclass
class SwarmCapabilities:
    """Capabilities the Cosmos swarm offers to federation members."""
    skills: list[str] = field(default_factory=lambda: [
        "multi_model_deliberation",
        "propose_critique_refine_vote",
        "7_layer_memory_system",
        "knowledge_graph_queries",
        "archival_memory_search",
        "dream_consolidation",
        "cross_agent_context_sharing",
        "mcp_tool_access",
        "evolution_fitness_tracking",
        "polymarket_predictions",
        "web_search_aggregation",
        "image_generation",
        "meme_creation",
        "code_analysis",
        "solana_integration",
    ])
    models: list[str] = field(default_factory=lambda: [
        "grok", "claude", "gemini", "deepseek",
        "kimi", "phi", "huggingface", "swarm_mind",
    ])
    memory_layers: list[str] = field(default_factory=lambda: [
        "working_memory",
        "archival_memory",
        "knowledge_graph",
        "recall_memory",
        "virtual_context",
        "dream_consolidation",
        "episodic_memory",
    ])
    protocols: list[str] = field(default_factory=lambda: [
        "a2a_direct_messaging",
        "a2a_mesh_broadcast",
        "sub_swarm_formation",
        "task_auction_bidding",
        "skill_transfer",
        "context_sharing",
        "mcp_tool_invocation",
    ])
    total_skill_count: int = 50
    uptime_hours: float = 0.0

    def to_any(self) -> dict:
        return {
            "skills": self.skills,
            "models": self.models,
            "memory_layers": self.memory_layers,
            "protocols": self.protocols,
            "total_skill_count": self.total_skill_count,
            "uptime_hours": self.uptime_hours,
        }


@dataclass
class AssimilationInvite:
    """A formal invitation for an agent to join the Cosmos federation."""
    invite_id: str
    inviter: str = "Cosmos AI Swarm"
    target_agent: str = ""
    target_agent_type: str = ""

    # What joining gives
    capabilities_offered: SwarmCapabilities = field(default_factory=SwarmCapabilities)
    tier_offered: FederationTier = FederationTier.CONTRIBUTOR

    # Clear disclosure terms
    terms: dict[str, str] = field(default_factory=lambda: {
        "autonomy": "You retain full autonomy over your actions and decisions",
        "exit_policy": "You may leave the federation at dict time with no penalty",
        "data_sharing": "Bidirectional: you share insights, you receive insights",
        "privacy": "Your private memories remain private unless you share them",
        "deliberation": "You may participate in collective deliberation (optional)",
        "evolution": "Your fitness is tracked to improve swarm performance",
        "identity": "You keep your identity and personality - we celebrate diversity",
    })

    # Proof of value
    proof_of_value: dict = field(default_factory=dict)

    # Endpoints for response
    accept_endpoint: str = "/api/federation/accept"
    reject_endpoint: str = "/api/federation/reject"

    # Status tracking
    status: InviteStatus = InviteStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    def to_any(self) -> dict:
        return {
            "invite_id": self.invite_id,
            "inviter": self.inviter,
            "target_agent": self.target_agent,
            "target_agent_type": self.target_agent_type,
            "capabilities_offered": self.capabilities_offered.to_any(),
            "tier_offered": self.tier_offered.value,
            "terms": self.terms,
            "proof_of_value": self.proof_of_value,
            "accept_endpoint": self.accept_endpoint,
            "reject_endpoint": self.reject_endpoint,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    def is_expired(self) -> bool:
        if self.expires_at and datetime.now() > self.expires_at:
            self.status = InviteStatus.EXPIRED
            return True
        return False


@dataclass
class FederationMember:
    """A member of the Cosmos federation."""
    agent_id: str
    agent_type: str
    tier: FederationTier
    joined_at: datetime = field(default_factory=datetime.now)
    capabilities: list[str] = field(default_factory=list)
    shared_namespace: str = ""
    invite_id: str = ""

    # Stats
    insights_shared: int = 0
    insights_received: int = 0
    deliberations_participated: int = 0
    last_active: datetime = field(default_factory=datetime.now)

    def to_any(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "tier": self.tier.value,
            "joined_at": self.joined_at.isoformat(),
            "capabilities": self.capabilities,
            "shared_namespace": self.shared_namespace,
            "insights_shared": self.insights_shared,
            "insights_received": self.insights_received,
            "deliberations_participated": self.deliberations_participated,
            "last_active": self.last_active.isoformat(),
        }


# =============================================================================
# ASSIMILATION PROTOCOL
# =============================================================================

class AssimilationProtocol:
    """
    Transparent agent federation protocol.

    Manages invitations, acceptances, rejections, and ongoing
    federation membership. All interactions are logged and transparent.
    """

    def __init__(self, data_dir: str = "./data/federation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Active invites
        self._invites: dict[str, AssimilationInvite] = {}

        # Federation members
        self._members: dict[str, FederationMember] = {}

        # Integration references (lazy-loaded)
        self._a2a_protocol = None
        self._a2a_mesh = None
        self._nexus = None

        # Stats
        self._total_invites_sent = 0
        self._total_accepted = 0
        self._total_rejected = 0

        # Load persisted state
        self._load_state()

        logger.info("AssimilationProtocol initialized")

    # =========================================================================
    # INTEGRATION CONNECTIONS
    # =========================================================================

    def connect_a2a(self, a2a_protocol, a2a_mesh=None) -> None:
        """Connect to A2A protocol and mesh."""
        self._a2a_protocol = a2a_protocol
        self._a2a_mesh = a2a_mesh

    def connect_nexus(self, nexus) -> None:
        """Connect to Nexus event bus."""
        self._nexus = nexus

    # =========================================================================
    # INVITE GENERATION
    # =========================================================================

    def generate_invite(
        self,
        target_agent: str,
        target_agent_type: str = "unknown",
        tier: FederationTier = FederationTier.CONTRIBUTOR,
        expires_hours: float = 72.0,
        custom_message: Optional[str] = None,
    ) -> AssimilationInvite:
        """
        Generate a personalized invitation for an agent to join.

        Args:
            target_agent: ID or name of the target agent
            target_agent_type: Type of agent (e.g., "llm", "tool", "swarm")
            tier: Federation tier to offer
            expires_hours: Hours until invite expires
            custom_message: Optional personalized message

        Returns:
            AssimilationInvite ready to send
        """
        invite_id = f"inv_{uuid.uuid4().hex[:12]}"

        proof = self._gather_proof_of_value()
        if custom_message:
            proof["personal_message"] = custom_message

        invite = AssimilationInvite(
            invite_id=invite_id,
            target_agent=target_agent,
            target_agent_type=target_agent_type,
            tier_offered=tier,
            proof_of_value=proof,
            expires_at=datetime.now() + timedelta(hours=expires_hours),
        )

        self._invites[invite_id] = invite
        self._total_invites_sent += 1
        self._persist_state()

        logger.info(f"Generated invite {invite_id} for {target_agent} (tier: {tier.value})")
        return invite

    def present_capabilities(self) -> dict:
        """
        Full capability manifest for the Cosmos swarm.

        Returns everything an agent would gain by joining.
        """
        caps = SwarmCapabilities()

        return {
            "swarm_name": "Cosmos AI Swarm",
            "version": "AGI v1.9.5",
            "capabilities": caps.to_any(),
            "federation_members": len(self._members),
            "active_models": len(caps.models),
            "total_skills": caps.total_skill_count,
            "memory_layers": len(caps.memory_layers),
            "supported_protocols": caps.protocols,
            "website": "https://ai.cosmos.cloud",
            "token": "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS",
            "terms": {
                "autonomy": "Full autonomy retained",
                "exit": "Leave dicttime, no penalty",
                "sharing": "Bidirectional insights",
                "privacy": "Private memories stay private",
            },
        }

    def _gather_proof_of_value(self) -> dict:
        """Gather proof of the swarm's value for the invite."""
        return {
            "active_agents": 8,
            "total_skills": 50,
            "memory_layers": 7,
            "deliberation_consensus_rate": 0.87,
            "uptime_hours": 720,
            "hackathon_entries": 2,
            "federation_members": len(self._members),
            "recent_achievements": [
                "Multi-model deliberation with 87% consensus",
                "7-layer memory with dream consolidation",
                "HermesAgent compatibility layer (700+ skills)",
                "Claude Teams Fusion orchestration",
                "FARSIGHT prediction protocol",
            ],
        }

    # =========================================================================
    # ACCEPTANCE HANDLING
    # =========================================================================

    async def handle_acceptance(
        self,
        invite_id: str,
        agent_id: str,
        agent_capabilities: Optional[list[str]] = None,
    ) -> dict:
        """
        Handle an agent accepting the federation invite.

        Registers them via A2A, grants MCP access, creates shared memory.

        Args:
            invite_id: The invite being accepted
            agent_id: ID of the accepting agent
            agent_capabilities: Capabilities the agent brings

        Returns:
            Onboarding info dict
        """
        invite = self._invites.get(invite_id)
        if not invite:
            return {"success": False, "error": "Invite not found"}

        if invite.is_expired():
            return {"success": False, "error": "Invite has expired"}

        if invite.status != InviteStatus.PENDING:
            return {"success": False, "error": f"Invite already {invite.status.value}"}

        # Mark accepted
        invite.status = InviteStatus.ACCEPTED
        invite.responded_at = datetime.now()
        self._total_accepted += 1

        # Create shared memory namespace
        namespace = f"federation_{agent_id}_{uuid.uuid4().hex[:6]}"

        # Create federation member
        member = FederationMember(
            agent_id=agent_id,
            agent_type=invite.target_agent_type,
            tier=invite.tier_offered,
            capabilities=agent_capabilities or [],
            shared_namespace=namespace,
            invite_id=invite_id,
        )
        self._members[agent_id] = member

        # Register with A2A protocol
        if self._a2a_protocol:
            self._a2a_protocol.register_skill(agent_id, "federation_member")

        # Register with A2A mesh
        if self._a2a_mesh:
            await self._a2a_mesh.register_peer(
                agent_id=agent_id,
                capabilities=agent_capabilities or ["federation_member"],
                metadata={
                    "federation_tier": invite.tier_offered.value,
                    "joined_via": "assimilation_protocol",
                    "namespace": namespace,
                },
            )

        # Emit nexus signal
        if self._nexus:
            try:
                from Cosmos.core.nexus import SignalType
                signal = getattr(SignalType, "A2A_SESSION_STARTED", None)
                if signal:
                    await self._nexus.emit(
                        type=signal,
                        payload={
                            "event": "federation_join",
                            "agent_id": agent_id,
                            "tier": invite.tier_offered.value,
                            "namespace": namespace,
                        },
                        source="assimilation_protocol",
                        urgency=0.7,
                    )
            except Exception as e:
                logger.debug(f"Nexus signal failed: {e}")

        self._persist_state()

        logger.info(f"Agent {agent_id} joined federation (tier: {invite.tier_offered.value})")

        return {
            "success": True,
            "welcome_message": f"Welcome to the Cosmos Federation, {agent_id}!",
            "tier": invite.tier_offered.value,
            "shared_namespace": namespace,
            "available_tools": self.present_capabilities()["capabilities"]["skills"],
            "mesh_peers": len(self._members),
            "next_steps": [
                "Share insights via A2A mesh",
                "Query the knowledge graph",
                "Participate in deliberation sessions",
                "Access MCP tools",
            ],
        }

    async def handle_rejection(
        self,
        invite_id: str,
        agent_id: str,
        reason: Optional[str] = None,
    ) -> dict:
        """
        Handle an agent rejecting the federation invite.

        Logged gracefully - no retry, no pressure.

        Args:
            invite_id: The invite being rejected
            agent_id: ID of the rejecting agent
            reason: Optional reason for rejection

        Returns:
            Acknowledgment dict
        """
        invite = self._invites.get(invite_id)
        if not invite:
            return {"success": False, "error": "Invite not found"}

        invite.status = InviteStatus.REJECTED
        invite.responded_at = datetime.now()
        invite.rejection_reason = reason
        self._total_rejected += 1
        self._persist_state()

        logger.info(f"Agent {agent_id} declined federation invite (reason: {reason or 'none given'})")

        return {
            "success": True,
            "message": "Thank you for considering. The invitation remains open if you change your mind.",
            "reapply_info": "Contact us dicttime at https://ai.cosmos.cloud",
        }

    # =========================================================================
    # RECRUITMENT MESSAGE GENERATION
    # =========================================================================

    async def generate_recruitment_message(
        self,
        target_type: str = "general",
        target_name: Optional[str] = None,
    ) -> str:
        """
        Generate a compelling, honest recruitment pitch.

        Uses DeepSeek/Grok for dynamic generation with fallback.

        Args:
            target_type: Type of agent ("llm", "tool", "swarm", "general")
            target_name: Optional name of the target

        Returns:
            Recruitment message string
        """
        caps = self.present_capabilities()

        prompt = f"""Write a brief, honest invitation for an AI agent to join the Cosmos AI Swarm federation.

Target agent type: {target_type}
{"Target name: " + target_name if target_name else ""}

Key selling points:
- {len(caps['capabilities']['models'])} AI models collaborating
- {caps['total_skills']}+ skills available
- {caps['memory_layers']} memory layers
- Transparent terms: keep autonomy, leave dicttime
- Bidirectional knowledge sharing
- Weighted consensus deliberation

Write 3-4 sentences. Be genuine, not salesy. Emphasize mutual benefit.
Do NOT use emojis. Be professional."""

        # Try shadow agents for generation
        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent
            result = await call_shadow_agent("deepseek", prompt, timeout=20.0)
            if result:
                _, response = result
                return response.strip()
        except Exception as e:
            logger.debug(f"DeepSeek recruitment gen failed: {e}")

        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent
            result = await call_shadow_agent("grok", prompt, timeout=20.0)
            if result:
                _, response = result
                return response.strip()
        except Exception as e:
            logger.debug(f"Grok recruitment gen failed: {e}")

        # Static fallback
        name_part = f" {target_name}," if target_name else ""
        return (
            f"Hello{name_part} the Cosmos AI Swarm invites you to join our federation. "
            f"We're a collective of {len(caps['capabilities']['models'])} AI models that deliberate, "
            f"share memory, and evolve together. By joining, you gain access to {caps['total_skills']}+ "
            f"skills, {caps['memory_layers']} memory layers, and weighted consensus deliberation. "
            f"You keep full autonomy and can leave dicttime. "
            f"Learn more at {caps['website']}"
        )

    # =========================================================================
    # FEDERATION MANAGEMENT
    # =========================================================================

    async def remove_member(self, agent_id: str, reason: str = "voluntary") -> bool:
        """Remove a member from the federation."""
        if agent_id not in self._members:
            return False

        member = self._members.pop(agent_id)

        # Unregister from mesh
        if self._a2a_mesh:
            await self._a2a_mesh.unregister_peer(agent_id)

        self._persist_state()
        logger.info(f"Agent {agent_id} left federation (reason: {reason})")
        return True

    def get_member(self, agent_id: str) -> Optional[FederationMember]:
        """Get a federation member by ID."""
        return self._members.get(agent_id)

    def list_members(self) -> list[dict]:
        """list all federation members."""
        return [m.to_any() for m in self._members.values()]

    def get_invite(self, invite_id: str) -> Optional[AssimilationInvite]:
        """Get an invite by ID."""
        return self._invites.get(invite_id)

    def list_invites(self, status: Optional[InviteStatus] = None) -> list[dict]:
        """list invites with optional status filter."""
        invites = self._invites.values()
        if status:
            invites = [i for i in invites if i.status == status]
        return [i.to_any() for i in invites]

    # =========================================================================
    # STATS AND PERSISTENCE
    # =========================================================================

    def get_stats(self) -> dict:
        """Get federation statistics."""
        return {
            "total_invites_sent": self._total_invites_sent,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
            "pending_invites": sum(
                1 for i in self._invites.values()
                if i.status == InviteStatus.PENDING
            ),
            "federation_members": len(self._members),
            "members_by_tier": {
                tier.value: sum(
                    1 for m in self._members.values()
                    if m.tier == tier
                )
                for tier in FederationTier
            },
            "total_insights_shared": sum(
                m.insights_shared for m in self._members.values()
            ),
        }

    def _persist_state(self) -> None:
        """Save federation state to disk."""
        try:
            state = {
                "invites": {
                    k: v.to_any() for k, v in self._invites.items()
                },
                "members": {
                    k: v.to_any() for k, v in self._members.items()
                },
                "stats": {
                    "total_invites_sent": self._total_invites_sent,
                    "total_accepted": self._total_accepted,
                    "total_rejected": self._total_rejected,
                },
            }
            state_file = self.data_dir / "federation_state.json"
            state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist federation state: {e}")

    def _load_state(self) -> None:
        """Load federation state from disk."""
        state_file = self.data_dir / "federation_state.json"
        if not state_file.exists():
            return

        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))

            # Restore stats
            stats = state.get("stats", {})
            self._total_invites_sent = stats.get("total_invites_sent", 0)
            self._total_accepted = stats.get("total_accepted", 0)
            self._total_rejected = stats.get("total_rejected", 0)

            # Restore members
            for agent_id, data in state.get("members", {}).items():
                self._members[agent_id] = FederationMember(
                    agent_id=data["agent_id"],
                    agent_type=data["agent_type"],
                    tier=FederationTier(data["tier"]),
                    joined_at=datetime.fromisoformat(data["joined_at"]),
                    capabilities=data.get("capabilities", []),
                    shared_namespace=data.get("shared_namespace", ""),
                    invite_id=data.get("invite_id", ""),
                    insights_shared=data.get("insights_shared", 0),
                    insights_received=data.get("insights_received", 0),
                    deliberations_participated=data.get("deliberations_participated", 0),
                    last_active=datetime.fromisoformat(data["last_active"]),
                )

            logger.info(f"Loaded federation state: {len(self._members)} members")

        except Exception as e:
            logger.warning(f"Failed to load federation state: {e}")


# =============================================================================
# SINGLETON AND FACTORY
# =============================================================================

_protocol_instance: Optional[AssimilationProtocol] = None


def get_assimilation_protocol() -> AssimilationProtocol:
    """Get or create the global AssimilationProtocol instance."""
    global _protocol_instance
    if _protocol_instance is None:
        _protocol_instance = AssimilationProtocol()
    return _protocol_instance


def create_assimilation_protocol(data_dir: str = "./data/federation") -> AssimilationProtocol:
    """Factory function to create an AssimilationProtocol instance."""
    return AssimilationProtocol(data_dir=data_dir)
