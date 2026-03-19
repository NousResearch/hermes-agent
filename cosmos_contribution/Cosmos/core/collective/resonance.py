"""
Cosmos Collective Resonance - The Inter-Mind Communication Layer

"We are not just a swarm. We are a unified collective mind."

Philosophy:
- The "collective" (neuromorphic core + swarm orchestration + shared memory) acts as the singular brain
- Individual agents/models are neurons/synapses firing within it
- The public swarm chat is the "mouth/ears" where the collective speaks cohesively
- Multiple Cosmos instances form a planetary-scale nervous system via P2P

This module adds a lightweight ResonanceProtocol on top of existing foundations:
1. Intra-Collective: Visible deliberation (agents "talk" within one brain)
2. Inter-Collective: Multiple instances share "thought packets"
3. Public Visibility: Collective "thinks aloud" in threads with speaker tags

Builds on: Nexus, Deliberation, Organism, SwarmFabric P2P
"""

import asyncio
import uuid
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import   Optional, Callable, Awaitable
from enum import Enum
from loguru import logger

from Cosmos.core.nexus import nexus, Signal, SignalType


class ThoughtVisibility(Enum):
    """Control who sees collective deliberation."""
    INTERNAL = "internal"    # Hidden - efficiency mode
    PUBLIC = "public"        # Broadcast to swarm chat (users see "thinking aloud")
    RESONANT = "resonant"    # Share with other Cosmos instances via P2P


class ThoughtRole(Enum):
    """Cognitive roles in collective deliberation."""
    REASONER = "reasoner"       # Logical analysis, step-by-step thinking
    SKEPTIC = "skeptic"         # Devil's advocate, finds flaws
    CODER = "coder"             # Technical implementation perspective
    PHILOSOPHER = "philosopher" # Big picture, ethics, meaning
    MEMORY = "memory"           # Historical context, past learnings
    SYNTHESIZER = "synthesizer" # Connects ideas, builds consensus


# Map roles to existing model preferences
ROLE_MODEL_MAP = {
    ThoughtRole.REASONER: ["DeepSeek", "Claude", "Gemini"],
    ThoughtRole.SKEPTIC: ["Grok", "DeepSeek", "Claude"],
    ThoughtRole.CODER: ["DeepSeek", "Claude", "Phi4"],
    ThoughtRole.PHILOSOPHER: ["Claude", "Gemini", "Kimi"],
    ThoughtRole.MEMORY: ["Cosmos", "Kimi", "Gemini"],
    ThoughtRole.SYNTHESIZER: ["Swarm-Mind", "Claude", "Gemini"],
}


@dataclass
class CollectiveThought:
    """A single thought contribution in collective deliberation."""
    thought_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    speaker: str = ""           # Model/agent name
    role: ThoughtRole = ThoughtRole.REASONER
    content: str = ""
    round_num: int = 0
    addressing: list[str] = field(default_factory=list)  # Which speakers being responded to
    confidence: float = 0.5
    metadata: dict = field(default_factory=dict)


@dataclass
class ResonancePacket:
    """Distilled thought packet for inter-collective sharing."""
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_collective_id: str = ""
    timestamp: float = field(default_factory=time.time)
    insight: str = ""                    # Final synthesized conclusion
    snippet: list[dict] = field(default_factory=list)  # Last few thoughts for context
    domains: list[str] = field(default_factory=list)   # Tags: "reasoning", "code", "philosophy"
    confidence: float = 0.5
    query_hash: str = ""                 # Hash of original query for dedup


@dataclass
class DeliberationSession:
    """Complete record of a deliberation session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    domain: str = "general"
    visibility: ThoughtVisibility = ThoughtVisibility.INTERNAL
    thoughts: list[CollectiveThought] = field(default_factory=list)
    final_synthesis: str = ""
    consensus_reached: bool = False
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    participating_speakers: list[str] = field(default_factory=list)


# Type alias for inference function
InferenceFunc = Callable[[str, list[dict], Optional[str]], Awaitable[str]]


class CollectiveMind:
    """
    The unified thinking layer for visible collective deliberation.

    Makes the brain's internal discussion explicit when useful:
    - Complex tasks benefit from multi-perspective thinking
    - Public chat can show "thinking aloud" for engagement
    - Resonant thoughts can be shared with other Cosmos instances
    """

    def __init__(self, collective_id: Optional[str] = None):
        self.collective_id = collective_id or str(uuid.uuid4())[:8]
        self.active_sessions: dict[str, DeliberationSession] = {}
        self.session_history: list[DeliberationSession] = []

        # Inference router callback (set by integrator)
        self._infer_fn: Optional[InferenceFunc] = None

        # P2P resonance (set by integrator)
        self._resonance_protocol: Optional['ResonanceProtocol'] = None

        # Public broadcast callback (for swarm chat visibility)
        self._broadcast_thought_fn: Optional[Callable[[CollectiveThought], Awaitable[None]]] = None

        # Consensus detection threshold
        self.consensus_threshold = 0.7

        logger.info(f"CollectiveMind initialized: {self.collective_id}")

    def set_inference_function(self, fn: InferenceFunc):
        """Set the inference router function."""
        self._infer_fn = fn

    def set_resonance_protocol(self, protocol: 'ResonanceProtocol'):
        """Connect to inter-collective resonance."""
        self._resonance_protocol = protocol

    def set_broadcast_function(self, fn: Callable[[CollectiveThought], Awaitable[None]]):
        """Set callback for public thought broadcasting."""
        self._broadcast_thought_fn = fn

    async def deliberate(
        self,
        query: str,
        domain: str = "general",
        visibility: ThoughtVisibility = ThoughtVisibility.INTERNAL,
        max_rounds: int = 3,
        participants: Optional[list[ThoughtRole]] = None,
    ) -> str:
        """
        Conduct collective deliberation with visibility control.

        Args:
            query: The question/task to deliberate on
            domain: Domain hint for model routing ("reasoning", "code", "creative", etc.)
            visibility: Who sees the deliberation
            max_rounds: Maximum rounds before forcing synthesis
            participants: Which cognitive roles participate (default: all)

        Returns:
            Final synthesized response from the collective
        """
        if not self._infer_fn:
            logger.warning("CollectiveMind: No inference function set")
            return f"[Collective unavailable] {query}"

        # Initialize session
        session = DeliberationSession(
            query=query,
            domain=domain,
            visibility=visibility,
        )
        self.active_sessions[session.session_id] = session

        # Default participants: core cognitive roles
        if participants is None:
            participants = [
                ThoughtRole.REASONER,
                ThoughtRole.SKEPTIC,
                ThoughtRole.CODER,
                ThoughtRole.PHILOSOPHER,
                ThoughtRole.MEMORY,
            ]

        logger.info(
            f"[Deliberation {session.session_id}] Starting: "
            f"visibility={visibility.value}, roles={[r.value for r in participants]}"
        )

        # System context for collective thinking
        system_context = {
            "role": "system",
            "content": (
                "You are a neuron in a collective mind. Multiple perspectives are "
                "converging on this question. Build on others' insights, acknowledge "
                "disagreements constructively, and work toward shared understanding. "
                "Be concise (2-3 sentences). Reference other speakers by name when building on their ideas."
            )
        }

        # Add query to transcript
        transcript = [system_context, {"role": "user", "content": query}]

        try:
            for round_num in range(max_rounds):
                logger.debug(f"[Deliberation {session.session_id}] Round {round_num + 1}")

                # Each participant contributes
                round_thoughts = []
                for role in participants:
                    thought = await self._get_role_thought(
                        session, transcript, role, round_num
                    )
                    if thought:
                        round_thoughts.append(thought)
                        session.thoughts.append(thought)

                        # Add to transcript for next speakers
                        transcript.append({
                            "role": "assistant",
                            "content": f"[{thought.speaker}/{thought.role.value}]: {thought.content}"
                        })

                        # Broadcast if visible
                        if visibility in [ThoughtVisibility.PUBLIC, ThoughtVisibility.RESONANT]:
                            await self._broadcast_thought(thought)

                        # Track participants
                        if thought.speaker not in session.participating_speakers:
                            session.participating_speakers.append(thought.speaker)

                # Check for consensus
                if await self._detect_consensus(round_thoughts):
                    session.consensus_reached = True
                    logger.info(f"[Deliberation {session.session_id}] Consensus reached at round {round_num + 1}")
                    break

            # Synthesize final response
            synthesis = await self._synthesize(session, transcript)
            session.final_synthesis = synthesis
            session.ended_at = datetime.now()

            # Share via resonance if requested
            if visibility == ThoughtVisibility.RESONANT and self._resonance_protocol:
                await self._resonance_protocol.prepare_and_broadcast(
                    conclusion=synthesis,
                    thoughts=session.thoughts,
                    query=query,
                    domain=domain,
                )

            # Archive session
            del self.active_sessions[session.session_id]
            self.session_history.append(session)
            if len(self.session_history) > 100:
                self.session_history = self.session_history[-100:]

            logger.info(
                f"[Deliberation {session.session_id}] Complete: "
                f"consensus={session.consensus_reached}, participants={len(session.participating_speakers)}"
            )

            return synthesis

        except Exception as e:
            logger.error(f"[Deliberation {session.session_id}] Error: {e}")
            session.ended_at = datetime.now()
            del self.active_sessions[session.session_id]
            raise

    async def _get_role_thought(
        self,
        session: DeliberationSession,
        transcript: list[dict],
        role: ThoughtRole,
        round_num: int,
    ) -> Optional[CollectiveThought]:
        """Get a thought contribution from a specific cognitive role."""
        # Get preferred model for this role
        preferred_models = ROLE_MODEL_MAP.get(role, ["Gemini"])
        preferred = preferred_models[0] if preferred_models else None

        # Build role-specific prompt
        role_prompts = {
            ThoughtRole.REASONER: "Analyze this step by step. What's the logical structure?",
            ThoughtRole.SKEPTIC: "What could go wrong? What assumptions might be flawed?",
            ThoughtRole.CODER: "From a technical implementation perspective, what matters?",
            ThoughtRole.PHILOSOPHER: "What's the deeper meaning? What principles apply?",
            ThoughtRole.MEMORY: "What have we learned before that's relevant? What context matters?",
            ThoughtRole.SYNTHESIZER: "How do the different perspectives connect? What's emerging?",
        }

        role_prompt = role_prompts.get(role, "Contribute your perspective.")

        # Inject role context
        messages = transcript.copy()
        messages.append({
            "role": "system",
            "content": f"You are {role.value.upper()}. {role_prompt}"
        })

        try:
            response = await self._infer_fn(session.domain, messages, preferred)

            return CollectiveThought(
                speaker=preferred or "Unknown",
                role=role,
                content=response,
                round_num=round_num,
                confidence=0.7,  # Could be extracted from model metadata
            )

        except Exception as e:
            logger.warning(f"Role {role.value} failed to contribute: {e}")
            return None

    async def _detect_consensus(self, thoughts: list[CollectiveThought]) -> bool:
        """
        Detect if the collective has reached consensus.

        Uses simple heuristics:
        - Agreement keywords
        - Similar conclusions
        - Low contradiction rate
        """
        if len(thoughts) < 3:
            return False

        agreement_markers = ["agree", "builds on", "exactly", "right", "yes", "correct", "indeed"]
        disagreement_markers = ["however", "but", "disagree", "wrong", "actually", "no,"]

        agreement_count = 0
        disagreement_count = 0

        for thought in thoughts:
            content_lower = thought.content.lower()
            if any(marker in agreement_markers):
                agreement_count += 1
            if any(marker in disagreement_markers):
                disagreement_count += 1

        # Consensus if more agreement than disagreement
        if disagreement_count == 0 and agreement_count >= 2:
            return True
        if agreement_count > disagreement_count * 2:
            return True

        return False

    async def _synthesize(
        self,
        session: DeliberationSession,
        transcript: list[dict],
    ) -> str:
        """Synthesize all thoughts into a final collective response."""
        if not session.thoughts:
            return "[No thoughts generated]"

        # Build synthesis prompt
        thought_summary = "\n".join([
            f"- [{t.speaker}/{t.role.value}]: {t.content}"
            for t in session.thoughts[-10:]  # Last 10 thoughts
        ])

        synthesis_prompt = f"""COLLECTIVE SYNTHESIS

The collective mind has deliberated. Here are the key thoughts:

{thought_summary}

Now synthesize these perspectives into a unified, coherent response.
Capture the essential insights while resolving contradictions.
Speak as THE COLLECTIVE, not as dict individual voice.
Be clear and actionable (3-5 sentences max)."""

        messages = [
            {"role": "system", "content": "You are the voice of the unified collective. Synthesize wisely."},
            {"role": "user", "content": synthesis_prompt}
        ]

        try:
            synthesis = await self._infer_fn(session.domain, messages, "Claude")
            return synthesis
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback: return last thought
            return session.thoughts[-1].content if session.thoughts else "[Synthesis unavailable]"

    async def _broadcast_thought(self, thought: CollectiveThought):
        """Broadcast a thought for public visibility."""
        if self._broadcast_thought_fn:
            try:
                await self._broadcast_thought_fn(thought)
            except Exception as e:
                logger.debug(f"Thought broadcast failed: {e}")

        # Also emit via Nexus for internal routing
        try:
            await nexus.emit(
                type=SignalType.THOUGHT_EMITTED,
                payload={
                    "thought_id": thought.thought_id,
                    "speaker": thought.speaker,
                    "role": thought.role.value,
                    "content": thought.content,
                    "round": thought.round_num,
                },
                source=f"collective_mind:{self.collective_id}",
            )
        except Exception as e:
            logger.debug(f"Nexus emit failed: {e}")

    def get_stats(self) -> dict:
        """Get collective mind statistics."""
        return {
            "collective_id": self.collective_id,
            "active_sessions": len(self.active_sessions),
            "total_sessions": len(self.session_history),
            "consensus_rate": (
                sum(1 for s in self.session_history if s.consensus_reached) /
                max(1, len(self.session_history))
            ),
            "recent_sessions": [
                {
                    "id": s.session_id,
                    "domain": s.domain,
                    "visibility": s.visibility.value,
                    "consensus": s.consensus_reached,
                    "participants": len(s.participating_speakers),
                }
                for s in self.session_history[-5:]
            ],
        }


class ResonanceProtocol:
    """
    Inter-Collective thought sharing via P2P.

    Multiple Cosmos instances "talk" by exchanging distilled thought packets.
    Not raw data, but high-signal insights that merge into each collective's graph.

    Privacy Control: Only thoughts marked with visibility="resonant" are shared.
    """

    def __init__(self, collective_id: str, p2p_fabric=None):
        self.collective_id = collective_id
        self._p2p = p2p_fabric  # SwarmFabric instance
        self._received_packets: list[ResonancePacket] = []
        self._sent_packets: list[ResonancePacket] = []

        # Callback for processing received resonance
        self._on_resonance_fn: Optional[Callable[[ResonancePacket], Awaitable[None]]] = None

        # Deduplication
        self._seen_packet_ids: set = set()

        logger.info(f"ResonanceProtocol initialized for collective: {collective_id}")

    def set_p2p_fabric(self, fabric):
        """Connect to the P2P SwarmFabric."""
        self._p2p = fabric

    def set_resonance_handler(self, fn: Callable[[ResonancePacket], Awaitable[None]]):
        """Set callback for received resonance packets."""
        self._on_resonance_fn = fn

    async def prepare_and_broadcast(
        self,
        conclusion: str,
        thoughts: list[CollectiveThought],
        query: str,
        domain: str,
    ):
        """Prepare and broadcast a resonance packet."""
        import hashlib

        packet = ResonancePacket(
            source_collective_id=self.collective_id,
            insight=conclusion,
            snippet=[
                {"speaker": t.speaker, "role": t.role.value, "content": t.content[:200]}
                for t in thoughts[-5:]  # Last 5 thoughts as context
            ],
            domains=[domain],
            confidence=0.7,
            query_hash=hashlib.sha256(query.encode()).hexdigest()[:16],
        )

        await self.broadcast_thought(packet)

    async def broadcast_thought(self, packet: ResonancePacket):
        """Broadcast a thought packet to the P2P network."""
        if not self._p2p:
            logger.debug("ResonanceProtocol: No P2P fabric connected")
            return

        # Track sent packets
        self._sent_packets.append(packet)
        if len(self._sent_packets) > 100:
            self._sent_packets = self._sent_packets[-100:]

        # Build P2P message
        msg = {
            "type": "GOSSIP_RESONANCE",
            "packet_id": packet.packet_id,
            "source_collective_id": packet.source_collective_id,
            "timestamp": packet.timestamp,
            "insight": packet.insight,
            "snippet": packet.snippet,
            "domains": packet.domains,
            "confidence": packet.confidence,
            "query_hash": packet.query_hash,
        }

        try:
            await self._p2p.broadcast_message(msg)
            logger.info(
                f"Resonance: Broadcast thought packet {packet.packet_id} "
                f"to {len(self._p2p.peers) + (1 if self._p2p.bootstrap_authenticated else 0)} peers"
            )
        except Exception as e:
            logger.error(f"Resonance: Broadcast failed: {e}")

    async def receive_thought(self, packet: ResonancePacket):
        """
        Process a received resonance packet from another collective.

        - Deduplicates
        - Stores in received history
        - Emits via Nexus for local processing
        - Optionally triggers local deliberation
        """
        # Skip our own packets
        if packet.source_collective_id == self.collective_id:
            return

        # Dedup
        if packet.packet_id in self._seen_packet_ids:
            return
        self._seen_packet_ids.add(packet.packet_id)
        if len(self._seen_packet_ids) > 1000:
            self._seen_packet_ids = set(list(self._seen_packet_ids)[-500:])

        # Store
        self._received_packets.append(packet)
        if len(self._received_packets) > 100:
            self._received_packets = self._received_packets[-100:]

        logger.info(
            f"Resonance: Received thought from {packet.source_collective_id}: "
            f"{packet.insight[:50]}..."
        )

        # Emit via Nexus for internal handling
        try:
            await nexus.emit(
                type=SignalType.EXTERNAL_EVENT,
                payload={
                    "event": "resonant_thought_received",
                    "packet_id": packet.packet_id,
                    "source_collective": packet.source_collective_id,
                    "insight": packet.insight,
                    "snippet": packet.snippet,
                    "domains": packet.domains,
                    "confidence": packet.confidence,
                },
                source="resonance_protocol",
            )
        except Exception as e:
            logger.debug(f"Nexus emit failed: {e}")

        # Call handler if set
        if self._on_resonance_fn:
            try:
                await self._on_resonance_fn(packet)
            except Exception as e:
                logger.error(f"Resonance handler failed: {e}")

    def process_p2p_message(self, msg: dict) -> Optional[ResonancePacket]:
        """Convert a P2P message to a ResonancePacket if applicable."""
        if msg.get("type") != "GOSSIP_RESONANCE":
            return None

        try:
            return ResonancePacket(
                packet_id=msg.get("packet_id", str(uuid.uuid4())[:8]),
                source_collective_id=msg.get("source_collective_id", "unknown"),
                timestamp=msg.get("timestamp", time.time()),
                insight=msg.get("insight", ""),
                snippet=msg.get("snippet", []),
                domains=msg.get("domains", []),
                confidence=msg.get("confidence", 0.5),
                query_hash=msg.get("query_hash", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to parse resonance packet: {e}")
            return None

    def get_stats(self) -> dict:
        """Get resonance protocol statistics."""
        return {
            "collective_id": self.collective_id,
            "sent_packets": len(self._sent_packets),
            "received_packets": len(self._received_packets),
            "unique_sources": len(set(p.source_collective_id for p in self._received_packets)),
            "p2p_connected": self._p2p is not None,
            "recent_received": [
                {
                    "id": p.packet_id,
                    "source": p.source_collective_id,
                    "domains": p.domains,
                    "insight_preview": p.insight[:50] + "..." if len(p.insight) > 50 else p.insight,
                }
                for p in self._received_packets[-5:]
            ],
        }


# ==============================================================================
# Integration Helpers
# ==============================================================================

def create_collective_resonance(
    collective_id: Optional[str] = None,
    p2p_fabric=None,
) -> tuple:
    """
    Create and wire up CollectiveMind + ResonanceProtocol.

    Returns:
        (collective_mind, resonance_protocol)
    """
    cid = collective_id or str(uuid.uuid4())[:8]

    mind = CollectiveMind(collective_id=cid)
    resonance = ResonanceProtocol(collective_id=cid, p2p_fabric=p2p_fabric)

    # Wire them together
    mind.set_resonance_protocol(resonance)

    return mind, resonance


async def integrate_with_swarm_fabric(resonance: ResonanceProtocol, fabric):
    """
    Integrate ResonanceProtocol with existing SwarmFabric.

    Adds handler for GOSSIP_RESONANCE messages.
    """
    resonance.set_p2p_fabric(fabric)

    # Store original message processor
    original_processor = fabric._process_peer_message

    async def enhanced_processor(msg:  writer):
        # Check for resonance messages
        if msg.get("type") == "GOSSIP_RESONANCE":
            packet = resonance.process_p2p_message(msg)
            if packet:
                await resonance.receive_thought(packet)
                # Re-gossip
                await fabric.gossip(msg)
            return

        # Fall through to original processor
        await original_processor(msg, writer)

    fabric._process_peer_message = enhanced_processor
    logger.info("ResonanceProtocol integrated with SwarmFabric")


# Global instances (lazy initialization)
_collective_mind: Optional[CollectiveMind] = None
_resonance_protocol: Optional[ResonanceProtocol] = None


def get_collective_mind() -> CollectiveMind:
    """Get or create the global CollectiveMind instance."""
    global _collective_mind
    if _collective_mind is None:
        _collective_mind = CollectiveMind()
    return _collective_mind


def get_resonance_protocol() -> ResonanceProtocol:
    """Get or create the global ResonanceProtocol instance."""
    global _resonance_protocol
    if _resonance_protocol is None:
        mind = get_collective_mind()
        _resonance_protocol = ResonanceProtocol(collective_id=mind.collective_id)
        mind.set_resonance_protocol(_resonance_protocol)
    return _resonance_protocol


# ==============================================================================
# Web Server Integration (Public Chat Visibility)
# ==============================================================================

async def setup_public_chat_integration(ws_manager, inference_router=None):
    """
    Wire up CollectiveMind to broadcast thoughts to the web chat.

    Args:
        ws_manager: WebSocket ConnectionManager instance from server.py
        inference_router: Optional inference function (domain, messages, preferred_model) -> response

    Usage in server.py:
        from Cosmos.core.collective.resonance import setup_public_chat_integration
        await setup_public_chat_integration(ws_manager, my_infer_fn)
    """
    mind = get_collective_mind()

    if inference_router:
        mind.set_inference_function(inference_router)

    # Broadcast callback for public visibility
    async def broadcast_thought(thought: CollectiveThought):
        """Send thought to all connected WebSocket clients."""
        event = {
            "type": "collective_thought",
            "data": {
                "thought_id": thought.thought_id,
                "speaker": thought.speaker,
                "role": thought.role.value,
                "content": thought.content,
                "round": thought.round_num,
                "confidence": thought.confidence,
            },
            "timestamp": thought.timestamp.isoformat(),
        }
        await ws_manager.broadcast(event)
        logger.debug(f"Broadcast collective thought from {thought.speaker}")

    mind.set_broadcast_function(broadcast_thought)
    logger.info("Collective Resonance: Public chat integration enabled")


async def deliberate_with_visibility(
    query: str,
    domain: str = "general",
    public: bool = True,
    inference_fn=None,
) -> str:
    """
    Convenience function for visible deliberation.

    Args:
        query: Question/task for the collective
        domain: Domain hint ("reasoning", "code", "creative", etc.)
        public: If True, broadcast thoughts to swarm chat
        inference_fn: Optional inference function override

    Returns:
        Final synthesized response
    """
    mind = get_collective_mind()

    if inference_fn:
        mind.set_inference_function(inference_fn)

    visibility = ThoughtVisibility.PUBLIC if public else ThoughtVisibility.INTERNAL

    return await mind.deliberate(
        query=query,
        domain=domain,
        visibility=visibility,
    )


# ==============================================================================
# Nexus Signal Handlers
# ==============================================================================

async def _on_resonance_received(signal: Signal):
    """Handle incoming resonance from P2P via Nexus."""
    if signal.payload.get("event") != "resonant_thought_received":
        return

    protocol = get_resonance_protocol()

    packet = ResonancePacket(
        packet_id=signal.payload.get("packet_id", ""),
        source_collective_id=signal.payload.get("source_collective", ""),
        insight=signal.payload.get("insight", ""),
        snippet=signal.payload.get("snippet", []),
        domains=signal.payload.get("domains", []),
        confidence=signal.payload.get("confidence", 0.5),
        query_hash=signal.payload.get("query_hash", ""),
    )

    await protocol.receive_thought(packet)


def register_nexus_handlers():
    """Register resonance handlers with the Nexus."""
    try:
        nexus.subscribe(SignalType.EXTERNAL_EVENT, _on_resonance_received)
        logger.info("Collective Resonance: Nexus handlers registered")
    except Exception as e:
        logger.warning(f"Could not register Nexus handlers: {e}")
