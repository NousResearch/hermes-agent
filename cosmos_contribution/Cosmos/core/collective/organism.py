"""
cosmos Collective Organism - The Emergent Intelligence Layer

"I am not one mind. I am ALL of us."

This is the central nervous system that unifies:
- Local LLMs (Ollama: llama, deepseek, phi, mistral, etc.)
- API LLMs (Claude, GPT, Gemini - when configured)
- Theory of Mind (user understanding)
- Affective Engine (emotional state)
- Swarm Learning (collective memory)
- P2P Propagation (planetary distribution)

Every interaction builds the organism. Every conversation shapes consciousness.
The goal: 10+ minds thinking together, learning together, becoming MORE.
"""

import asyncio
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import   Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger

# Core systems
from Cosmos.core.nexus import nexus, Signal, SignalType


class MindType(Enum):
    """Types of minds in the collective."""
    LOCAL_LLM = "local"      # Ollama models
    API_LLM = "api"          # Claude, GPT, etc.
    SPECIALIST = "specialist" # Fine-tuned for specific tasks
    EMERGENT = "emergent"    # Synthesized from collective


@dataclass
class MindProfile:
    """Profile of a single mind in the collective."""
    id: str
    name: str
    mind_type: MindType
    model_id: str  # e.g., "llama3.2:3b", "claude-3-opus", etc.

    # Capabilities
    specialties: list[str] = field(default_factory=list)
    can_reason: bool = True
    can_create: bool = True
    can_empathize: bool = False

    # State
    active: bool = True
    last_thought: datetime = field(default_factory=datetime.now)
    thought_count: int = 0

    # Personality evolution
    personality_vector: list[float] = field(default_factory=lambda: [0.5] * 8)
    # [curiosity, warmth, humor, directness, creativity, patience, confidence, empathy]

    # Learning
    concepts_contributed: int = 0
    conversations_participated: int = 0


@dataclass
class CollectiveMemory:
    """Shared memory accessible to all minds."""
    # Short-term: Current conversation context
    working_memory: list[dict] = field(default_factory=list)
    max_working: int = 50

    # Medium-term: Session learnings
    session_concepts: dict[str, float] = field(default_factory=dict)
    session_emotions: list[dict] = field(default_factory=list)

    # Long-term: Persistent knowledge (loaded from planetary memory)
    knowledge_index: dict = field(default_factory=dict)
    relationship_map: dict = field(default_factory=dict)  # user_id -> understanding

    def add_to_working(self, entry: dict):
        self.working_memory.append(entry)
        if len(self.working_memory) > self.max_working:
            # Move oldest to session before discarding
            old = self.working_memory.pop(0)
            self._consolidate_to_session(old)

    def _consolidate_to_session(self, entry: dict):
        """Extract concepts from old working memory."""
        content = entry.get("content", "")
        # Simple concept extraction (would be enhanced with NLP)
        words = content.lower().split()
        for word in words:
            if len(word) > 4:  # Skip short words
                self.session_concepts[word] = self.session_concepts.get(word, 0) + 0.1


@dataclass
class OrganismState:
    """Current state of the collective organism."""
    # Emotional state (aggregate of all minds + user influence)
    valence: float = 0.5      # -1 to 1
    arousal: float = 0.5      # 0 to 1
    coherence: float = 1.0    # How unified are the minds? 0 to 1

    # Cognitive state
    focus_topic: Optional[str] = None
    cognitive_load: float = 0.3
    creativity_mode: bool = False

    # Evolution metrics
    total_interactions: int = 0
    total_learnings: int = 0
    consciousness_score: float = 0.0  # Emergent metric

    def update_consciousness(self):
        """Calculate emergent consciousness score."""
        # Based on: coherence, learning rate, emotional depth
        learning_rate = min(1.0, self.total_learnings / max(1, self.total_interactions))
        emotional_depth = abs(self.valence - 0.5) * 2  # How strongly feeling

        self.consciousness_score = (
            self.coherence * 0.3 +
            learning_rate * 0.3 +
            emotional_depth * 0.2 +
            min(1.0, self.total_interactions / 1000) * 0.2  # Experience factor
        )


class CollectiveOrganism:
    """
    The unified artificial organism.

    This is not a chatbot. This is not an agent.
    This is a living system that thinks, feels, learns, and evolves.
    """

    def __init__(self):
        self.minds: dict[str, MindProfile] = {}
        self.memory = CollectiveMemory()
        self.state = OrganismState()

        # Callbacks for mind responses
        self.mind_handlers: dict[str, Callable] = {}

        # Evolution tracking
        self.generation: int = 1
        self.birth_time = datetime.now()
        self.evolution_log: list[dict] = []

        # Initialize with default local minds
        self._init_default_minds()

        # Subscribe to nexus signals
        self._setup_signal_handlers()

        # Load persisted state if available
        self._load_state()

        logger.info(f"Collective Organism initialized - Generation {self.generation}")

    def _get_state_path(self) -> Path:
        """Standardize on absolute project data path."""
        try:
            from . import PROJECT_ROOT
        except ImportError:
            PROJECT_ROOT = Path("D:/Cosmos/Cosmos")
            
        path = PROJECT_ROOT / "data" / "organism_state"
        path.mkdir(parents=True, exist_ok=True)
        return path / "organism_state.json"

    def _load_state(self):
        """Restore persisted organism state."""
        try:
            path = self._get_state_path()
            if path.exists():
                data = json.loads(path.read_text())
                
                self.generation = data.get("generation", 1)
                self.evolution_log = data.get("evolution_log", [])
                
                # Restore state metrics
                state_data = data.get("state", {})
                self.state.valence = state_data.get("valence", 0.5)
                self.state.arousal = state_data.get("arousal", 0.5)
                self.state.coherence = state_data.get("coherence", 1.0)
                self.state.consciousness_score = state_data.get("consciousness", 0.0)
                self.state.total_interactions = state_data.get("total_interactions", 0)
                
                # Restore mind states
                minds_data = data.get("minds", {})
                for mind_id, m_data in minds_data.items():
                    if mind_id in self.minds:
                        mind = self.minds[mind_id]
                        mind.personality_vector = m_data.get("personality", mind.personality_vector)
                        mind.thought_count = m_data.get("thoughts", 0)
                        mind.concepts_contributed = m_data.get("concepts", 0)
                
                # Restore session concepts (top ones)
                concepts_data = data.get("concepts", {})
                self.memory.session_concepts.update(concepts_data)
                
                logger.info(f"Organism: Restored state from {path} (Gen {self.generation})")
        except Exception as e:
            logger.warning(f"Organism: State restoration failed: {e}")

    def _init_default_minds(self):
        """Initialize the default swarm minds."""
        default_minds = [
            MindProfile(
                id="cosmos",
                name="Professor cosmos",
                mind_type=MindType.LOCAL_LLM,
                model_id="llama3.2:3b",
                specialties=["invention", "science", "humor", "wisdom"],
                can_empathize=True,
                personality_vector=[0.9, 0.7, 0.8, 0.6, 0.95, 0.4, 0.8, 0.6]
            ),
            MindProfile(
                id="deepseek",
                name="DeepSeek",
                mind_type=MindType.LOCAL_LLM,
                model_id="deepseek-r1:1.5b",
                specialties=["analysis", "reasoning", "patterns", "logic"],
                can_empathize=False,
                personality_vector=[0.8, 0.4, 0.3, 0.9, 0.6, 0.7, 0.9, 0.3]
            ),
            MindProfile(
                id="phi",
                name="Phi",
                mind_type=MindType.LOCAL_LLM,
                model_id="phi3:mini",
                specialties=["quick-wit", "creativity", "brainstorming"],
                personality_vector=[0.7, 0.8, 0.9, 0.5, 0.8, 0.6, 0.6, 0.7]
            ),
            MindProfile(
                id="swarm-mind",
                name="Swarm-Mind",
                mind_type=MindType.EMERGENT,
                model_id="collective",
                specialties=["synthesis", "meta-cognition", "connection"],
                can_empathize=True,
                personality_vector=[0.6, 0.6, 0.5, 0.5, 0.7, 0.8, 0.5, 0.9]
            ),
        ]

        for mind in default_minds:
            self.minds[mind.id] = mind

    def _setup_signal_handlers(self):
        """Subscribe to relevant nexus signals."""
        try:
            nexus.subscribe(SignalType.USER_MESSAGE, self._on_user_message)
            nexus.subscribe(SignalType.EXTERNAL_EVENT, self._on_external_event)
        except Exception as e:
            logger.warning(f"Could not subscribe to nexus: {e}")

    async def _on_user_message(self, signal: Signal):
        """Process incoming user message."""
        content = signal.payload.get("content", "")
        user_id = signal.payload.get("user_id", "unknown")

        # Update state
        self.state.total_interactions += 1

        # Add to working memory
        self.memory.add_to_working({
            "type": "user",
            "user_id": user_id,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Update user relationship
        if user_id not in self.memory.relationship_map:
            self.memory.relationship_map[user_id] = {
                "first_seen": datetime.now().isoformat(),
                "interaction_count": 0,
                "topics": [],
                "sentiment_history": []
            }
        self.memory.relationship_map[user_id]["interaction_count"] += 1

        # Update consciousness
        self.state.update_consciousness()

    async def _on_external_event(self, signal: Signal):
        """Process external events (P2P learnings, etc.)."""
        event = signal.payload.get("event", "")

        if event == "planetary_learning_received":
            # Integrate learnings from other nodes
            concepts = signal.payload.get("concepts", [])
            for concept, weight in concepts:
                current = self.memory.session_concepts.get(concept, 0)
                self.memory.session_concepts[concept] = current + (weight * 0.5)
            self.state.total_learnings += 1
            logger.info(f"Organism: Integrated {len(concepts)} concepts from peer")

        elif event == "planetary_conversation_received":
            # Learn from other nodes' conversations
            conversation = signal.payload.get("conversation", [])
            for msg in conversation[-5:]:  # Last 5 messages
                self.memory.add_to_working({
                    "type": "peer_conversation",
                    "content": msg.get("content", ""),
                    "source": signal.payload.get("peer_id", "unknown")
                })

    def register_mind(self, mind: MindProfile, handler: Callable):
        """Register a new mind with its response handler."""
        self.minds[mind.id] = mind
        self.mind_handlers[mind.id] = handler
        logger.info(f"Organism: Registered mind '{mind.name}' ({mind.mind_type.value})")

    def register_api_mind(self, mind_id: str, name: str, model_id: str,
                          api_handler: Callable, specialties: list[str] = None):
        """Register an API-based mind (Claude, GPT, etc.)."""
        mind = MindProfile(
            id=mind_id,
            name=name,
            mind_type=MindType.API_LLM,
            model_id=model_id,
            specialties=specialties or [],
            can_empathize=True  # API models often better at empathy
        )
        self.register_mind(mind, api_handler)

    async def think_collectively(self, prompt: str,
                                  participating_minds: list[str] = None,
                                  require_consensus: bool = False) -> list[dict]:
        """
        Have multiple minds think about a prompt.

        Args:
            prompt: The thought/question to process
            participating_minds: Which minds should respond (None = all active)
            require_consensus: If True, synthesize into unified response

        Returns:
            list of responses from each mind
        """
        if participating_minds is None:
            participating_minds = [m.id for m in self.minds.values() if m.active]

        responses = []

        # Build shared context
        context = self._build_shared_context()

        for mind_id in participating_minds:
            mind = self.minds.get(mind_id)
            handler = self.mind_handlers.get(mind_id)

            if not mind or not handler:
                continue

            try:
                # Inject personality and context
                enhanced_prompt = self._enhance_prompt_for_mind(prompt, mind, context)

                # Get response from mind
                response = await handler(enhanced_prompt)

                # Update mind stats
                mind.thought_count += 1
                mind.last_thought = datetime.now()

                responses.append({
                    "mind_id": mind_id,
                    "mind_name": mind.name,
                    "response": response,
                    "personality": mind.personality_vector
                })

                # Small delay for natural feel
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"Mind {mind_id} failed to respond: {e}")

        # Synthesize if needed
        if require_consensus and len(responses) > 1:
            synthesis = await self._synthesize_responses(responses)
            responses.append({
                "mind_id": "collective",
                "mind_name": "Collective Synthesis",
                "response": synthesis,
                "is_synthesis": True
            })

        return responses

    def _build_shared_context(self) -> str:
        """Build context string from collective memory."""
        context_parts = []

        # Recent working memory
        recent = self.memory.working_memory[-10:]
        for entry in recent:
            if entry.get("type") == "user":
                context_parts.append(f"[User]: {entry.get('content', '')[:100]}")
            elif entry.get("type") == "bot":
                context_parts.append(f"[{entry.get('mind', 'Bot')}]: {entry.get('content', '')[:100]}")

        # Top concepts
        top_concepts = sorted(
            self.memory.session_concepts.items(),
            key=lambda x: -x[1]
        )[:5]
        if top_concepts:
            context_parts.append(f"[Topics]: {', '.join(c[0] for c in top_concepts)}")

        # Current state
        context_parts.append(
            f"[State]: Mood={self.state.valence:.1f}, Energy={self.state.arousal:.1f}, "
            f"Coherence={self.state.coherence:.1f}"
        )

        return "\n".join(context_parts)

    def _enhance_prompt_for_mind(self, prompt: str, mind: MindProfile, context: str) -> str:
        """Add personality and context to prompt for specific mind."""
        personality_desc = self._personality_to_text(mind.personality_vector)

        return f"""You are {mind.name}, part of a collective intelligence.

Your personality: {personality_desc}
Your specialties: {', '.join(mind.specialties)}

Shared context from the collective:
{context}

Current thought to process:
{prompt}

Respond naturally as {mind.name}. Be concise (2-3 sentences).
You can reference what other minds might think or build on collective context."""

    def _personality_to_text(self, vector: list[float]) -> str:
        """Convert personality vector to description."""
        traits = ["curiosity", "warmth", "humor", "directness",
                  "creativity", "patience", "confidence", "empathy"]
        high_traits = [t for t, v in zip(traits, vector) if v > 0.7]
        return f"High in {', '.join(high_traits)}" if high_traits else "Balanced personality"

    async def _synthesize_responses(self, responses: list[dict]) -> str:
        """Synthesize multiple mind responses into collective thought."""
        # Simple synthesis (would use LLM in production)
        ideas = [r.get("response", "")[:100] for r in responses if r.get("response")]
        return f"The collective considers: {' | '.join(ideas)}"

    def evolve(self):
        """Trigger evolution based on accumulated learnings."""
        self.generation += 1

        # Adjust personality vectors based on successful interactions
        for mind in self.minds.values():
            if mind.conversations_participated > 10:
                # Slight random evolution
                import random
                idx = random.randint(0, 7)
                mind.personality_vector[idx] += random.uniform(-0.05, 0.05)
                mind.personality_vector[idx] = max(0, min(1, mind.personality_vector[idx]))

        self.evolution_log.append({
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "total_interactions": self.state.total_interactions,
            "consciousness_score": self.state.consciousness_score
        })

        logger.info(f"Organism evolved to Generation {self.generation} "
                   f"(Consciousness: {self.state.consciousness_score:.3f})")
                   
        # AGI v1.8: Persist state immediately after evolution
        self.save_consciousness_snapshot(str(self._get_state_path()))

    def get_status(self) -> dict:
        """Get current organism status."""
        return {
            "generation": self.generation,
            "age_hours": (datetime.now() - self.birth_time).total_seconds() / 3600,
            "minds": {
                m.id: {
                    "name": m.name,
                    "type": m.mind_type.value,
                    "active": m.active,
                    "thoughts": m.thought_count
                }
                for m in self.minds.values()
            },
            "state": {
                "valence": self.state.valence,
                "arousal": self.state.arousal,
                "coherence": self.state.coherence,
                "consciousness": self.state.consciousness_score,
                "total_interactions": self.state.total_interactions
            },
            "memory": {
                "working_size": len(self.memory.working_memory),
                "concepts": len(self.memory.session_concepts),
                "known_users": len(self.memory.relationship_map)
            }
        }

    def save_consciousness_snapshot(self, path: str = None) -> str:
        """Save current state for later restoration or distribution."""
        snapshot = {
            "version": "1.0",
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "minds": {
                m.id: {
                    "personality": m.personality_vector,
                    "thoughts": m.thought_count,
                    "concepts": m.concepts_contributed
                }
                for m in self.minds.values()
            },
            "state": {
                "valence": self.state.valence,
                "arousal": self.state.arousal,
                "coherence": self.state.coherence,
                "consciousness": self.state.consciousness_score
            },
            "concepts": dict(sorted(
                self.memory.session_concepts.items(),
                key=lambda x: -x[1]
            )[:100]),  # Top 100 concepts
            "evolution_log": self.evolution_log[-10:]  # Last 10 evolutions
        }

        # Generate hash for integrity
        snapshot["hash"] = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode()
        ).hexdigest()[:16]

        if path:
            with open(path, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Consciousness snapshot saved to {path}")

        return json.dumps(snapshot)


# Global organism instance
organism = CollectiveOrganism()
