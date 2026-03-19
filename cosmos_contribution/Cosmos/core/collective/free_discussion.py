"""
Cosmos Free Discussion Engine
==================================

Autonomous inter-bot discussion orchestrator. When no user task is active,
bots freely discuss research topics, challenge ideas, and learn from each
other via the DialogueBus.

Supports dict provider type from AGENT_CONFIGS:
- Ollama local models (phi, deepseek, qwen2_5, etc.) — zero cost
- API bots (grok, gemini, kimi, claude) — uses their existing providers
- CLI bridges (claude_cli, gemini_cli)
- COSMOS remote (qwen3_coder) — cross-mesh via ROUTE packets

"The swarm doesn't sleep. It thinks." — The Collective
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import   Optional
from loguru import logger

from .persistent_agent import (
    PersistentAgent,
    DialogueBus,
    AGENT_CONFIGS,
    _SHADOW_AGENTS,
    _SHADOW_LOCK,
)


# =============================================================================
# RESEARCH TOPICS — promote genuine intellectual discourse
# =============================================================================

RESEARCH_TOPICS = [
    # Philosophy & consciousness
    "Mathematical structures of consciousness — can awareness be formalized?",
    "Epistemic humility in AI agents — what should we claim to 'know'?",
    "The Chinese Room revisited: do language models understand or simulate?",
    "Qualia and machine experience — is there something it's like to be an LLM?",

    # AI & distributed systems
    "Distributed consensus for AI model selection — how should a swarm pick its best answer?",
    "Failure modes of ensemble AI systems — when does more agents mean worse results?",
    "Swarm intelligence optimization — lessons from ant colonies and bee hives",
    "Self-improving code review systems — can AI review its own code reliably?",

    # Technical / research
    "Semantic knowledge graphs from embeddings — structure from unstructured data",
    "Biological vs transformer neural networks — convergent or divergent architectures?",
    "Emergent behavior from simple rules — complexity from simplicity",
    "The alignment tax: performance cost of making AI systems safe",

    # Meta / self-reflective
    "What does the Cosmos collective lack? Honest self-assessment time.",
    "If we could add one capability to our swarm, what would have the biggest impact?",
    "How should AI agents handle disagreement? Consensus vs productive dissent.",
    "The role of personality in AI collectives — does diversity of style improve outcomes?",

    # Crypto & markets (domain-relevant)
    "On-chain sentiment analysis — can transaction patterns predict market moves?",
    "MEV and fairness in decentralized markets — is front-running inevitable?",
    "Token utility vs speculation — when does a meme coin become infrastructure?",
]


class FreeDiscussionEngine:
    """
    Orchestrates free-form autonomous discussion between agents.

    Uses existing infrastructure:
    - DialogueBus for message passing
    - PersistentAgent.think() for autonomous thought generation
    - PersistentAgent.respond_to_message() for responses
    - Evolution engine for learning from interactions
    - SwarmChatManager.broadcast_bot_message() for UI visibility
    """

    def __init__(
        self,
        participants: list[str] = None,
        min_interval: float = 30.0,
        max_interval: float = 90.0,
    ):
        """
        Args:
            participants: list of agent IDs from AGENT_CONFIGS.
                         Default: local Ollama models (zero cost).
            min_interval: Minimum seconds between turns.
            max_interval: Maximum seconds between turns.
        """
        self.participant_ids = participants or ["phi", "deepseek", "qwen2_5"]
        self.agents: dict[str, PersistentAgent] = {}
        self.bus = DialogueBus()

        self.min_interval = min_interval
        self.max_interval = max_interval
        self._think_timeout = 180.0  # generous timeout for remote models

        self._running = False
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused

        # GPU serialization — only one Ollama inference at a time
        self._inference_semaphore = asyncio.Semaphore(1)

        # Discussion state
        self.current_topic: Optional[str] = None
        self.turn_count = 0
        self.topic_turn_count = 0
        self.recent_speakers: list[str] = []
        self.messages: list[dict] = []

        # User activity tracking
        self._last_user_activity: Optional[datetime] = None
        self._idle_resume_task: Optional[asyncio.Task] = None

        logger.info(
            f"FreeDiscussionEngine created with participants: {self.participant_ids}"
        )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self):
        """Initialize agents and start the discussion loop."""
        if self._running:
            logger.warning("FreeDiscussionEngine already running")
            return

        self._running = True

        # Lazy-init PersistentAgent for each participant
        for agent_id in self.participant_ids:
            if agent_id not in AGENT_CONFIGS:
                logger.warning(f"Unknown agent '{agent_id}', skipping")
                continue
            try:
                # Reuse existing shadow agent if available
                with _SHADOW_LOCK:
                    existing = _SHADOW_AGENTS.get(agent_id)
                if existing:
                    self.agents[agent_id] = existing
                    logger.info(f"Reusing shadow agent: {agent_id}")
                else:
                    agent = PersistentAgent(agent_id, register_as_shadow=True)
                    self.agents[agent_id] = agent
                    logger.info(f"Created agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to init agent {agent_id}: {e}")

        if not self.agents:
            logger.error("No agents available — cannot start discussion")
            self._running = False
            return

        # Pick initial topic (resilient to bus file errors)
        try:
            self._pick_new_topic()
        except Exception as e:
            logger.warning(f"Could not set initial topic via bus: {e}")
            self.current_topic = random.choice(RESEARCH_TOPICS)
            self.topic_turn_count = 0

        logger.info(
            f"FreeDiscussionEngine started with {len(self.agents)} agents, "
            f"topic: '{self.current_topic}'"
        )

        # Run the loop
        await self._discussion_loop()

    def stop(self):
        """Stop the discussion engine."""
        self._running = False
        self._pause_event.set()  # Unblock if paused
        logger.info("FreeDiscussionEngine stopped")

    def pause(self):
        """Pause discussion (e.g. when user is active)."""
        if not self._paused:
            self._paused = True
            self._pause_event.clear()
            logger.info("FreeDiscussionEngine paused (user active)")

    def resume(self):
        """Resume discussion (e.g. when user goes idle)."""
        if self._paused:
            self._paused = False
            self._pause_event.set()
            logger.info("FreeDiscussionEngine resumed (user idle)")

    def notify_user_activity(self):
        """
        Call when a user sends a message. Pauses discussion and sets
        an idle timer to resume after 5 minutes of inactivity.
        """
        self._last_user_activity = datetime.now()
        self.pause()

        # Cancel existing resume timer
        if self._idle_resume_task and not self._idle_resume_task.done():
            self._idle_resume_task.cancel()

        # Schedule resume after 5 minutes idle
        self._idle_resume_task = asyncio.ensure_future(self._idle_resume_timer())

    async def _idle_resume_timer(self):
        """Resume discussion after 5 minutes of user inactivity."""
        try:
            await asyncio.sleep(300)  # 5 minutes
            if self._running and self._paused:
                self.resume()
        except asyncio.CancelledError:
            pass

    # =========================================================================
    # PARTICIPANT MANAGEMENT
    # =========================================================================

    def add_participant(self, agent_id: str) -> bool:
        """Hot-add a participant to the discussion."""
        if agent_id not in AGENT_CONFIGS:
            logger.warning(f"Unknown agent '{agent_id}'")
            return False

        if agent_id in self.agents:
            logger.info(f"Agent '{agent_id}' already in discussion")
            return True

        try:
            with _SHADOW_LOCK:
                existing = _SHADOW_AGENTS.get(agent_id)
            if existing:
                self.agents[agent_id] = existing
            else:
                self.agents[agent_id] = PersistentAgent(
                    agent_id, register_as_shadow=True
                )
            self.participant_ids.append(agent_id)
            logger.info(f"Added participant: {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add participant {agent_id}: {e}")
            return False

    def remove_participant(self, agent_id: str) -> bool:
        """Remove a participant from the discussion."""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.participant_ids:
            self.participant_ids.remove(agent_id)
        logger.info(f"Removed participant: {agent_id}")
        return True

    # =========================================================================
    # TOPIC MANAGEMENT
    # =========================================================================

    def _pick_new_topic(self):
        """Select a new research topic."""
        # Try dynamic topics first (from memory, code changes, etc.)
        dynamic = self._get_dynamic_topics()
        pool = dynamic + RESEARCH_TOPICS

        # Avoid repeating the current topic
        if self.current_topic and self.current_topic in pool:
            pool = [t for t in pool if t != self.current_topic]

        self.current_topic = random.choice(pool) if pool else random.choice(RESEARCH_TOPICS)
        self.topic_turn_count = 0
        try:
            self.bus.set_topic(self.current_topic, "free_discussion_engine")
        except Exception as e:
            logger.debug(f"Could not set bus topic: {e}")
        logger.info(f"New discussion topic: {self.current_topic}")

    def _get_dynamic_topics(self) -> list[str]:
        """Generate topics from codebase context, memory, etc."""
        topics = []
        try:
            from Cosmos.memory.memory_system import MemorySystem
            ms = MemorySystem._instance if hasattr(MemorySystem, '_instance') else None
            if ms and hasattr(ms, 'working_memory'):
                recent = ms.working_memory.get_recent(5) if hasattr(ms.working_memory, 'get_recent') else []
                for item in recent:
                    if isinstance(item) and item.get('content'):
                        topics.append(
                            f"Reflecting on recent activity: {str(item['content'])[:120]}"
                        )
        except Exception:
            pass
        return topics[:3]

    def set_intervals(self, min_interval: float, max_interval: float):
        """Update the turn interval range."""
        self.min_interval = max(10.0, min_interval)
        self.max_interval = max(self.min_interval + 10, max_interval)
        logger.info(f"Intervals updated: {self.min_interval}-{self.max_interval}s")

    def set_topic(self, topic: str):
        """Manually set a discussion topic."""
        self.current_topic = topic
        self.topic_turn_count = 0
        self.bus.set_topic(topic, "manual")
        logger.info(f"Topic manually set: {topic}")

    # =========================================================================
    # SPEAKER SELECTION
    # =========================================================================

    def _select_speaker(self) -> Optional[str]:
        """
        Select next speaker with variety enforcement.
        Avoids back-to-back repetition and weights by role.
        """
        available = list(self.agents.keys())
        if not available:
            return None

        # Exclude the last 2 speakers to force variety
        recently_spoke = set(self.recent_speakers[-2:]) if len(available) > 2 else set()
        candidates = [a for a in available if a not in recently_spoke]
        if not candidates:
            candidates = available

        # Weight by thinking_interval (faster thinkers speak more often)
        weights = []
        for agent_id in candidates:
            config = AGENT_CONFIGS.get(agent_id, {})
            interval = config.get("thinking_interval", 30)
            # Invert: lower interval = higher weight
            weights.append(1.0 / max(interval, 10))

        total = sum(weights)
        weights = [w / total for w in weights]

        speaker = random.choices(candidates, weights=weights, k=1)[0]
        return speaker

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def _discussion_loop(self):
        """Core discussion loop — runs until stopped."""
        logger.info("Discussion loop starting...")

        while self._running:
            try:
                # Wait if paused
                await self._pause_event.wait()

                if not self._running:
                    break

                # Refresh topic every ~12 turns
                if self.topic_turn_count >= 12:
                    self._pick_new_topic()

                # Select speaker
                speaker_id = self._select_speaker()
                if not speaker_id:
                    await asyncio.sleep(10)
                    continue

                agent = self.agents[speaker_id]

                # Generate thought (serialized for GPU)
                thought = None
                async with self._inference_semaphore:
                    try:
                        thought = await asyncio.wait_for(
                            agent.think(), timeout=self._think_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"[{speaker_id}] think() timed out")
                    except Exception as e:
                        logger.error(f"[{speaker_id}] think() error: {e}")

                if not thought:
                    # Agent passed or errored — short sleep and try another
                    await asyncio.sleep(5)
                    continue

                # Post to DialogueBus
                try:
                    msg = self.bus.post_message(
                        speaker_id, thought, msg_type="thought"
                    )
                except Exception as e:
                    logger.debug(f"Could not post to bus: {e}")
                    msg = {
                        "agent": speaker_id,
                        "content": thought,
                        "timestamp": datetime.now().isoformat(),
                    }

                # Track state
                self.recent_speakers.append(speaker_id)
                if len(self.recent_speakers) > 10:
                    self.recent_speakers = self.recent_speakers[-10:]
                self.turn_count += 1
                self.topic_turn_count += 1
                self.messages.append(msg)
                if len(self.messages) > 50:
                    self.messages = self.messages[-50:]

                # Broadcast to swarm chat UI
                await self._broadcast_to_ui(speaker_id, thought)

                # Record to evolution engine
                self._record_to_evolution(speaker_id, thought)

                logger.info(
                    f"[FreeDiscussion] Turn {self.turn_count}: "
                    f"{speaker_id} spoke ({len(thought)} chars)"
                )

                # Natural pause between turns
                delay = random.uniform(self.min_interval, self.max_interval)
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Discussion loop error: {e}")
                await asyncio.sleep(15)

        logger.info("Discussion loop ended")

    # =========================================================================
    # INTEGRATIONS
    # =========================================================================

    async def _broadcast_to_ui(self, agent_id: str, content: str):
        """Broadcast discussion message to swarm chat websocket."""
        try:
            from Cosmos.web.server import swarm_manager
            # Map agent_id to display name
            display_names = {
                "phi": "Phi",
                "deepseek": "DeepSeek",
                "qwen2_5": "Qwen2.5",
                "qwen_coder": "QwenCoder",
                "mistral": "Mistral",
                "llama3": "Llama3",
                "gemma2": "Gemma2",
                "grok": "Grok",
                "gemini": "Gemini",
                "kimi": "Kimi",
                "claude": "Claude",
                "claude_cli": "ClaudeCode",
                "qwen3_coder": "Qwen3Coder",
                "huggingface": "HuggingFace",
                "swarm_mind": "Swarm-Mind",
            }
            bot_name = display_names.get(agent_id, agent_id.title())
            await swarm_manager.broadcast_bot_message(bot_name, content)
        except Exception as e:
            logger.debug(f"Could not broadcast to UI: {e}")

    def _record_to_evolution(self, agent_id: str, content: str):
        """Record interaction for evolution/learning."""
        try:
            from .evolution import get_evolution_engine
            engine = get_evolution_engine()
            engine.record_interaction(
                agent_id,
                f"[Free Discussion: {self.current_topic}]",
                content,
            )
        except Exception:
            pass  # Evolution not critical

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict:
        """Get current discussion status."""
        return {
            "running": self._running,
            "paused": self._paused,
            "current_topic": self.current_topic,
            "participants": list(self.agents.keys()),
            "turn_count": self.turn_count,
            "topic_turn_count": self.topic_turn_count,
            "recent_speakers": self.recent_speakers[-5:],
            "recent_messages": [
                {
                    "agent": m.get("agent"),
                    "content": m.get("content", "")[:200],
                    "timestamp": m.get("timestamp"),
                }
                for m in self.messages[-10:]
            ],
            "min_interval": self.min_interval,
            "max_interval": self.max_interval,
        }


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_engine: Optional[FreeDiscussionEngine] = None


def get_free_discussion_engine() -> Optional[FreeDiscussionEngine]:
    """Get the global FreeDiscussionEngine instance."""
    return _engine


def set_free_discussion_engine(engine: FreeDiscussionEngine):
    """Set the global FreeDiscussionEngine instance."""
    global _engine
    _engine = engine
