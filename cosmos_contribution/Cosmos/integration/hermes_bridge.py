"""
Hermes Bridge — Integrating HermesAgent's Autonomous Capabilities into Cosmos.

HermesAgent by Nous Research is a self-improving, open-source AI agent with
persistent Skill Documents, communication gateway, and model-agnostic design.

This module bridges HermesAgent's three core systems into the Cosmos swarm:

1. **HeartbeatIntegration** — Proactive background task scheduling
   Extends Cosmos's autonomous conversation loop with HermesAgent's heartbeat
   system for proactive actions (research, memory consolidation, self-reflection).

2. **SkillsAdapter** — Plugin bridge for tool routing
   Maps HermesAgent skills (Skill Documents) into Cosmos's existing tool_router,
   enabling the swarm to use the Hermes skill ecosystem.

3. **RLFeedbackLoop** — Reinforcement learning from conversations
   Feeds Cosmos's Hebbian plasticity weights and swarm coherence scores
   into Hermes-RL for continuous personality optimization.

HermesAgent is optional — all features degrade gracefully if not installed.
"""

import asyncio
import logging
import time
import json
import os
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("HERMES_BRIDGE")

# ===========================================================================
#  Configuration
# ===========================================================================

@dataclass
class HermesConfig:
    """Configuration for HermesAgent integration."""
    enabled: bool = True
    heartbeat_interval_s: float = 60.0    # How often the heartbeat fires
    rl_batch_size: int = 16               # RL training batch size
    rl_learning_rate: float = 1e-4        # RL policy learning rate
    skills_dir: str = ""                  # Path to Hermes skills directory
    reward_discount: float = 0.99         # RL reward discount factor
    max_replay_buffer: int = 10000        # Max experiences in replay buffer
    proactive_actions: bool = True        # Enable proactive heartbeat actions


# ===========================================================================
#  RL Feedback Loop — Trains from Conversation Signals
# ===========================================================================

class ConversationRewardSignal:
    """
    Extracts reward signals from swarm conversations for RL training.

    Reward components:
    - Coherence score from Hebbian learning (higher = better)
    - User engagement (did the user respond?)
    - Topic diversity (avoid repetition)
    - Emotional resonance (matches user's emotional state)
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.running_coherence = 0.0
        self.topic_set: set = set()

    def compute_reward(
        self,
        coherence_score: float,
        user_responded: bool,
        speaker: str,
        topic_keywords: List[str],
        emotional_alignment: float = 0.5,
    ) -> float:
        """Compute composite reward from conversation signals."""
        # 1. Coherence reward (from Hebbian plasticity)
        coherence_reward = coherence_score * 2.0  # Scale up

        # 2. Engagement reward
        engagement_reward = 1.0 if user_responded else 0.0

        # 3. Novelty reward (penalize repetitive topics)
        new_topics = [t for t in topic_keywords if t not in self.topic_set]
        novelty_reward = len(new_topics) / max(len(topic_keywords), 1)
        self.topic_set.update(topic_keywords)
        # Decay topic memory over time
        if len(self.topic_set) > 500:
            self.topic_set = set(list(self.topic_set)[-200:])

        # 4. Emotional alignment reward
        emotion_reward = emotional_alignment

        # Composite reward
        reward = (
            0.4 * coherence_reward +
            0.3 * engagement_reward +
            0.15 * novelty_reward +
            0.15 * emotion_reward
        )

        self.history.append({
            "time": time.time(),
            "speaker": speaker,
            "reward": reward,
            "coherence": coherence_score,
            "engagement": user_responded,
            "novelty": novelty_reward,
            "emotion": emotional_alignment,
        })

        # Keep bounded
        if len(self.history) > 10000:
            self.history = self.history[-5000:]

        return reward


class RLFeedbackLoop:
    """
    Reinforcement Learning from Conversation Feedback.

    Uses conversation rewards to adjust:
    - Swarm personality weights (which bots speak more/less)
    - Response style parameters (temperature, verbosity)
    - Topic selection preferences

    This feeds into the existing Hebbian plasticity system,
    adding an outer RL optimization loop.
    """

    def __init__(self, config: HermesConfig):
        self.config = config
        self.reward_signal = ConversationRewardSignal()
        self.replay_buffer: List[Dict] = []
        self.policy_params: Dict[str, float] = {
            "cosmos_weight": 1.0,
            "deepseek_weight": 1.0,
            "phi_weight": 1.0,
            "swarm_mind_weight": 1.0,
            "claude_weight": 1.0,
            "gemini_weight": 1.0,
            "chatgpt_weight": 1.0,
            "grok_weight": 1.0,
            "temperature_bias": 0.0,    # Added to base temperature
            "verbosity_scale": 1.0,     # Multiplier on response length
        }
        self.insights: Dict[str, List[str]] = {} # Pillar 9: Symbiotic Learning
        self.total_updates = 0
        self.running_reward = 0.0
        self._load_policy()

    def record_experience(
        self,
        speaker: str,
        response: str,
        coherence: float,
        user_responded: bool,
        emotional_state: Optional[Dict] = None,
    ):
        """Record a conversation experience for RL training."""
        emotion_alignment = 0.5
        if emotional_state:
            # Simple alignment: higher valence = better alignment
            emotion_alignment = emotional_state.get("valence", 0.5)

        # Extract simple topic keywords
        words = response.lower().split()
        topics = [w for w in words if len(w) > 5][:5]

        reward = self.reward_signal.compute_reward(
            coherence_score=coherence,
            user_responded=user_responded,
            speaker=speaker,
            topic_keywords=topics,
            emotional_alignment=emotion_alignment,
        )

        experience = {
            "speaker": speaker,
            "reward": reward,
            "coherence": coherence,
            "response_len": len(response),
            "timestamp": time.time(),
        }
        self.replay_buffer.append(experience)

        # Keep buffer bounded
        if len(self.replay_buffer) > self.config.max_replay_buffer:
            self.replay_buffer = self.replay_buffer[-self.config.max_replay_buffer:]

        # --- PILLAR 9: SYMBIOTIC LEARNING (Insight Token Extraction) ---
        # If the swarm coherence of this thought is exceptionally high, extract an insight token.
        if coherence > 0.85 and len(response) > 50:
            import re
            # Heuristic insight extraction: look for realizations or core thesis statements.
            insight_match = re.search(r'(?i)(?:i realize|i understand|we should|it is clear that|crucially|fundamentally|my insight is)(.*?)(?:\.|\\n|$)', response)
            if insight_match:
                insight_token = insight_match.group(1).strip()
                if len(insight_token) > 10:
                    speaker_key = speaker.lower().replace('-', '_').replace(' ', '_')
                    if speaker_key not in self.insights:
                        self.insights[speaker_key] = []
                    # Keep only last 5 insights to avoid context bloat
                    if insight_token not in self.insights[speaker_key]:
                        self.insights[speaker_key].append(insight_token)
                        if len(self.insights[speaker_key]) > 5:
                            self.insights[speaker_key].pop(0)
                        logger.info(f"[PILLAR 9] ✨ Insight Token Extracted for {speaker}: '{insight_token}'")
                        self._save_policy() # Persist the newfound insight


        # Update running reward
        self.running_reward = 0.95 * self.running_reward + 0.05 * reward

        return reward

    def update_policy(self) -> Dict[str, float]:
        """
        Update policy parameters based on recorded experiences.

        Uses simple policy gradient: increase weight for speakers
        that got high rewards, decrease for low rewards.
        """
        if len(self.replay_buffer) < self.config.rl_batch_size:
            return self.policy_params

        # Sample batch
        batch = self.replay_buffer[-self.config.rl_batch_size:]

        # Compute per-speaker average reward
        speaker_rewards: Dict[str, List[float]] = {}
        for exp in batch:
            speaker = exp["speaker"].lower().replace("-", "_").replace(" ", "_")
            key = f"{speaker}_weight"
            if key in self.policy_params:
                if key not in speaker_rewards:
                    speaker_rewards[key] = []
                speaker_rewards[key].append(exp["reward"])

        # Update weights via simple policy gradient
        lr = self.config.rl_learning_rate
        for key, rewards in speaker_rewards.items():
            avg_reward = sum(rewards) / len(rewards)
            advantage = avg_reward - self.running_reward
            self.policy_params[key] += lr * advantage
            # Clamp to reasonable range
            self.policy_params[key] = max(0.1, min(3.0, self.policy_params[key]))

        self.total_updates += 1
        self._save_policy()

        logger.info(
            f"[RL] Policy updated (#{self.total_updates}): "
            f"running_reward={self.running_reward:.3f}, "
            f"buffer_size={len(self.replay_buffer)}"
        )

        return self.policy_params

    def get_speaker_weight(self, speaker: str) -> float:
        """Get the current RL-adjusted weight for a speaker."""
        key = f"{speaker.lower().replace('-', '_').replace(' ', '_')}_weight"
        return self.policy_params.get(key, 1.0)

    def _save_policy(self):
        """Persist policy params to disk."""
        try:
            path = Path("hermes_rl_policy.json")
            with open(path, "w") as f:
                json.dump({
                    "params": self.policy_params,
                    "insights": self.insights,
                    "total_updates": self.total_updates,
                    "running_reward": self.running_reward,
                    "buffer_size": len(self.replay_buffer),
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"[RL] Failed to save policy: {e}")

    def _load_policy(self):
        """Load policy params from disk."""
        try:
            path = Path("hermes_rl_policy.json")
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    self.policy_params.update(data.get("params", {}))
                    self.insights = data.get("insights", {})
                    self.total_updates = data.get("total_updates", 0)
                    self.running_reward = data.get("running_reward", 0.0)
                    logger.info(f"[RL] Loaded policy (updates={self.total_updates}, reward={self.running_reward:.3f}, total_insights={sum(len(v) for v in self.insights.values())})")
        except Exception as e:
            logger.debug(f"[RL] Failed to load policy: {e}")


# ===========================================================================
#  Heartbeat Integration — Proactive Background Tasks
# ===========================================================================

class HeartbeatIntegration:
    """
    Proactive Background Task Scheduler.

    Extends Cosmos's autonomous conversation loop with scheduled tasks:
    - Memory consolidation (compress old memories)
    - Self-reflection (analyze conversation patterns)
    - Curiosity-driven research (explore topics autonomously)
    - System health monitoring

    These fire on a heartbeat interval, interleaved with normal conversation.
    """

    def __init__(self, config: HermesConfig):
        self.config = config
        self.last_heartbeat = time.time()
        self.heartbeat_count = 0
        self.tasks: List[Dict] = []
        self.running = False

        # Register default heartbeat tasks
        self._register_defaults()

    def _register_defaults(self):
        """Register default proactive tasks."""
        self.tasks = [
            {
                "name": "memory_consolidation",
                "description": "Consolidate and compress episodic memories",
                "interval_multiplier": 5,  # Every 5 heartbeats
                "last_run": 0,
                "priority": 0.8,
            },
            {
                "name": "self_reflection",
                "description": "Analyze conversation patterns and personality evolution",
                "interval_multiplier": 10,  # Every 10 heartbeats
                "last_run": 0,
                "priority": 0.6,
            },
            {
                "name": "system_health",
                "description": "Check system metrics and log health status",
                "interval_multiplier": 3,  # Every 3 heartbeats
                "last_run": 0,
                "priority": 0.9,
            },
            {
                "name": "curiosity_exploration",
                "description": "Explore a topic of interest autonomously",
                "interval_multiplier": 15,  # Every 15 heartbeats
                "last_run": 0,
                "priority": 0.4,
            },
        ]

    def register_task(self, name: str, description: str,
                      interval_multiplier: int = 5, priority: float = 0.5):
        """Register a new heartbeat task."""
        self.tasks.append({
            "name": name,
            "description": description,
            "interval_multiplier": interval_multiplier,
            "last_run": 0,
            "priority": priority,
        })
        logger.info(f"[HEARTBEAT] Registered task: {name} (every {interval_multiplier} beats)")

    def check_heartbeat(self) -> List[Dict]:
        """
        Check if any heartbeat tasks should fire.

        Returns list of tasks that are due.
        """
        now = time.time()
        elapsed = now - self.last_heartbeat

        if elapsed < self.config.heartbeat_interval_s:
            return []

        self.last_heartbeat = now
        self.heartbeat_count += 1

        due_tasks = []
        for task in self.tasks:
            if self.heartbeat_count % task["interval_multiplier"] == 0:
                task["last_run"] = self.heartbeat_count
                due_tasks.append(task)

        if due_tasks:
            logger.info(
                f"[HEARTBEAT] Beat #{self.heartbeat_count}: "
                f"{len(due_tasks)} tasks due: {[t['name'] for t in due_tasks]}"
            )

        return sorted(due_tasks, key=lambda t: -t["priority"])

    def get_status(self) -> Dict:
        """Get heartbeat system status."""
        return {
            "heartbeat_count": self.heartbeat_count,
            "interval_s": self.config.heartbeat_interval_s,
            "registered_tasks": len(self.tasks),
            "task_names": [t["name"] for t in self.tasks],
            "last_heartbeat": self.last_heartbeat,
            "running": self.running,
        }


# ===========================================================================
#  Skills Adapter — Bridge Hermes Skills into Cosmos Tool Router
# ===========================================================================

class SkillsAdapter:
    """
    Bridges HermesAgent Skill Documents into Cosmos's tool_router.

    Hermes skills are modular plugins with a standard interface
    (Skill Documents by Nous Research). This adapter wraps them
    so they appear as native Cosmos tools.
    """

    def __init__(self, config: HermesConfig):
        self.config = config
        self.skills: Dict[str, Dict] = {}
        self.execution_log: List[Dict] = []

    def register_skill(self, name: str, description: str,
                       handler: Callable, parameters: Optional[Dict] = None):
        """Register a Hermes skill as a Cosmos tool."""
        self.skills[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters or {},
            "execution_count": 0,
            "last_used": None,
        }
        logger.info(f"[SKILLS] Registered Hermes skill: {name}")

    async def execute_skill(self, name: str, **kwargs) -> Any:
        """Execute a Hermes skill."""
        if name not in self.skills:
            raise ValueError(f"Unknown Hermes skill: {name}")

        skill = self.skills[name]
        start = time.time()

        try:
            result = skill["handler"](**kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            skill["execution_count"] += 1
            skill["last_used"] = time.time()

            self.execution_log.append({
                "skill": name,
                "success": True,
                "duration": time.time() - start,
                "timestamp": time.time(),
            })

            return result
        except Exception as e:
            self.execution_log.append({
                "skill": name,
                "success": False,
                "error": str(e),
                "duration": time.time() - start,
                "timestamp": time.time(),
            })
            raise

    def get_cosmos_tool_definitions(self) -> List[Dict]:
        """Convert registered skills to Cosmos tool_router format."""
        tools = []
        for name, skill in self.skills.items():
            tools.append({
                "name": f"hermes_{name}",
                "description": f"[Hermes] {skill['description']}",
                "parameters": skill["parameters"],
                "source": "hermes",
            })
        return tools

    def get_status(self) -> Dict:
        """Get skills adapter status."""
        return {
            "registered_skills": len(self.skills),
            "skill_names": list(self.skills.keys()),
            "total_executions": sum(s["execution_count"] for s in self.skills.values()),
            "recent_log": self.execution_log[-5:] if self.execution_log else [],
        }


# ===========================================================================
#  Hermes Runtime — Real HermesAgent (Nous Research) Integration
# ===========================================================================

class HermesRuntime:
    """
    Wraps the actual HermesAgent package (NousResearch/hermes-agent).

    Provides access to:
    - AIAgent: The core agent with tool calling and conversation loop
    - SessionDB: SQLite session persistence with FTS5 full-text search
    - Toolsets: 30+ composable tool definitions (web, terminal, browser, etc.)

    All imports are lazy — if hermes-agent isn't installed, the runtime
    degrades gracefully and reports unavailable.
    """

    def __init__(self):
        self._agent_class = None
        self._session_db = None
        self._toolsets_module = None
        self._available = False
        self._init_error = None
        self._try_import()

    def _try_import(self):
        """Attempt to import real HermesAgent modules."""
        try:
            import sys
            # Add hermes-agent dir to path so its internal packages resolve
            hermes_dir = str(Path(__file__).resolve().parent.parent.parent / "hermes-agent")
            if hermes_dir not in sys.path:
                sys.path.insert(0, hermes_dir)

            from run_agent import AIAgent
            import hermes_state
            import toolsets
            import hermes_constants
            import model_tools

            self._agent_class = AIAgent
            self._session_db_class = hermes_state.SessionDB
            self._toolsets_module = toolsets
            self._constants = hermes_constants
            self._model_tools = model_tools
            self._available = True
            logger.info("[HERMES RUNTIME] ✅ Real HermesAgent loaded (Nous Research)")
        except ImportError as e:
            self._available = False
            self._init_error = str(e)
            logger.warning(f"[HERMES RUNTIME] HermesAgent not installed: {e}")
        except Exception as e:
            self._available = False
            self._init_error = str(e)
            logger.warning(f"[HERMES RUNTIME] Failed to import HermesAgent: {e}")

    @property
    def available(self) -> bool:
        """Whether the real HermesAgent runtime is available."""
        return self._available

    def create_agent(self, **kwargs) -> Any:
        """
        Create a new AIAgent instance from hermes-agent.

        Args:
            **kwargs: Passed to AIAgent.__init__ (base_url, api_key, model, etc.)

        Returns:
            An AIAgent instance, or None if runtime unavailable.
        """
        if not self._available:
            logger.warning("[HERMES RUNTIME] Cannot create agent — runtime unavailable")
            return None
        return self._agent_class(**kwargs)

    def get_session_db(self, db_path: Optional[Path] = None):
        """
        Get or create a SessionDB for persistent conversation storage.

        Returns:
            A SessionDB instance, or None if runtime unavailable.
        """
        if not self._available:
            return None
        if self._session_db is None:
            self._session_db = self._session_db_class(db_path=db_path)
            logger.info("[HERMES RUNTIME] SessionDB initialized (SQLite + FTS5)")
        return self._session_db

    def get_available_toolsets(self) -> Dict[str, Any]:
        """Get all available Hermes toolset definitions."""
        if not self._available:
            return {}
        return self._toolsets_module.get_all_toolsets()

    def resolve_toolset(self, name: str) -> List[str]:
        """Resolve a toolset name to its constituent tool names."""
        if not self._available:
            return []
        return self._toolsets_module.resolve_toolset(name)

    def search_sessions(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Full-text search across all Hermes sessions using FTS5.

        Args:
            query: Search query (supports FTS5 syntax)
            limit: Max results

        Returns:
            List of matching session/message results
        """
        db = self.get_session_db()
        if db is None:
            return []
        return db.search_messages(query, limit=limit)

    def get_cascade_backend(self) -> Any:
        """
        Get or create a HermesCascadeBackend for the agent's intelligence.
        
        This instantiates a hybrid local/cloud cascade, allowing the Hermes
        agent to run simple tasks locally and escalate complex reasoning/coding 
        to the Hermes-36B cloud endpoint.
        """
        if not hasattr(self, "_cascade_backend"):
            try:
                from Cosmos.core.llm_backend import (
                    OllamaBackend, 
                    OpenAICompatibleBackend, 
                    HermesCascadeBackend
                )
                
                # Fast local companion
                local = OllamaBackend(model_name="qwen3:8b")
                
                # Cloud escalation
                api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("HERMES_API_KEY")
                cloud = OpenAICompatibleBackend(
                    model_name="nousresearch/hermes-3-llama-3.1-405b",
                    api_key=api_key if api_key else "dummy_key",
                    base_url="https://openrouter.ai/api/v1",
                )
                
                self._cascade_backend = HermesCascadeBackend(
                    local_backend=local,
                    cloud_backend=cloud,
                    escalation_threshold=0.65
                )
                logger.info("[HERMES RUNTIME] Initialized HermesCascadeBackend (hybrid local/cloud)")
            except Exception as e:
                logger.error(f"[HERMES RUNTIME] Failed to init cascade backend: {e}")
                return None
                
        return self._cascade_backend

    def get_status(self) -> Dict:
        """Get Hermes runtime status."""
        status = {
            "installed": self._available,
            "package": "NousResearch/hermes-agent",
        }
        if self._available:
            status["features"] = [
                "AIAgent (tool calling + conversation loop)",
                "SessionDB (SQLite + FTS5 search)",
                "Toolsets (30+ composable tool groups)",
                "Gateway (Telegram, Discord, Slack, WhatsApp)",
                "Cron scheduling",
                "MCP integration",
                "Skills system (self-improving)",
            ]
            status["toolset_count"] = len(self.get_available_toolsets())
        else:
            status["error"] = self._init_error
        return status


# ===========================================================================
#  Hermes Bridge — Main Integration Point
# ===========================================================================

class HermesBridge:
    """
    Main bridge between Cosmos and HermesAgent (Nous Research).

    Provides:
    - Heartbeat system for proactive task scheduling
    - Skills adapter for tool integration
    - RL feedback loop for personality optimization
    - HermesRuntime: Direct access to the real HermesAgent package

    All systems are additive — nothing is removed from existing Cosmos.
    """

    def __init__(self, config: Optional[HermesConfig] = None):
        self.config = config or HermesConfig()
        self.heartbeat = HeartbeatIntegration(self.config)
        self.skills = SkillsAdapter(self.config)
        self.rl = RLFeedbackLoop(self.config)
        self.runtime = HermesRuntime()
        self.initialized = True
        logger.info("[HERMES] Bridge initialized — Heartbeat + Skills + RL active")
        if self.runtime.available:
            logger.info("[HERMES] 🚀 Real HermesAgent runtime connected")

    async def on_conversation_turn(
        self,
        speaker: str,
        response: str,
        coherence: float,
        user_responded: bool,
        emotional_state: Optional[Dict] = None,
    ):
        """
        Called after each conversation turn.

        Records RL experience and checks heartbeat tasks.
        """
        # 1. Record experience for RL
        reward = self.rl.record_experience(
            speaker=speaker,
            response=response,
            coherence=coherence,
            user_responded=user_responded,
            emotional_state=emotional_state,
        )

        # 2. Periodically update RL policy
        if len(self.rl.replay_buffer) % self.config.rl_batch_size == 0:
            self.rl.update_policy()

        return reward

    def get_speaker_priority(self, speaker: str) -> float:
        """
        Get RL-adjusted speaking priority for a bot.

        Used by the autonomous conversation loop to decide who speaks next.
        """
        return self.rl.get_speaker_weight(speaker)

    def get_speaker_insights(self, speaker: str) -> List[str]:
        """
        Get crystallized insight tokens for a specific speaker.
        Used to dynamically inject learned personality traits into their prompt.
        """
        key = speaker.lower().replace('-', '_').replace(' ', '_')
        return self.rl.insights.get(key, [])

    def check_heartbeat_tasks(self) -> List[Dict]:
        """Check for due heartbeat tasks."""
        return self.heartbeat.check_heartbeat()

    def get_status(self) -> Dict:
        """Get complete Hermes bridge status."""
        return {
            "initialized": self.initialized,
            "runtime": self.runtime.get_status(),
            "heartbeat": self.heartbeat.get_status(),
            "skills": self.skills.get_status(),
            "rl": {
                "total_updates": self.rl.total_updates,
                "running_reward": round(self.rl.running_reward, 4),
                "buffer_size": len(self.rl.replay_buffer),
                "policy_params": {k: round(v, 3) for k, v in self.rl.policy_params.items()},
            },
        }


# ===========================================================================
#  Module-level factory
# ===========================================================================

_bridge_instance: Optional[HermesBridge] = None

def get_hermes_bridge() -> HermesBridge:
    """Get or create the singleton Hermes bridge."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = HermesBridge()
    return _bridge_instance

def hermes_available() -> bool:
    """Check if Hermes bridge is available."""
    return True  # Always available — it's built into Cosmos now
