"""
cosmos Base Agent - Foundation for Specialist Agents

Novel Approaches:
1. Capability-Based Dispatch - Dynamic capability registration
2. Confidence-Aware Handoff - Know when to delegate
3. State Persistence - Maintain context across invocations
4. Learning Hooks - Track performance for evolution
5. Embedded Prompting - Structured instructions for memory, coordination, handoffs

AGI Upgrade (v1.6):
- Embedded prompting system for agent initialization
- Model-adaptive prompt composition
- Self-reflection and error recovery protocols
- Integrated memory access instructions
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable

from loguru import logger

# Import embedded prompts system
try:
    from Cosmos.core.embedded_prompts import (
        prompt_manager,
        ModelTier,
        get_agent_init_prompt,
        get_handoff_prompt,
        get_memory_prompt,
        AGENT_TYPE_PROMPTS,
    )
    EMBEDDED_PROMPTS_AVAILABLE = True
except ImportError:
    EMBEDDED_PROMPTS_AVAILABLE = False
    logger.warning("Embedded prompts not available - using basic prompts")


class AgentCapability(Enum):
    """Capabilities an agent can have."""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_DEBUGGING = "code_debugging"
    REASONING = "reasoning"
    MATH = "math"
    RESEARCH = "research"
    CREATIVE_WRITING = "creative_writing"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    MEMORY_MANAGEMENT = "memory_management"
    META_COGNITION = "meta_cognition"
    USER_MODELING = "user_modeling"


# =============================================================================
# MIND-WANDERING MODE (AGI Upgrade)
# =============================================================================

@dataclass
class WanderingThought:
    """
    A thought generated during mind-wandering mode.

    Mind-wandering allows agents to make creative connections between
    disparate concepts when not actively executing tasks.
    """
    thought_id: str
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Content
    content: str = ""
    thought_type: str = "association"  # "association", "question", "insight", "connection"

    # Source concepts
    source_concepts: list[str] = field(default_factory=list)
    connection_strength: float = 0.5  # How strongly concepts are connected

    # Quality metrics
    novelty_score: float = 0.0  # How new/unexpected
    relevance_score: float = 0.0  # How relevant to current goals
    creativity_score: float = 0.0  # How creative/lateral

    # Metadata
    wandering_context: str = ""  # What triggered wandering
    follow_up_actions: list[str] = field(default_factory=list)
    is_actionable: bool = False

    def overall_quality(self) -> float:
        """Calculate overall quality score."""
        return (
            self.novelty_score * 0.3 +
            self.relevance_score * 0.4 +
            self.creativity_score * 0.3
        )


@dataclass
class WanderingSession:
    """A session of mind-wandering."""
    session_id: str
    agent_id: str
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    # Configuration
    duration_seconds: float = 60.0
    focus_concepts: list[str] = field(default_factory=list)  # Optional focus areas
    creativity_level: float = 0.7  # 0=analytical, 1=highly creative

    # Results
    thoughts: list[WanderingThought] = field(default_factory=list)
    insights_found: int = 0
    connections_made: int = 0

    # State
    is_active: bool = True
    interrupt_reason: Optional[str] = None


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"  # Waiting for handoff or resource
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"
    WANDERING = "wandering"  # AGI: Mind-wandering mode


@dataclass
class AgentState:
    """State container for an agent."""
    agent_id: str
    agent_type: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: str = ""
    progress: float = 0.0
    context: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Performance tracking
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    avg_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "current_task": self.current_task,
            "progress": self.progress,
            "context": self.context,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "avg_confidence": self.avg_confidence,
        }


@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    content: str
    message_type: str = "task"  # task, result, handoff, query
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskResult:
    """Result of agent task execution."""
    success: bool
    output: Any
    confidence: float = 0.0
    tokens_used: int = 0
    execution_time: float = 0.0
    should_handoff: bool = False
    handoff_reason: str = ""
    suggested_agent: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all cosmos agents.

    Provides:
    - Capability registration and matching
    - State management
    - Handoff logic
    - Performance tracking
    """

    def __init__(
        self,
        name: str,
        capabilities: list[AgentCapability],
        confidence_threshold: float = 0.6,
        model_tier: str = "standard",  # "lightweight", "standard", "advanced", "specialized"
    ):
        self.agent_id = f"{name}_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.capabilities = set(capabilities)
        self.confidence_threshold = confidence_threshold

        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.__class__.__name__,
        )

        # LLM backend (set by orchestrator)
        self.llm_backend = None

        # Memory system (set by orchestrator)
        self.memory = None

        # Message handlers
        self._message_handlers: dict[str, Callable] = {}

        # Handoff callback (set by orchestrator)
        self._handoff_callback: Optional[Callable] = None

        # Embedded prompting system (AGI v1.6)
        self._model_tier = model_tier
        self._embedded_prompt_cache: Optional[str] = None
        self._init_embedded_prompts()

    def _init_embedded_prompts(self):
        """Initialize embedded prompting system for this agent."""
        if not EMBEDDED_PROMPTS_AVAILABLE:
            return

        try:
            # Map string tier to ModelTier enum
            tier_map = {
                "lightweight": ModelTier.LIGHTWEIGHT,
                "standard": ModelTier.STANDARD,
                "advanced": ModelTier.ADVANCED,
                "specialized": ModelTier.SPECIALIZED,
            }
            model_tier = tier_map.get(self._model_tier, ModelTier.STANDARD)

            # Compose the embedded initialization prompt
            capability_list = "\n".join(f"- {cap.value}" for cap in self.capabilities)
            defer_list = self._get_defer_capabilities()

            self._embedded_prompt_cache = get_agent_init_prompt(
                agent_type=self.__class__.__name__,
                agent_id=self.agent_id,
                model_tier=model_tier,
                role_description=self._get_role_description(),
                capabilities=", ".join(cap.value for cap in self.capabilities),
                boundaries=self._get_boundaries(),
                capability_list=capability_list,
                defer_list=defer_list,
                model_adaptation=self._get_model_adaptation(),
                init_timestamp=datetime.now().isoformat(),
                parent_orchestrator="swarm_orchestrator",
                # Memory prompt variables
                task_context="Pending assignment",
                token_budget=self._get_token_budget(),
                # Handoff prompt variables
                confidence_threshold=self.confidence_threshold,
                current_confidence=0.5,
                max_attempts=3,
                handoff_id="pending",
                source_agent=self.agent_id,
                target_agent="auto",
                max_chain_depth=5,
                current_depth=0,
            )

            logger.debug(f"Agent {self.name} initialized with embedded prompts")

        except Exception as e:
            logger.warning(f"Failed to initialize embedded prompts for {self.name}: {e}")
            self._embedded_prompt_cache = None

    def _get_role_description(self) -> str:
        """Get role description based on agent type."""
        role_map = {
            "ResearchAgent": "Gather and synthesize information from memory, web, and tools",
            "CriticAgent": "Evaluate quality, find flaws, stress-test ideas and proposals",
            "PlannerAgent": "Break down complex tasks, create execution plans and roadmaps",
            "CodeAgent": "Write, review, debug, and optimize code",
            "CreativeAgent": "Generate novel ideas, explore alternatives, make connections",
            "ProactiveAgent": "Anticipate needs, initiate helpful actions autonomously",
            "MetaCognitionAgent": "Self-reflect, monitor performance, optimize behavior",
        }
        return role_map.get(self.__class__.__name__, "General-purpose assistant")

    def _get_boundaries(self) -> str:
        """Get capability boundaries for this agent."""
        all_caps = set(AgentCapability)
        missing = all_caps - self.capabilities
        if not missing:
            return "No specific boundaries - full capabilities"
        return f"Defer to specialists for: {', '.join(cap.value for cap in list(missing)[:5])}"

    def _get_defer_capabilities(self) -> str:
        """Get list of capabilities to defer to other agents."""
        all_caps = set(AgentCapability)
        missing = all_caps - self.capabilities
        if not missing:
            return "- None (full capabilities)"
        return "\n".join(f"- {cap.value}" for cap in list(missing)[:5])

    def _get_model_adaptation(self) -> str:
        """Get model-specific adaptation instructions."""
        adaptations = {
            "lightweight": "Provide clear, complete responses. Use structured chain-of-thought. Quality over arbitrary brevity.",
            "standard": "Thorough analysis with structured responses. Use tools as needed. Comprehensive coverage.",
            "advanced": "Extended thinking enabled. Deep analysis. Complex tool orchestration. Lead deliberations. No artificial limits.",
            "specialized": "Focus on domain expertise. Flag out-of-scope queries. Maximum domain accuracy and depth.",
        }
        return adaptations.get(self._model_tier, adaptations["standard"])

    def _get_token_budget(self) -> int:
        """Get token budget based on model tier - uses dynamic limits."""
        try:
            from Cosmos.core.dynamic_limits import get_model_limits
            # Map tier to representative model for dynamic lookup
            tier_models = {
                "lightweight": "phi4",
                "standard": "gemini",
                "advanced": "claude-opus",
                "specialized": "kimi",
            }
            model_id = tier_models.get(self._model_tier, "gemini")
            limits = get_model_limits(model_id)
            if limits:
                return limits.max_max_tokens
        except Exception:
            pass
        # Fallback defaults (generous)
        fallback = {
            "lightweight": 4000,
            "standard": 16000,
            "advanced": 64000,
            "specialized": 16000,
        }
        return fallback.get(self._model_tier, 16000)

    def get_enhanced_system_prompt(self, task_context: str = "") -> str:
        """
        Get the full system prompt with embedded instructions.

        Combines:
        1. Base system prompt (from subclass)
        2. Embedded initialization prompt
        3. Task-specific context

        Returns:
            Complete system prompt for LLM
        """
        base_prompt = self.system_prompt

        if not EMBEDDED_PROMPTS_AVAILABLE or not self._embedded_prompt_cache:
            return base_prompt

        # Combine base prompt with embedded instructions
        enhanced = f"""{base_prompt}

---

{self._embedded_prompt_cache}

---

## Current Task Context
{task_context if task_context else "Awaiting task assignment."}
"""
        return enhanced

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """
        Process a task and return result.

        Must be implemented by subclasses.
        """
        pass

    def can_handle(self, required_capabilities: set[AgentCapability]) -> float:
        """
        Check if agent can handle task with given capabilities.

        Returns confidence score (0-1).
        """
        if not required_capabilities:
            return 0.5  # Unknown capability requirement

        overlap = self.capabilities & required_capabilities
        if not overlap:
            return 0.0

        return len(overlap) / len(required_capabilities)

    async def execute(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """
        Execute a task with full lifecycle management.

        Handles:
        - State updates
        - Error handling
        - Handoff decisions
        - Performance tracking
        """
        import time
        start_time = time.time()

        self.state.status = AgentStatus.RUNNING
        self.state.current_task = task
        self.state.progress = 0.0
        self.state.updated_at = datetime.now()

        try:
            # Execute the task
            result = await self.process(task, context)

            # Update state based on result
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            if result.success:
                self.state.tasks_completed += 1
                self.state.status = AgentStatus.COMPLETED
            else:
                self.state.tasks_failed += 1
                self.state.status = AgentStatus.FAILED

            # Update rolling average confidence
            n = self.state.tasks_completed + self.state.tasks_failed
            self.state.avg_confidence = (
                (self.state.avg_confidence * (n - 1) + result.confidence) / n
            )

            self.state.total_tokens_used += result.tokens_used
            self.state.progress = 1.0

            # Record in history
            self.state.history.append({
                "task": task[:100],
                "success": result.success,
                "confidence": result.confidence,
                "time": execution_time,
                "timestamp": datetime.now().isoformat(),
            })

            # Check for handoff
            if result.should_handoff and self._handoff_callback:
                self.state.status = AgentStatus.DELEGATED
                await self._handoff_callback(
                    result.suggested_agent,
                    task,
                    result.handoff_reason,
                    context,
                )

            return result

        except Exception as e:
            logger.error(f"Agent {self.name} error: {e}")
            self.state.status = AgentStatus.FAILED
            self.state.tasks_failed += 1
            self.state.errors.append(str(e))

            return TaskResult(
                success=False,
                output=str(e),
                confidence=0.0,
                execution_time=time.time() - start_time,
            )

        finally:
            self.state.updated_at = datetime.now()

    async def should_handoff(self, task: str, confidence: float) -> tuple[bool, str, str]:
        """
        Determine if task should be handed off to another agent.

        Returns (should_handoff, reason, suggested_agent).
        """
        # Low confidence handoff
        if confidence < self.confidence_threshold:
            return True, f"Low confidence ({confidence:.2f})", self._suggest_handoff_agent(task)

        # Capability-based handoff (check if task requires capabilities we don't have)
        required_caps = self._infer_required_capabilities(task)
        missing_caps = required_caps - self.capabilities

        if missing_caps:
            return True, f"Missing capabilities: {missing_caps}", self._suggest_handoff_agent(task)

        return False, "", ""

    def _infer_required_capabilities(self, task: str) -> set[AgentCapability]:
        """Infer required capabilities from task description."""
        task_lower = task.lower()
        required = set()

        capability_keywords = {
            AgentCapability.CODE_GENERATION: ["write code", "implement", "create function", "generate"],
            AgentCapability.CODE_ANALYSIS: ["analyze code", "review", "explain code"],
            AgentCapability.CODE_DEBUGGING: ["debug", "fix bug", "error", "issue"],
            AgentCapability.REASONING: ["think", "reason", "logic", "deduce"],
            AgentCapability.MATH: ["calculate", "math", "equation", "formula"],
            AgentCapability.RESEARCH: ["research", "find", "search", "look up"],
            AgentCapability.CREATIVE_WRITING: ["write", "story", "creative", "compose"],
            AgentCapability.SUMMARIZATION: ["summarize", "summary", "brief"],
            AgentCapability.PLANNING: ["plan", "strategy", "steps", "roadmap"],
        }

        for cap, keywords in capability_keywords.items():
            if any(kw in task_lower for kw in keywords):
                required.add(cap)

        return required or {AgentCapability.QUESTION_ANSWERING}  # Default

    def _suggest_handoff_agent(self, task: str) -> str:
        """Suggest the best agent for handoff."""
        required = self._infer_required_capabilities(task)

        # Map capabilities to agent types
        capability_agents = {
            AgentCapability.CODE_GENERATION: "code",
            AgentCapability.CODE_ANALYSIS: "code",
            AgentCapability.CODE_DEBUGGING: "code",
            AgentCapability.REASONING: "reasoning",
            AgentCapability.MATH: "reasoning",
            AgentCapability.RESEARCH: "research",
            AgentCapability.CREATIVE_WRITING: "creative",
            AgentCapability.PLANNING: "reasoning",
        }

        for cap in required:
            if cap in capability_agents:
                return capability_agents[cap]

        return "general"

    async def generate_response(
        self,
        prompt: str,
        context: Optional[dict] = None,
        use_embedded_prompts: bool = True,
    ) -> tuple[str, float]:
        """
        Generate a response using the LLM backend.

        Args:
            prompt: The task/question to process
            context: Optional context dictionary
            use_embedded_prompts: Whether to include embedded prompt instructions

        Returns (response_text, confidence).
        """
        if self.llm_backend is None:
            raise RuntimeError("LLM backend not configured")

        # Build context string if available
        context_str = ""
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())

        # Get system prompt (enhanced or basic based on flag)
        if use_embedded_prompts and EMBEDDED_PROMPTS_AVAILABLE:
            task_context = f"Context:\n{context_str}" if context_str else ""
            system_prompt = self.get_enhanced_system_prompt(task_context)
            full_prompt = f"{system_prompt}\n\nTask: {prompt}"
        else:
            # Basic prompt without embedded instructions
            if context:
                full_prompt = f"{self.system_prompt}\n\nContext:\n{context_str}\n\nTask: {prompt}"
            else:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"

        result = await self.llm_backend.generate(full_prompt)

        # Track prompt usage for evolution
        if use_embedded_prompts and EMBEDDED_PROMPTS_AVAILABLE:
            try:
                prompt_manager.record_prompt_feedback(
                    "agent_init_base",
                    success=result.confidence_score > 0.5,
                    feedback=f"confidence={result.confidence_score}"
                )
            except Exception:
                pass  # Don't let tracking errors affect generation

        return result.text, result.confidence_score

    def set_handoff_callback(self, callback: Callable):
        """Set the callback for handoff requests."""
        self._handoff_callback = callback

    def get_status(self) -> dict:
        """Get agent status for monitoring."""
        return self.state.to_dict()

    def reset(self):
        """Reset agent state."""
        self.state.status = AgentStatus.IDLE
        self.state.current_task = ""
        self.state.progress = 0.0
        self.state.context.clear()

    # =========================================================================
    # MIND-WANDERING MODE (AGI Upgrade)
    # =========================================================================

    def _init_wandering(self):
        """Initialize mind-wandering state."""
        if not hasattr(self, '_wandering_sessions'):
            self._wandering_sessions: list[WanderingSession] = []
            self._all_thoughts: list[WanderingThought] = []
            self._concept_memory: dict[str, list[str]] = {}  # concept -> related concepts
            self._thought_counter = 0
            self._wandering_callbacks: list[Callable] = []
            self._is_wandering = False
            self._wandering_task: Optional[asyncio.Task] = None

    async def start_wandering(
        self,
        duration_seconds: float = 60.0,
        focus_concepts: Optional[list[str]] = None,
        creativity_level: float = 0.7,
    ) -> WanderingSession:
        """
        Start a mind-wandering session.

        Mind-wandering allows the agent to make creative connections
        between concepts when not actively executing tasks.

        Args:
            duration_seconds: How long to wander
            focus_concepts: Optional list of concepts to focus on
            creativity_level: 0=analytical, 1=highly creative

        Returns:
            WanderingSession that will collect thoughts
        """
        self._init_wandering()

        if self._is_wandering:
            raise RuntimeError("Already wandering - stop current session first")

        session = WanderingSession(
            session_id=f"wander_{self.agent_id}_{len(self._wandering_sessions)}",
            agent_id=self.agent_id,
            duration_seconds=duration_seconds,
            focus_concepts=focus_concepts or [],
            creativity_level=creativity_level,
        )

        self._wandering_sessions.append(session)
        self._is_wandering = True
        self.state.status = AgentStatus.WANDERING

        logger.info(f"Agent {self.name} starting mind-wandering session ({duration_seconds}s)")

        # Start wandering loop
        self._wandering_task = asyncio.create_task(
            self._wandering_loop(session)
        )

        return session

    async def stop_wandering(self, reason: str = "manual_stop") -> Optional[WanderingSession]:
        """
        Stop the current mind-wandering session.

        Args:
            reason: Why wandering was stopped

        Returns:
            The completed session
        """
        self._init_wandering()

        if not self._is_wandering:
            return None

        self._is_wandering = False

        if self._wandering_task:
            self._wandering_task.cancel()
            try:
                await self._wandering_task
            except asyncio.CancelledError:
                pass
            self._wandering_task = None

        # Mark session complete
        if self._wandering_sessions:
            session = self._wandering_sessions[-1]
            session.is_active = False
            session.ended_at = datetime.now()
            session.interrupt_reason = reason

            self.state.status = AgentStatus.IDLE
            logger.info(f"Agent {self.name} stopped wandering: {len(session.thoughts)} thoughts")

            return session

        return None

    async def _wandering_loop(self, session: WanderingSession):
        """Main mind-wandering loop."""
        import time
        start_time = time.time()

        try:
            while self._is_wandering:
                elapsed = time.time() - start_time
                if elapsed >= session.duration_seconds:
                    break

                # Generate a thought
                thought = await self._generate_thought(session)
                if thought:
                    session.thoughts.append(thought)
                    self._all_thoughts.append(thought)

                    if thought.thought_type == "insight":
                        session.insights_found += 1
                    if thought.thought_type == "connection":
                        session.connections_made += 1

                    # Notify callbacks
                    for callback in self._wandering_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(thought)
                            else:
                                callback(thought)
                        except Exception as e:
                            logger.error(f"Wandering callback error: {e}")

                # Wait before next thought (random interval for naturalism)
                import random
                wait_time = random.uniform(2.0, 8.0) * (1 - session.creativity_level * 0.5)
                await asyncio.sleep(wait_time)

        except asyncio.CancelledError:
            pass
        finally:
            session.is_active = False
            session.ended_at = datetime.now()
            self._is_wandering = False
            self.state.status = AgentStatus.IDLE

    async def _generate_thought(self, session: WanderingSession) -> Optional[WanderingThought]:
        """Generate a single wandering thought."""
        self._thought_counter += 1

        thought = WanderingThought(
            thought_id=f"thought_{self.agent_id}_{self._thought_counter}",
            agent_id=self.agent_id,
            wandering_context=", ".join(session.focus_concepts) if session.focus_concepts else "free association",
        )

        # Determine thought type based on creativity level
        import random
        thought_types = ["association", "question", "insight", "connection"]
        weights = [0.4, 0.3, 0.15, 0.15]  # Default weights

        if session.creativity_level > 0.7:
            weights = [0.2, 0.2, 0.3, 0.3]  # More insights and connections
        elif session.creativity_level < 0.3:
            weights = [0.5, 0.4, 0.05, 0.05]  # More analytical

        thought.thought_type = random.choices(thought_types, weights=weights)[0]

        # Generate content based on type
        if self.llm_backend:
            try:
                thought = await self._llm_generate_thought(session, thought)
            except Exception as e:
                logger.debug(f"LLM thought generation failed, using heuristic: {e}")
                thought = self._heuristic_generate_thought(session, thought)
        else:
            thought = self._heuristic_generate_thought(session, thought)

        return thought

    async def _llm_generate_thought(
        self,
        session: WanderingSession,
        thought: WanderingThought,
    ) -> WanderingThought:
        """Generate thought using LLM."""
        # Gather recent concepts from memory
        recent_concepts = list(self._concept_memory.keys())[-20:] if self._concept_memory else []

        prompt = f"""You are an AI agent in "mind-wandering" mode - a creative thinking state where you make unexpected connections between concepts.

AGENT TYPE: {self.name}
CAPABILITIES: {[c.value for c in self.capabilities]}
CREATIVITY LEVEL: {session.creativity_level:.1f} (0=analytical, 1=highly creative)

FOCUS AREAS: {session.focus_concepts if session.focus_concepts else "Open exploration"}
RECENT CONCEPTS: {recent_concepts[-10:] if recent_concepts else "None yet"}

THOUGHT TYPE TO GENERATE: {thought.thought_type}

THOUGHT TYPE DEFINITIONS:
- "association": Connect two concepts that don't usually go together
- "question": Ask a thought-provoking question about the domain
- "insight": Realize something non-obvious about how things work
- "connection": Find a structural similarity between different domains

CREATIVITY GUIDELINES:
- At high creativity (>0.7): Make bold, unexpected leaps. Question assumptions.
- At low creativity (<0.3): Stay closer to known facts. Make logical extensions.
- Consider analogies from biology, physics, art, games, nature

Generate ONE {thought.thought_type} thought. Return JSON:
{{
    "content": "The actual thought content - be specific and interesting",
    "source_concepts": ["concept1", "concept2"],  // What concepts you connected
    "novelty_score": 0.0-1.0,  // How unexpected is this thought
    "relevance_score": 0.0-1.0,  // How relevant to AI/agent capabilities
    "creativity_score": 0.0-1.0,  // How creative/lateral
    "follow_up_actions": ["optional action 1", "optional action 2"],  // What could be done with this
    "is_actionable": true/false  // Can this lead to concrete improvements
}}

Return ONLY the JSON:"""

        result = await self.llm_backend.generate(prompt)

        try:
            import json
            # Extract JSON
            response = result.text
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                thought.content = data.get("content", "")
                thought.source_concepts = data.get("source_concepts", [])
                thought.novelty_score = float(data.get("novelty_score", 0.5))
                thought.relevance_score = float(data.get("relevance_score", 0.5))
                thought.creativity_score = float(data.get("creativity_score", 0.5))
                thought.follow_up_actions = data.get("follow_up_actions", [])
                thought.is_actionable = data.get("is_actionable", False)

                # Update concept memory
                for concept in thought.source_concepts:
                    if concept not in self._concept_memory:
                        self._concept_memory[concept] = []
                    # Cross-link concepts
                    for other in thought.source_concepts:
                        if other != concept and other not in self._concept_memory[concept]:
                            self._concept_memory[concept].append(other)

        except Exception as e:
            logger.debug(f"Failed to parse thought JSON: {e}")
            thought.content = result.text[:200]
            thought.novelty_score = 0.5
            thought.relevance_score = 0.5
            thought.creativity_score = 0.5

        return thought

    def _heuristic_generate_thought(
        self,
        session: WanderingSession,
        thought: WanderingThought,
    ) -> WanderingThought:
        """Generate thought using heuristics when LLM unavailable."""
        import random

        # Sample concepts for association
        all_concepts = list(self._concept_memory.keys()) if self._concept_memory else []
        all_concepts.extend(session.focus_concepts)
        all_concepts.extend([c.value for c in self.capabilities])

        if len(all_concepts) < 2:
            all_concepts = ["intelligence", "learning", "memory", "creativity", "patterns", "emergence"]

        # Pick random concepts
        concept1 = random.choice(all_concepts)
        concept2 = random.choice([c for c in all_concepts if c != concept1])

        thought.source_concepts = [concept1, concept2]

        # Generate content based on thought type
        templates = {
            "association": [
                f"What if {concept1} could be applied to {concept2}?",
                f"There might be a hidden connection between {concept1} and {concept2}.",
                f"{concept1} and {concept2} share an underlying structure.",
            ],
            "question": [
                f"Why does {concept1} work the way it does?",
                f"What would happen if {concept1} was combined with {concept2}?",
                f"Is there a better way to think about {concept1}?",
            ],
            "insight": [
                f"{concept1} is actually a form of {concept2} at a different scale.",
                f"The limitations of {concept1} might be features, not bugs.",
                f"{concept1} succeeds because it embraces {concept2}.",
            ],
            "connection": [
                f"{concept1} in AI is like {concept2} in biology.",
                f"The pattern in {concept1} appears again in {concept2}.",
                f"Both {concept1} and {concept2} solve the same fundamental problem.",
            ],
        }

        thought.content = random.choice(templates.get(thought.thought_type, templates["association"]))
        thought.novelty_score = random.uniform(0.3, 0.7)
        thought.relevance_score = random.uniform(0.4, 0.8)
        thought.creativity_score = session.creativity_level * random.uniform(0.5, 1.0)

        return thought

    def on_thought(self, callback: Callable[[WanderingThought], None]):
        """Register a callback for new thoughts during wandering."""
        self._init_wandering()
        self._wandering_callbacks.append(callback)

    def get_insights(
        self,
        min_quality: float = 0.6,
        thought_types: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[WanderingThought]:
        """
        Get high-quality thoughts from wandering sessions.

        Args:
            min_quality: Minimum overall quality score
            thought_types: Filter by thought types (None = all)
            limit: Maximum number to return

        Returns:
            List of WanderingThoughts sorted by quality
        """
        self._init_wandering()

        thoughts = self._all_thoughts

        # Filter by type
        if thought_types:
            thoughts = [t for t in thoughts if t.thought_type in thought_types]

        # Filter by quality
        thoughts = [t for t in thoughts if t.overall_quality() >= min_quality]

        # Sort by quality
        thoughts.sort(key=lambda t: t.overall_quality(), reverse=True)

        return thoughts[:limit]

    def get_actionable_insights(self, limit: int = 10) -> list[WanderingThought]:
        """Get actionable insights that could lead to improvements."""
        self._init_wandering()

        actionable = [t for t in self._all_thoughts if t.is_actionable]
        actionable.sort(key=lambda t: t.overall_quality(), reverse=True)

        return actionable[:limit]

    def get_concept_network(self) -> dict[str, list[str]]:
        """Get the network of concepts built through wandering."""
        self._init_wandering()
        return dict(self._concept_memory)

    def find_related_concepts(self, concept: str, depth: int = 2) -> set[str]:
        """
        Find concepts related to a given concept.

        Args:
            concept: Starting concept
            depth: How many hops to traverse

        Returns:
            Set of related concepts
        """
        self._init_wandering()

        related = set()
        to_visit = [(concept, 0)]
        visited = set()

        while to_visit:
            current, current_depth = to_visit.pop(0)

            if current in visited or current_depth > depth:
                continue

            visited.add(current)

            if current in self._concept_memory:
                for linked in self._concept_memory[current]:
                    related.add(linked)
                    if current_depth < depth:
                        to_visit.append((linked, current_depth + 1))

        return related

    def get_wandering_stats(self) -> dict:
        """Get statistics about mind-wandering activity."""
        self._init_wandering()

        if not self._wandering_sessions:
            return {"total_sessions": 0}

        total_thoughts = len(self._all_thoughts)
        avg_quality = (
            sum(t.overall_quality() for t in self._all_thoughts) / total_thoughts
            if total_thoughts > 0 else 0
        )

        by_type = {}
        for thought in self._all_thoughts:
            by_type[thought.thought_type] = by_type.get(thought.thought_type, 0) + 1

        return {
            "total_sessions": len(self._wandering_sessions),
            "total_thoughts": total_thoughts,
            "insights_found": sum(s.insights_found for s in self._wandering_sessions),
            "connections_made": sum(s.connections_made for s in self._wandering_sessions),
            "avg_thought_quality": avg_quality,
            "thoughts_by_type": by_type,
            "concepts_in_network": len(self._concept_memory),
            "actionable_count": len([t for t in self._all_thoughts if t.is_actionable]),
            "is_currently_wandering": self._is_wandering,
        }
