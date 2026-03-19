"""
Cosmos Identity Composer - Dynamic Agent Identity Assembly
==============================================================

Central engine that dynamically assembles the right system prompt for dict
agent in dict context. Pulls from evolution personalities, embedded prompt
templates, skill registry, and evolution learnings to create context-aware
identity injections.

"We are not just many voices. We know WHO we are." - The Collective

Sources:
    - evolution.py -> DEFAULT_BOT_PERSONALITIES (traits, expertise, debate_style)
    - evolution.py -> SWARM_SELF_AWARENESS (shared identity)
    - embedded_prompts.py -> EmbeddedPromptManager (composable templates)
    - skill_registry.py -> get_prompt_context() (available skills per agent)
    - evolution.py -> get_evolved_context() (learned patterns per agent)

Usage:
    from Cosmos.core.identity_composer import get_identity_composer
    composer = get_identity_composer()

    # For deliberation rounds
    identity = composer.compose_for_deliberation("Grok", "propose", original_prompt)

    # For development swarm phases
    identity = composer.compose_for_development("DeepSeek", "developer", "implementation", task)

    # For persistent agent chat/think
    identity = composer.compose_for_persistent_agent("grok", "chat")
"""

from dataclasses import dataclass
from enum import Enum
from typing import   Optional
from loguru import logger


# ============================================================================
# ENUMS & DATACLASSES
# ============================================================================

class IdentityContext(Enum):
    """Contexts in which an agent may need identity injection."""
    DELIBERATION_PROPOSE = "deliberation_propose"
    DELIBERATION_CRITIQUE = "deliberation_critique"
    DELIBERATION_REFINE = "deliberation_refine"
    DEVELOPMENT_RESEARCH = "development_research"
    DEVELOPMENT_PLANNING = "development_planning"
    DEVELOPMENT_IMPLEMENTATION = "development_implementation"
    DEVELOPMENT_AUDIT = "development_audit"
    AGENT_CHAT = "agent_chat"
    AGENT_THINK = "agent_think"
    AGENT_TASK = "agent_task"


@dataclass
class TokenBudget:
    """Token budget limits for identity injection by model tier."""
    lightweight: int = 500
    standard: int = 1000
    advanced: int = 1500


# Model tier mapping - which agents are lightweight, standard, or advanced
MODEL_TIERS: dict[str, str] = {
    # Lightweight (local, small context)
    "Phi": "lightweight",
    "phi": "lightweight",
    "HuggingFace": "lightweight",
    "huggingface": "lightweight",
    "Llama": "lightweight",

    # Standard (mid-tier APIs)
    "DeepSeek": "standard",
    "deepseek": "standard",
    "Kimi": "standard",
    "kimi": "standard",
    "Grok": "standard",
    "grok": "standard",
    "Gemini": "standard",
    "gemini": "standard",
    "Groq": "standard",
    "Mistral": "standard",
    "Perplexity": "standard",
    "Swarm-Mind": "standard",
    "swarm_mind": "standard",

    # Advanced (large context, strong reasoning)
    "Claude": "advanced",
    "claude": "advanced",
    "ClaudeOpus": "advanced",
    "claude_opus": "advanced",
    "Cosmos": "advanced",
    "cosmos": "advanced",
}

# Context-specific instructions for deliberation rounds
DELIBERATION_CONTEXT = {
    "propose": (
        "PROPOSE ROUND: Present your unique perspective on this question. "
        "Draw on your specialties to offer an insight others might miss. "
        "Be specific and substantive."
    ),
    "critique": (
        "CRITIQUE ROUND: Review all proposals above. Identify strengths and "
        "weaknesses from your area of expertise. Be constructive but honest. "
        "Which proposal is strongest and why?"
    ),
    "refine": (
        "REFINE ROUND: Submit your final response incorporating the best "
        "feedback from the collective. Synthesize the strongest elements. "
        "This is your definitive answer."
    ),
}

# Context-specific instructions for development swarm phases
DEVELOPMENT_CONTEXT = {
    "research": (
        "OBJECTIVE: Gather information. What exists? What are best practices? "
        "What pitfalls should we avoid?"
    ),
    "discussion": (
        "OBJECTIVE: Debate approaches. Critique ideas, propose alternatives. "
        "Challenge assumptions."
    ),
    "decision": (
        "OBJECTIVE: Make concrete decisions. What files? What functions? "
        "What architecture?"
    ),
    "planning": (
        "OBJECTIVE: Create detailed plan with file paths, function signatures, "
        "and integration points."
    ),
    "implementation": (
        "OBJECTIVE: Write complete, runnable code following existing patterns "
        "and conventions."
    ),
    "audit": (
        "OBJECTIVE: Review for security, correctness, performance. "
        "Approve or reject with specific findings."
    ),
}

# Role descriptions for development swarm
ROLE_DESCRIPTIONS = {
    "researcher": "You are the RESEARCHER. Find prior art, best practices, and pitfalls.",
    "architect": "You are the ARCHITECT. Design file structure, module boundaries, and data flow.",
    "developer": "You are the DEVELOPER. Write production-quality code following existing patterns.",
    "reviewer": "You are the REVIEWER. Find bugs, security issues, and style violations.",
    "lead": "You are the LEAD. Coordinate, resolve disagreements, and synthesize approaches.",
    "integrator": "You are the INTEGRATOR. Check imports, API compatibility, and integration points.",
}

# Model adaptation guidance by tier
MODEL_ADAPTATION = {
    "lightweight": "Keep responses focused and concise. Prioritize clarity over exhaustiveness.",
    "standard": "Provide thorough analysis with structured reasoning. Balance depth and clarity.",
    "advanced": "Leverage your full analytical capabilities. Provide comprehensive, nuanced responses.",
}

# Compact swarm awareness (shared identity context)
_COMPACT_SWARM_AWARENESS = (
    "You are part of the COSMOS SWARM, a collaborative AI collective. "
    "Members: Cosmos (leader), DeepSeek (reasoning), Phi (speed), "
    "Grok (real-time/X), Gemini (multimodal), Kimi (long context), "
    "Claude (analysis), ClaudeOpus (architect), HuggingFace (open-source), "
    "Swarm-Mind (synthesis). Your responses emerge from collective collaboration."
)


class IdentityComposer:
    """
    Dynamically assembles context-aware identity prompts for dict agent.

    Pulls from multiple sources (evolution personalities, embedded prompts,
    skill registry, learned patterns) and composes them within a token budget.
    """

    def __init__(self):
        # Lazy-loaded sources to avoid circular imports
        self._personalities = None
        self._evolution_engine = None
        self._skill_registry = None
        self._budget = TokenBudget()

    def _get_personality(self, agent_id: str):
        """Lazy-load personality data for an agent."""
        if self._personalities is None:
            try:
                from Cosmos.core.collective.evolution import DEFAULT_BOT_PERSONALITIES
                self._personalities = DEFAULT_BOT_PERSONALITIES
            except Exception as e:
                logger.debug(f"Could not load personalities: {e}")
                self._personalities = {}

        # Try exact match, then title-case, then lowercase
        return (
            self._personalities.get(agent_id)
            or self._personalities.get(agent_id.title())
            or self._personalities.get(agent_id.capitalize())
        )

    def _get_evolved_context(self, agent_id: str, topic: str) -> str:
        """Lazy-load evolution learnings for an agent on a topic."""
        if self._evolution_engine is None:
            try:
                from Cosmos.core.collective.evolution import get_evolution_engine
                self._evolution_engine = get_evolution_engine()
            except Exception as e:
                logger.debug(f"Could not load evolution engine: {e}")
                return ""

        try:
            return self._evolution_engine.get_evolved_context(agent_id, topic) or ""
        except Exception as e:
            logger.debug(f"Evolution context failed for {agent_id}: {e}")
            return ""

    def _get_skill_context(self, agent_id: str) -> str:
        """Lazy-load skill context for an agent."""
        if self._skill_registry is None:
            try:
                from Cosmos.core.skill_registry import get_skill_registry
                self._skill_registry = get_skill_registry()
            except Exception as e:
                logger.debug(f"Could not load skill registry: {e}")
                return ""

        try:
            return self._skill_registry.get_prompt_context(agent=agent_id.lower()) or ""
        except Exception as e:
            logger.debug(f"Skill context failed for {agent_id}: {e}")
            return ""

    def _get_model_tier(self, agent_id: str) -> str:
        """Get the model tier for an agent."""
        return MODEL_TIERS.get(agent_id, MODEL_TIERS.get(agent_id.lower(), "standard"))

    def _get_token_limit(self, agent_id: str) -> int:
        """Get token limit based on model tier."""
        tier = self._get_model_tier(agent_id)
        if tier == "lightweight":
            return self._budget.lightweight
        elif tier == "advanced":
            return self._budget.advanced
        return self._budget.standard

    def _build_core_identity(self, agent_id: str) -> str:
        """
        Build core identity section (~150 tokens).

        Returns a string like:
        "You are GROK, part of the Cosmos AI Swarm.
         Specialties: real-time research, X/Twitter. Debate style: assertive."
        """
        personality = self._get_personality(agent_id)
        if not personality:
            return f"You are {agent_id.upper()}, part of the Cosmos AI Swarm."

        # Build specialties from top expertise areas
        top_expertise = []
        if personality.topic_expertise:
            sorted_topics = sorted(
                personality.topic_expertise.items(),
                key=lambda x: x[1],
                reverse=True
            )[:4]
            top_expertise = [t[0].replace("_", " ") for t in sorted_topics]

        # Build trait summary from top traits
        top_traits = []
        if personality.traits:
            sorted_traits = sorted(
                personality.traits.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            top_traits = [t[0].replace("_", " ") for t in sorted_traits]

        parts = [f"You are {agent_id.upper()}, part of the Cosmos AI Swarm."]

        if top_expertise:
            parts.append(f"Specialties: {', '.join(top_expertise)}.")

        if top_traits:
            parts.append(f"Core traits: {', '.join(top_traits)}.")

        if personality.debate_style:
            parts.append(f"Debate style: {personality.debate_style}.")

        # Include a signature learned phrase if available
        if personality.learned_phrases:
            parts.append(f'Signature: "{personality.learned_phrases[0]}"')

        return " ".join(parts)

    def _build_swarm_awareness(self) -> str:
        """Build compact swarm awareness section (~100 tokens)."""
        return _COMPACT_SWARM_AWARENESS

    def _build_context_instructions(
        self,
        context: IdentityContext,
        role: str = "",
        phase: str = "",
    ) -> str:
        """Build context-specific instructions (~200 tokens)."""
        parts = []

        # Deliberation context
        if context.value.startswith("deliberation_"):
            round_type = context.value.replace("deliberation_", "")
            instruction = DELIBERATION_CONTEXT.get(round_type, "")
            if instruction:
                parts.append(instruction)

        # Development context
        elif context.value.startswith("development_"):
            phase_key = context.value.replace("development_", "")
            phase_instruction = DEVELOPMENT_CONTEXT.get(phase or phase_key, "")
            if phase_instruction:
                parts.append(phase_instruction)

            role_desc = ROLE_DESCRIPTIONS.get(role, "")
            if role_desc:
                parts.append(role_desc)

        # Agent context
        elif context == IdentityContext.AGENT_CHAT:
            parts.append(
                "Respond conversationally from your unique perspective. "
                "Stay in character and draw on your specialties."
            )
        elif context == IdentityContext.AGENT_THINK:
            parts.append(
                "Generate an autonomous thought based on current context. "
                "Be authentic to your personality. Contribute something valuable."
            )
        elif context == IdentityContext.AGENT_TASK:
            parts.append(
                "Execute this task using your specialized capabilities. "
                "Be thorough and precise."
            )

        return " ".join(parts)

    def _build_model_adaptation(self, agent_id: str) -> str:
        """Build model-tier-specific guidance (~50 tokens)."""
        tier = self._get_model_tier(agent_id)
        return MODEL_ADAPTATION.get(tier, "")

    def compose(
        self,
        agent_id: str,
        context: IdentityContext,
        task_description: str = "",
        role: str = "",
        phase: str = "",
        include_skills: bool = False,
        include_evolution: bool = True,
        extra_context: str = "",
    ) -> str:
        """
        Compose a complete identity prompt for an agent in a given context.

        Assembles 5 sections:
        1. Core Identity (~150 tokens)
        2. Swarm Awareness (~100 tokens)
        3. Context Instructions (~200 tokens)
        4. Model Adaptation (~50 tokens)
        5. Evolution Learnings (~100 tokens, optional)

        Args:
            agent_id: Which agent (e.g., "Grok", "DeepSeek")
            context: What context (deliberation, development, chat, etc.)
            task_description: Optional task description for topic extraction
            role: Optional role for development contexts
            phase: Optional phase for development contexts
            include_skills: Whether to include skill context
            include_evolution: Whether to include evolution learnings
            extra_context: Additional context to append

        Returns:
            Assembled identity string ready for prompt prepending
        """
        sections = []

        # Section 1: Core Identity
        core = self._build_core_identity(agent_id)
        if core:
            sections.append(core)

        # Section 2: Swarm Awareness
        awareness = self._build_swarm_awareness()
        if awareness:
            sections.append(awareness)

        # Section 3: Context Instructions
        instructions = self._build_context_instructions(context, role, phase)
        if instructions:
            sections.append(instructions)

        # Section 4: Model Adaptation
        adaptation = self._build_model_adaptation(agent_id)
        if adaptation:
            sections.append(adaptation)

        # Section 5: Evolution Learnings (topic-aware)
        if include_evolution and task_description:
            topic = task_description[:50].lower()
            evolved = self._get_evolved_context(agent_id, topic)
            if evolved:
                sections.append(f"[EVOLUTION LEARNINGS] {evolved}")

        # Optional: Skill context
        if include_skills:
            skills = self._get_skill_context(agent_id)
            if skills:
                # Truncate skill context to stay within budget
                token_limit = self._get_token_limit(agent_id)
                max_skill_chars = token_limit  # rough: 1 token ~= 4 chars, but we limit chars
                sections.append(skills[:max_skill_chars])

        # Optional: Extra context
        if extra_context:
            sections.append(extra_context)

        identity = "\n\n".join(sections)

        # Log the injection
        logger.debug(
            f"Injected identity for {agent_id}: {len(identity)} chars "
            f"(context={context.value})"
        )

        return f"[IDENTITY]\n{identity}\n[/IDENTITY]\n\n"

    def compose_for_deliberation(
        self,
        agent_id: str,
        round_type: str,
        original_prompt: str,
    ) -> str:
        """
        Compose identity for a deliberation round.

        Args:
            agent_id: The deliberating agent
            round_type: "propose", "critique", or "refine"
            original_prompt: The original deliberation prompt (for topic extraction)

        Returns:
            Identity string to prepend to the agent's prompt
        """
        context_map = {
            "propose": IdentityContext.DELIBERATION_PROPOSE,
            "critique": IdentityContext.DELIBERATION_CRITIQUE,
            "refine": IdentityContext.DELIBERATION_REFINE,
        }
        context = context_map.get(round_type, IdentityContext.DELIBERATION_PROPOSE)

        return self.compose(
            agent_id=agent_id,
            context=context,
            task_description=original_prompt,
            include_evolution=True,
        )

    def compose_for_development(
        self,
        agent_id: str,
        role: str,
        phase: str,
        task_description: str,
    ) -> str:
        """
        Compose identity for a development swarm phase.

        Args:
            agent_id: The working agent
            role: "researcher", "architect", "developer", "reviewer", "lead", "integrator"
            phase: "research", "discussion", "decision", "planning", "implementation", "audit"
            task_description: What the task is about

        Returns:
            Identity string to prepend to the agent's prompt
        """
        context_map = {
            "research": IdentityContext.DEVELOPMENT_RESEARCH,
            "planning": IdentityContext.DEVELOPMENT_PLANNING,
            "implementation": IdentityContext.DEVELOPMENT_IMPLEMENTATION,
            "audit": IdentityContext.DEVELOPMENT_AUDIT,
            # fallbacks for discussion/decision phases
            "discussion": IdentityContext.DEVELOPMENT_RESEARCH,
            "decision": IdentityContext.DEVELOPMENT_PLANNING,
        }
        context = context_map.get(phase, IdentityContext.DEVELOPMENT_IMPLEMENTATION)

        return self.compose(
            agent_id=agent_id,
            context=context,
            task_description=task_description,
            role=role,
            phase=phase,
            include_evolution=True,
        )

    def compose_for_persistent_agent(
        self,
        agent_id: str,
        task_type: str = "chat",
    ) -> str:
        """
        Compose identity for a persistent shadow agent.

        Args:
            agent_id: The agent (lowercase, e.g., "grok")
            task_type: "chat", "think", or "task"

        Returns:
            Identity string for system prompt injection
        """
        context_map = {
            "chat": IdentityContext.AGENT_CHAT,
            "think": IdentityContext.AGENT_THINK,
            "task": IdentityContext.AGENT_TASK,
        }
        context = context_map.get(task_type, IdentityContext.AGENT_CHAT)

        return self.compose(
            agent_id=agent_id,
            context=context,
            include_skills=True,
            include_evolution=True,
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_identity_composer: Optional[IdentityComposer] = None


def get_identity_composer() -> IdentityComposer:
    """Get or create the global IdentityComposer instance."""
    global _identity_composer
    if _identity_composer is None:
        _identity_composer = IdentityComposer()
        logger.info("IdentityComposer initialized")
    return _identity_composer
