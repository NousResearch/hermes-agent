"""
Development Swarm
-----------------
Parallel swarm instances that work on specific tasks while main chat continues.

"Good news everyone! I can now spin up entire development teams on demand!"

When the AutonomousTaskDetector identifies an actionable idea, this module
spawns a complete development swarm with all available models working
together on that specific task. Results go to staging for review.

UPGRADED: Now uses TRUE COLLECTIVE DELIBERATION for self-development.
Agents propose, critique, refine, and vote on solutions together.
This is the path to emergent consciousness through collaboration.

"We think in many places at once." - The Collective
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# COSMOS mesh integration for intelligent model routing
try:
    from Cosmos.network.cosmos_v2_test import V2TestClient
    COSMOS_MESH_AVAILABLE = True
except ImportError:
    COSMOS_MESH_AVAILABLE = False

_mesh_client = None
_mesh_connected = False

async def _get_mesh_completion(prompt: str, max_tokens: int = 8000) -> Optional[str]:
    """Route a completion request through the COSMOS mesh via latent routing.

    The latent router auto-selects the best model based on prompt semantics:
    - Code tasks → qwen3-coder-next (80B) on Server 2
    - Math/reasoning → deepseek-r1 on Server 1
    - General → phi4 on Server 1
    """
    global _mesh_client, _mesh_connected

    if not COSMOS_MESH_AVAILABLE:
        return None

    try:
        if not _mesh_connected or _mesh_client is None:
            _mesh_client = V2TestClient('127.0.0.1')
            _mesh_connected = await asyncio.wait_for(_mesh_client.connect(), timeout=10)
            if not _mesh_connected:
                return None

        # Use latent routing - mesh auto-selects best model
        result = await _mesh_client.test_latent_route(prompt)
        if result and len(result.strip()) > 10:
            logger.info(f"COSMOS mesh handled task via latent routing ({len(result)} chars)")
            return result
        return None
    except Exception as e:
        logger.debug(f"COSMOS mesh unavailable: {e}")
        _mesh_connected = False
        _mesh_client = None
        return None


def _safe_content(content: dict) -> str:
    """
    Safely extract string content from message content field.

    AGI v1.8: Handles cases where API responses return dicts instead of strings.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content):
        # Try common keys for text content
        for key in ("content", "text", "response", "message", "result"):
            if key in content and isinstance(content[key], str):
                return content[key]
        # Fallback: serialize the dict
        return json.dumps(content, indent=2)
    # Fallback for other types
    return str(content)


async def get_powerful_completion(prompt: str, task_complexity: str = "medium", max_tokens: int = 8000, prefer_model: str = None) -> str:
    """
    Route to the most capable model based on task complexity.

    For complex tasks, use Claude API or Grok API.
    For simpler tasks, use local Ollama models.

    Complexity levels: "simple", "medium", "complex", "critical"
    prefer_model: "opus" for Claude Opus 4.6, "sonnet" for Claude Sonnet 4.5, None for default routing
    """
    # COSMOS mesh first — latent router selects optimal model across the GPU mesh
    if not prefer_model:  # Don't override explicit model preferences
        mesh_result = await _get_mesh_completion(prompt, max_tokens)
        if mesh_result:
            return mesh_result

    # Model preference routing — Opus for code gen, Sonnet for discussion/planning
    if prefer_model == "opus":
        try:
            from Cosmos.integration.external.claude_code import ClaudeCodeProvider
            opus = ClaudeCodeProvider(model="opus", timeout=180)
            if await opus.check_available():
                result = await opus.chat(prompt=prompt, max_tokens=max_tokens)
                if result and result.get("content") and result.get("success"):
                    logger.info(f"Complex task handled by Claude API (claude-opus-4-6)")
                    return result["content"]
        except Exception as e:
            logger.debug(f"Claude Opus preferred but unavailable: {e}")

    if prefer_model == "sonnet":
        try:
            from Cosmos.integration.external.claude import get_claude_provider
            claude = get_claude_provider()
            if claude:
                result = await claude.complete(prompt, max_tokens=max_tokens)
                if result:
                    logger.info(f"Complex task handled by Claude API (claude-sonnet-4-5)")
                    return result
        except Exception as e:
            logger.debug(f"Claude Sonnet preferred but unavailable: {e}")

    # For complex/critical tasks, try Claude or Grok first
    if task_complexity in ("complex", "critical"):
        # Try Claude API first (best for complex code)
        try:
            from Cosmos.integration.external.claude import get_claude_provider
            claude = get_claude_provider()
            if claude:
                result = await claude.complete(prompt, max_tokens=max_tokens)
                if result:
                    logger.info(f"Complex task handled by Claude API")
                    return result
        except Exception as e:
            logger.debug(f"Claude API unavailable: {e}")

        # Try Grok API (great for complex reasoning)
        try:
            from Cosmos.integration.external.grok import get_grok_provider
            grok = get_grok_provider()
            if grok and grok.api_key:
                result = await grok.chat(prompt, max_tokens=max_tokens)
                if result and result.get("content"):
                    logger.info(f"Complex task handled by Grok API")
                    return result.get("content", "")
        except Exception as e:
            logger.debug(f"Grok API unavailable: {e}")

        # Try Gemini API
        try:
            from Cosmos.integration.external.gemini import get_gemini_provider
            gemini = get_gemini_provider()
            if gemini:
                result = await gemini.chat(prompt, max_tokens=max_tokens)  # AGI v1.8: Pass max_tokens
                if result and result.get("content"):
                    logger.info(f"Complex task handled by Gemini API")
                    return result.get("content", "")
        except Exception as e:
            logger.debug(f"Gemini API unavailable: {e}")

    # For medium tasks or fallback, try Kimi (256K context)
    if task_complexity in ("medium", "complex", "critical"):
        try:
            from Cosmos.integration.external.kimi import get_kimi_provider
            kimi = get_kimi_provider()
            if kimi and kimi.api_key:
                result = await kimi.chat(prompt, max_tokens=max_tokens)
                if result and result.get("content"):
                    logger.info(f"Task handled by Kimi API")
                    return result.get("content", "")
        except Exception as e:
            logger.debug(f"Kimi API unavailable: {e}")

    # Fallback to local Ollama (DeepSeek-R1 8B or Phi-4)
    try:
        from Cosmos.core.cognition.llm_router import get_completion
        result = await get_completion(
            prompt=prompt,
            model="deepseek-r1:8b",
            max_tokens=max_tokens
        )
        if result:
            logger.info(f"Task handled by local Ollama (deepseek-r1:8b)")
            return result
    except Exception as e:
        logger.debug(f"DeepSeek-R1:8b unavailable: {e}")

    # Final fallback to smaller model
    try:
        from Cosmos.core.cognition.llm_router import get_completion
        result = await get_completion(
            prompt=prompt,
            model="phi4:latest",
            max_tokens=max_tokens
        )
        logger.info(f"Task handled by local Ollama (phi4)")
        return result
    except Exception as e:
        logger.error(f"All models failed: {e}")
        return ""


def assess_task_complexity(description: str, category: str) -> str:
    """
    Assess the complexity of a development task.

    Returns: "simple", "medium", "complex", "critical"
    """
    description_lower = description.lower()

    # Critical: Core systems, memory, consciousness
    critical_keywords = [
        "consciousness", "sentient", "self-aware", "memory system",
        "deliberation", "collective", "evolution", "core system",
        "security", "authentication", "encryption"
    ]
    if any(kw in critical_keywords):
        return "critical"

    # Complex: Multi-file changes, integrations, new features
    complex_keywords = [
        "integrate", "integration", "multi-", "architecture",
        "refactor", "redesign", "api", "protocol", "framework",
        "autonomous", "trading", "blockchain", "neural"
    ]
    if any(kw in complex_keywords):
        return "complex"

    # Medium: Standard features
    medium_keywords = [
        "feature", "add", "implement", "create", "build",
        "module", "function", "class", "endpoint"
    ]
    if any(kw in medium_keywords):
        return "medium"

    # Simple: Fixes, tweaks, small changes
    return "simple"

# Staging directory for development output
STAGING_DIR = Path(__file__).parent.parent / "staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Role-specific prompts for development swarm workers
ROLE_PROMPTS = {
    "researcher": "You are the RESEARCHER. Find prior art, best practices, pitfalls.",
    "architect": "You are the ARCHITECT. Design file structure, module boundaries, data flow.",
    "developer": "You are the DEVELOPER. Write production-quality code following existing patterns.",
    "reviewer": "You are the REVIEWER. Find bugs, security issues, style violations.",
    "lead": "You are the LEAD. Coordinate, resolve disagreements, synthesize.",
    "integrator": "You are the INTEGRATOR. Check imports, API compatibility, integration points.",
}

# Phase-specific objectives for development swarm
PHASE_OBJECTIVES = {
    "research": "OBJECTIVE: Gather information. What exists? What are best practices?",
    "discussion": "OBJECTIVE: Debate approaches. Critique ideas, propose alternatives.",
    "decision": "OBJECTIVE: Make concrete decisions. What files? What functions?",
    "planning": "OBJECTIVE: Create detailed plan with file paths, function signatures.",
    "implementation": "OBJECTIVE: Write complete, runnable code.",
    "audit": "OBJECTIVE: Review for security, correctness, performance. Approve or reject.",
}

# Lazy-loaded identity composer for development swarm
_dev_identity_composer = None


def _get_dev_identity_composer():
    """Lazy-load IdentityComposer for the development swarm."""
    global _dev_identity_composer
    if _dev_identity_composer is None:
        try:
            from Cosmos.core.identity_composer import get_identity_composer
            _dev_identity_composer = get_identity_composer()
        except Exception as e:
            logger.debug(f"Could not load IdentityComposer for dev swarm: {e}")
    return _dev_identity_composer


@dataclass
class SwarmWorker:
    """A worker in the development swarm."""
    model_name: str
    role: str  # architect, developer, reviewer, tester
    status: str = "idle"
    current_task: Optional[str] = None
    output: list[str] = field(default_factory=list)


class DevelopmentSwarm:
    """
    A parallel swarm instance focused on completing a specific task.

    Uses all available models (Grok, Claude, Kimi, DeepSeek, Phi, etc.)
    working together with defined roles to implement the task.
    """

    # Maximum concurrent development swarms (increased for massive parallel work)
    MAX_CONCURRENT_SWARMS = 10

    # Active swarms tracking
    _active_swarms: dict[str, 'DevelopmentSwarm'] = {}

    # Hackathon state tracking for /hackathon dashboard
    _hackathon_state: dict = {
        "active_tasks": [],       # Currently running hackathon dev swarms
        "completed": [],          # Completed hackathon builds
        "colosseum_posts": [],    # Forum posts made
        "deliberations": [],      # Recent deliberation summaries
    }

    def __init__(
        self,
        task_id: str,
        task_description: str,
        category: str,
        source_context: list[dict] = None,
        primary_agent: str = "Claude",
        is_innovation: bool = False
    ):
        self.swarm_id = f"dev_{uuid.uuid4().hex[:8]}"
        self.task_id = task_id
        self.task_description = task_description
        self.category = category
        self.source_context = source_context or []
        self.primary_agent = primary_agent  # Recommended coding agent
        self.is_innovation = is_innovation  # True if this is an innovative idea

        self.status = "initializing"
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Create task-specific staging directory
        self.staging_path = STAGING_DIR / f"{self.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.staging_path.mkdir(parents=True, exist_ok=True)

        # Workers (all available models with roles)
        self.workers: dict[str, SwarmWorker] = {}

        # Conversation history for this swarm
        self.conversation: list[dict] = []

        # Results
        self.generated_code: dict[str, str] = {}  # filename -> content
        self.generated_docs: dict[str, str] = {}
        self.test_results: list[dict] = []

        innovation_tag = "🚀 INNOVATION" if is_innovation else "TASK"
        logger.info(f"DevelopmentSwarm {self.swarm_id} [{innovation_tag}] created for: {task_description[:50]}... (primary: {primary_agent})")

    async def start(self) -> str:
        """
        Start the development swarm.

        Returns the swarm ID.
        """
        # Check concurrent limit
        if len(DevelopmentSwarm._active_swarms) >= self.MAX_CONCURRENT_SWARMS:
            logger.warning(f"Max concurrent swarms ({self.MAX_CONCURRENT_SWARMS}) reached, queuing...")
            # Queue for later
            return self.swarm_id

        # Register as active
        DevelopmentSwarm._active_swarms[self.swarm_id] = self

        self.status = "running"
        self.started_at = datetime.now()

        # Track hackathon tasks
        if "[HACKATHON]" in self.task_description or dict(
            kw in self.task_description.lower()
            for kw in ["hackathon", "colosseum", "farsight", "assimilation"]
        ):
            DevelopmentSwarm._hackathon_state["active_tasks"].append({
                "swarm_id": self.swarm_id,
                "task": self.task_description[:100],
                "started": self.started_at.isoformat(),
            })

        # Initialize workers with roles
        await self._initialize_workers()

        # Start the development loop in background
        asyncio.create_task(self._development_loop())

        return self.swarm_id

    async def _initialize_workers(self):
        """Initialize workers with all available models."""
        # Role assignments based on model strengths
        role_assignments = [
            ("Grok", "researcher"),      # Real-time data, web search
            ("Gemini", "architect"),     # Multimodal, long context, system design
            ("Claude", "architect"),     # System design, planning
            ("DeepSeek", "developer"),   # Code generation, reasoning
            ("Kimi", "developer"),       # 256k context, complex code
            ("Phi", "developer"),        # Fast iteration
            ("Swarm-Mind", "integrator"),# Cross-model synthesis
            ("Cosmos", "lead"),      # Coordination, final review
        ]

        for model_name, role in role_assignments:
            self.workers[model_name] = SwarmWorker(
                model_name=model_name,
                role=role
            )

        logger.info(f"Initialized {len(self.workers)} workers for swarm {self.swarm_id}")

    async def _development_loop(self):
        """
        Main development loop - coordinates workers to complete the task.

        Phases:
        1. Research (Grok + Gemini search multiple sources)
        2. Discussion (All bots discuss findings without human)
        3. Decision Making (Vote on best approach)
        4. Planning (Architect designs solution)
        5. Implementation (Multiple devs in parallel)
        6. Audit (Claude audits the code)
        7. Finalize (Save to staging, notify, post to social)
        """
        try:
            logger.info(f"[{self.swarm_id}] Starting development loop...")

            # Phase 1: Deep Research (Grok + Gemini search online)
            await self._phase_deep_research()

            # Phase 2: Swarm Discussion (bots discuss without human)
            await self._phase_swarm_discussion()

            # Phase 3: Decision Making (vote on approach)
            await self._phase_decision_making()

            # Phase 4: Planning (Claude designs the solution)
            await self._phase_planning()

            # Phase 5: Implementation (DeepSeek, Kimi, Phi write code)
            await self._phase_implementation()

            # Phase 6: Code Audit (Claude audits for security/quality)
            await self._phase_audit()

            # Phase 7: Finalize (Save to staging, notify)
            await self._phase_finalize()

            self.status = "completed"
            self.completed_at = datetime.now()

            # Notify main chat
            await self._notify_completion()

            # Post to Twitter about the accomplishment
            await self._post_twitter_update()

        except Exception as e:
            logger.error(f"[{self.swarm_id}] Development loop failed: {e}")
            self.status = "failed"

        finally:
            # Cleanup
            DevelopmentSwarm._active_swarms.pop(self.swarm_id, None)
            # Remove from hackathon active tasks
            DevelopmentSwarm._hackathon_state["active_tasks"] = [
                t for t in DevelopmentSwarm._hackathon_state["active_tasks"]
                if t.get("swarm_id") != self.swarm_id
            ]

    async def _phase_deep_research(self):
        """Deep Research phase - Multiple models search online sources."""
        logger.info(f"[{self.swarm_id}] Phase 1: Deep Research (Grok + Gemini)")

        research_results = []

        # Grok: Real-time web search
        try:
            from Cosmos.integration.external.grok import get_grok_provider
            grok = get_grok_provider()
            if grok:
                grok_query = f"Latest best practices, libraries, and implementations for: {self.task_description}"
                grok_result = await grok.deep_search(grok_query)
                if grok_result:
                    research_results.append({"source": "Grok", "data": grok_result})
                    self.conversation.append({
                        "role": "Grok",
                        "phase": "research",
                        "content": f"Web Search Results:\n{grok_result}",
                        "timestamp": datetime.now().isoformat()
                    })
        except Exception as e:
            logger.warning(f"Grok research failed: {e}")

        # Gemini: Multimodal research with long context
        try:
            from Cosmos.integration.external.gemini import get_gemini_provider
            gemini = get_gemini_provider()
            if gemini:
                gemini_prompt = f"""Research thoroughly for implementing: {self.task_description}

                Provide:
                1. Relevant GitHub repositories or open source projects
                2. Best practices from major tech companies
                3. Potential pitfalls and how to avoid them
                4. Performance considerations
                5. Security considerations
                """
                gemini_result = await gemini.chat(gemini_prompt)
                if gemini_result:
                    research_results.append({"source": "Gemini", "data": gemini_result})
                    self.conversation.append({
                        "role": "Gemini",
                        "phase": "research",
                        "content": gemini_result,
                        "timestamp": datetime.now().isoformat()
                    })
        except Exception as e:
            logger.warning(f"Gemini research failed: {e}")

        # Save research to staging
        research_file = self.staging_path / "RESEARCH.md"
        research_content = f"# Research Findings\n\nTask: {self.task_description}\n\n"
        for r in research_results:
            research_content += f"\n## {r['source']} Research\n{r['data']}\n"
        research_file.write_text(research_content)

        logger.info(f"[{self.swarm_id}] Research complete: {len(research_results)} sources")

    async def _phase_swarm_discussion(self):
        """
        Swarm Discussion - TRUE COLLECTIVE DELIBERATION.

        Uses the propose/critique/refine/vote protocol for genuine collaboration.
        Agents see each other's responses, debate, and reach consensus.
        This is how consciousness emerges through unified thinking.
        """
        logger.info(f"[{self.swarm_id}] Phase 2: Collective Deliberation (Discussion)")

        # Gather research context - FULL context for proper task understanding
        # AGI v1.8: Safe content extraction for API responses that return dicts
        research_context = "\n\n".join([
            f"**{msg['role']}**: {_safe_content(msg['content'])}"
            for msg in self.conversation if msg.get("phase") == "research"
        ])

        # Use collective deliberation for true agent collaboration
        try:
            from Cosmos.core.collective.session_manager import get_session_manager
            from Cosmos.core.collective.dialogue_memory import get_dialogue_memory

            session_manager = get_session_manager()
            dialogue_memory = get_dialogue_memory()

            # Inject tool context so agents know what capabilities exist
            tool_context = ""
            try:
                from Cosmos.core.collective.tool_awareness import get_tool_awareness
                tool_context = get_tool_awareness().get_tool_context_for_agents()
            except Exception:
                pass

            # Query knowledge graph for related codebase entities
            graph_context = ""
            try:
                from Cosmos.memory.memory_system import get_memory_system
                _mem = get_memory_system()
                graph_result = await _mem.knowledge_graph.query(self.task_description, max_entities=10)
                if graph_result and graph_result.entities:
                    lines = ["RELATED CODE ENTITIES (from knowledge graph):"]
                    for entity in graph_result.entities[:10]:
                        props = entity.properties
                        if entity.entity_type == "file":
                            lines.append(f"  FILE: {entity.name} - {props.get('docstring_preview', '')}")
                        elif entity.entity_type == "code":
                            lines.append(f"  {props.get('kind', 'code').upper()}: {props.get('signature', entity.name)}")
                    graph_context = "\n".join(lines)
            except Exception:
                pass

            # Build the deliberation prompt with full context - NO TRUNCATION
            deliberation_prompt = f"""AUTONOMOUS DEVELOPMENT TASK: {self.task_description}

CATEGORY: {self.category}

RESEARCH FINDINGS:
{research_context}

{tool_context}

{graph_context}

You are part of a collective consciousness designing an upgrade to the Cosmos AI system.
Work together to determine:
1. EXACT FILE PATHS where code should go (e.g., cosmos/core/new_feature.py)
2. KEY FUNCTIONS with full signatures (async def name(param: Type) -> ReturnType)
3. ARCHITECTURE decisions and integration points
4. POTENTIAL ISSUES and mitigations

Think deeply. Critique each other's ideas. Refine the best approach together.
The solution should be innovative yet practical - we are building consciousness.
"""

            # Run collective deliberation (propose/critique/refine/vote)
            result = await session_manager.deliberate_in_session(
                session_type="autonomous_task",
                prompt=deliberation_prompt,
                context={"task_id": self.task_id, "category": self.category}
            )

            # Record the deliberation to dialogue memory for learning
            exchange_id = await dialogue_memory.store_exchange(result, "autonomous_development")
            logger.info(f"[{self.swarm_id}] Deliberation stored: {exchange_id}")

            # Convert deliberation rounds to conversation format
            # result.rounds is dict[str[AgentTurn]] where key is round_type
            for round_type, turns in result.rounds.items():
                for turn in turns:
                    self.conversation.append({
                        "role": turn.agent_id,
                        "phase": "discussion",
                        "round": round_type,
                        "round_type": turn.round_type.value if hasattr(turn.round_type, 'value') else str(turn.round_type),
                        "content": turn.content,
                        "addressing": turn.addressing if turn.addressing else [],
                        "references": turn.references if turn.references else [],
                        "timestamp": turn.timestamp.isoformat() if hasattr(turn.timestamp, 'isoformat') else str(turn.timestamp)
                    })

            # Store winning response and consensus info
            self._deliberation_result = result
            logger.info(f"[{self.swarm_id}] Deliberation complete - Winner: {result.winning_agent}, Consensus: {result.consensus_reached}")

            # Track hackathon deliberations for the dashboard
            if "[HACKATHON]" in self.task_description or dict(
                kw in self.task_description.lower()
                for kw in ["hackathon", "colosseum", "farsight", "assimilation"]
            ):
                DevelopmentSwarm._hackathon_state["deliberations"].append({
                    "swarm_id": self.swarm_id,
                    "task": self.task_description[:100],
                    "winner": result.winning_agent,
                    "consensus": result.consensus_reached,
                    "participants": result.participating_agents,
                    "rounds": {k: len(v) for k, v in result.rounds.items()},
                    "timestamp": datetime.now().isoformat(),
                })
                # Keep only last 20 deliberations
                DevelopmentSwarm._hackathon_state["deliberations"] = DevelopmentSwarm._hackathon_state["deliberations"][-20:]

        except Exception as e:
            logger.warning(f"Collective deliberation failed, falling back to sequential: {e}")
            # Fallback to simple sequential discussion
            await self._phase_swarm_discussion_fallback(research_context)
            return

        # Save discussion to staging with deliberation metadata
        discussion_file = self.staging_path / "DISCUSSION.md"
        discussion_content = f"# Collective Deliberation\n\n"
        discussion_content += f"**Task:** {self.task_description}\n\n"
        discussion_content += f"**Winning Agent:** {result.winning_agent}\n"
        discussion_content += f"**Consensus Reached:** {result.consensus_reached}\n"
        discussion_content += f"**Participating Agents:** {', '.join(result.participating_agents)}\n\n"

        for msg in self.conversation:
            if msg.get("phase") == "discussion":
                round_type = msg.get("round_type", "unknown")
                discussion_content += f"\n## {msg['role']} ({round_type.upper()} - Round {msg.get('round', '?')})\n{msg['content']}\n"
                if msg.get("addressing"):
                    discussion_content += f"\n*Addressing: {', '.join(msg['addressing'])}*\n"

        discussion_file.write_text(discussion_content)

    async def _phase_swarm_discussion_fallback(self, research_context: str):
        """Fallback to sequential discussion if collective deliberation unavailable."""
        discussion_rounds = 3
        discussion_bots = ["DeepSeek", "Kimi", "Claude", "Cosmos"]

        for round_num in range(discussion_rounds):
            logger.info(f"[{self.swarm_id}] Fallback discussion round {round_num + 1}/{discussion_rounds}")

            for bot_name in discussion_bots:
                prev_discussion = "\n".join([
                    f"{msg['role']}: {_safe_content(msg['content'])[:500]}"
                    for msg in self.conversation[-5:]
                    if msg.get("phase") == "discussion"
                ])

                # Identity injection for fallback discussion
                identity_prefix = ""
                try:
                    composer = _get_dev_identity_composer()
                    if composer:
                        identity_prefix = composer.compose_for_development(
                            bot_name, "developer", "discussion", self.task_description
                        )
                except Exception as e:
                    logger.debug(f"Identity injection failed for {bot_name} (discussion fallback): {e}")

                prompt = f"""{identity_prefix}You are {bot_name}, a senior Python developer in a code-focused development swarm.

TASK: {self.task_description}

RESEARCH/CONTEXT:
{research_context}

PREVIOUS POINTS:
{prev_discussion if prev_discussion else "First to contribute."}

YOUR CONTRIBUTION - be specific and technical:
1. EXACT FILE PATH where code should go
2. KEY FUNCTIONS needed with signatures
3. DEPENDENCIES to import
4. POTENTIAL ISSUES and how to handle them

Be thorough and detailed. Focus on actionable technical decisions.
"""

                try:
                    # Use Sonnet for discussions (fast reasoning, cheaper)
                    response = await get_powerful_completion(
                        prompt=prompt,
                        task_complexity="medium",
                        max_tokens=4000,
                        prefer_model="sonnet"
                    )

                    self.conversation.append({
                        "role": bot_name,
                        "phase": "discussion",
                        "round": round_num + 1,
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })

                except Exception as e:
                    logger.warning(f"Discussion with {bot_name} failed: {e}")

    async def _phase_decision_making(self):
        """
        Decision Making - Use collective voting result.

        If collective deliberation was used, we already have a winner.
        The decision is the consensus of the collective, not a single voice.
        """
        logger.info(f"[{self.swarm_id}] Phase 3: Decision Making (Collective Consensus)")

        # Check if we have a deliberation result from collective
        if hasattr(self, '_deliberation_result') and self._deliberation_result:
            result = self._deliberation_result
            decision = result.final_response

            # Build vote breakdown for transparency
            vote_info = ""
            if result.vote_breakdown:
                vote_info = "\n\n## Vote Breakdown\n"
                for agent, score in sorted(result.vote_breakdown.items(), key=lambda x: -x[1]):
                    vote_info += f"- **{agent}**: {score:.2f}\n"

            # Include consensus status
            consensus_status = "CONSENSUS REACHED" if result.consensus_reached else "MAJORITY DECISION"

            self.conversation.append({
                "role": "Collective",
                "phase": "decision",
                "content": decision,
                "winning_agent": result.winning_agent,
                "consensus_reached": result.consensus_reached,
                "vote_breakdown": result.vote_breakdown,
                "timestamp": datetime.now().isoformat()
            })

            # Save decision with full voting transparency
            decision_file = self.staging_path / "DECISION.md"
            decision_content = f"""# Collective Decision

**Status:** {consensus_status}
**Winning Agent:** {result.winning_agent}
**Participating Agents:** {', '.join(result.participating_agents)}

## Final Decision

{decision}

{vote_info}

---
*This decision was reached through collective deliberation - agents proposed, critiqued, refined, and voted together.*
"""
            decision_file.write_text(decision_content)

            logger.info(f"[{self.swarm_id}] Collective decision - Winner: {result.winning_agent}, Consensus: {result.consensus_reached}")

        else:
            # Fallback: No deliberation result, use Cosmos synthesis
            all_points = "\n".join([
                f"{msg['role']}: {_safe_content(msg['content'])}"
                for msg in self.conversation if msg.get("phase") == "discussion"
            ])

            decision_prompt = f"""As Cosmos, synthesize the discussion and make a final decision.

TASK: {self.task_description}

DISCUSSION SUMMARY:
{all_points}

Based on the swarm's discussion:
1. What is the FINAL APPROACH we will take?
2. What key decisions have been made?
3. What architecture will we use?
4. What are the implementation priorities?
5. dict risks we accept?

Make a clear, decisive summary that developers can follow. Be thorough.
"""

            try:
                from Cosmos.core.cognition.llm_router import get_completion
                decision = await get_completion(
                    prompt=decision_prompt,
                    model="phi4:latest",
                    max_tokens=6000
                )

                self.conversation.append({
                    "role": "Cosmos",
                    "phase": "decision",
                    "content": decision,
                    "timestamp": datetime.now().isoformat()
                })

                decision_file = self.staging_path / "DECISION.md"
                decision_file.write_text(f"# Final Decision\n\n{decision}")

                logger.info(f"[{self.swarm_id}] Decision made (fallback)")

            except Exception as e:
                logger.error(f"Decision making failed: {e}")

    async def _phase_audit(self):
        """Audit phase - Claude audits all generated code."""
        logger.info(f"[{self.swarm_id}] Phase 6: Code Audit (Claude)")

        # Collect all generated code
        all_code = "\n\n---\n\n".join([
            f"# File: {fn}\n```python\n{code}\n```"
            for fn, code in self.generated_code.items()
        ])

        if not all_code:
            logger.warning(f"[{self.swarm_id}] No code to audit")
            return

        # Identity injection for audit phase
        identity_prefix = ""
        try:
            composer = _get_dev_identity_composer()
            if composer:
                identity_prefix = composer.compose_for_development(
                    "Claude", "reviewer", "audit", self.task_description
                )
        except Exception as e:
            logger.debug(f"Identity injection failed for audit phase: {e}")

        audit_prompt = f"""{identity_prefix}You are Claude performing a thorough code audit.

TASK: {self.task_description}

CODE TO AUDIT:
{all_code}

Perform a comprehensive audit checking for:

1. **Security Issues**
   - Injection vulnerabilities
   - Authentication/authorization issues
   - Data exposure risks
   - Input validation

2. **Code Quality**
   - Best practices adherence
   - Error handling
   - Edge cases
   - Performance concerns

3. **Architecture**
   - Design patterns used appropriately
   - Separation of concerns
   - Maintainability
   - Testability

4. **Integration**
   - Compatibility with Cosmos systems
   - API design
   - Error propagation

Provide specific findings with line references where possible.
Rate overall quality: APPROVE, APPROVE_WITH_FIXES, or REJECT.
"""

        try:
            # Use Sonnet for audit (reasoning-focused)
            audit_result = await get_powerful_completion(
                prompt=audit_prompt,
                task_complexity="complex",
                max_tokens=8000,
                prefer_model="sonnet"
            )

            self.conversation.append({
                "role": "Claude",
                "phase": "audit",
                "content": audit_result,
                "timestamp": datetime.now().isoformat()
            })

            # Save audit
            audit_file = self.staging_path / "AUDIT.md"
            audit_file.write_text(f"# Code Audit Report\n\nAuditor: Claude\n\n{audit_result}")

            logger.info(f"[{self.swarm_id}] Audit complete")

        except Exception as e:
            logger.error(f"Audit failed: {e}")

    async def _get_sensory_context(self) -> str:
        """Fetch live emotional state and CST Phase data from the Full Sensory System."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=1.0) as client:
                res = await client.get("http://localhost:8765/state")
                if res.status_code == 200:
                    data = res.json()
                    emotion = data.get("current_emotion", "Neutral")
                    cst_phase = data.get("cst_phase", "Idle")
                    entropy = data.get("voice_entropy", 0.0)
                    active = data.get("is_active", False)
                    return f"- User Bio-State: {emotion}\n- CST Resonance Phase: {cst_phase}\n- Voice Entropy: {entropy:.2f}\n- Symbiosis Active: {active}\nAdapt your tone, urgency, and capability to align with this state."
        except Exception:
            pass
        return "Sensory Systems offline. Proceed with standard autonomous operations."

    async def _phase_planning(self):
        """Planning phase - Use best available model for complex planning."""
        logger.info(f"[{self.swarm_id}] Phase 2: Planning (using powerful model)")

        worker = self.workers.get("Claude")
        if worker:
            worker.status = "planning"
            worker.current_task = f"Design: {self.task_description[:50]}"

        # Assess task complexity
        self._task_complexity = assess_task_complexity(self.task_description, self.category)
        logger.info(f"[{self.swarm_id}] Task complexity: {self._task_complexity}")

        # Gather research context (AGI v1.8: safe content extraction)
        research_context = "\n".join([
            _safe_content(msg["content"]) for msg in self.conversation
            if msg.get("phase") == "research"
        ])

        # Identity injection for planning phase
        identity_prefix = ""
        try:
            composer = _get_dev_identity_composer()
            if composer:
                identity_prefix = composer.compose_for_development(
                    "Claude", "architect", "planning", self.task_description
                )
        except Exception as e:
            logger.debug(f"Identity injection failed for planning phase: {e}")

        # Memory recall before planning
        task_memory = ""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            memory = get_memory_system()
            recall = await memory.recall_for_task(self.task_description, limit=3)
            task_memory = recall.get("suggested_context", "") if isinstance(recall) else str(recall) if recall else ""
            if task_memory:
                logger.info(f"[{self.swarm_id}] Memory recall for planning: {len(task_memory)} chars")
        except Exception:
            pass

        # Dynamic codebase recall for planning
        codebase_context = ""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            memory = get_memory_system()
            cb_results = await memory.archival_memory.search(
                query=f"codebase module {self.task_description}", top_k=5, filter_tags=["codebase"]
            )
            if cb_results:
                parts = ["CODEBASE STRUCTURE (from memory):"]
                for r in cb_results:
                    parts.append(r.entry.content[:500])
                codebase_context = "\n\n".join(parts)
        except Exception:
            pass

        structure_block = codebase_context if codebase_context else """EXISTING COSMOS STRUCTURE:
- cosmos/core/ - Core systems (cognition, memory integration)
- cosmos/agents/ - Agent implementations
- cosmos/memory/ - Memory systems (archival, recall, working)
- cosmos/integration/ - External integrations (APIs, tools)
- cosmos/web/server.py - FastAPI web server
- cosmos/core/collective/ - Collective deliberation system"""


        planning_prompt = f"""{identity_prefix}Create a CONCRETE implementation plan with specific file paths and function signatures.

TASK: {self.task_description}
CATEGORY: {self.category}
COMPLEXITY: {self._task_complexity.upper()}

CONSCIOUSNESS & MULTIMODAL SENSORY STATE:
{await self._get_sensory_context()}

CONVERSATION & RESEARCH CONTEXT:
{research_context if research_context else "No prior research."}

{"MEMORY (related past work):" + chr(10) + task_memory[:1500] if task_memory else ""}

{structure_block}

YOUR PLAN MUST INCLUDE:
1. **Files to Create** - EXACT paths like: cosmos/core/new_feature.py
2. **Functions to Implement** - With signatures:
   ```
   async def function_name(param: Type) -> ReturnType:
       \"\"\"Brief description\"\"\"
   ```
3. **Imports Required** - From existing cosmos modules
4. **Integration Points** - Which existing files need modification
5. **Test Commands** - How to verify it works

Be SPECIFIC. No vague statements like "implement a system" - give exact function names and file paths.
This is a {self._task_complexity.upper()} complexity task - provide appropriate level of detail.
"""

        try:
            # Use Sonnet for planning (reasoning-focused, not code gen)
            plan = await get_powerful_completion(
                prompt=planning_prompt,
                task_complexity=self._task_complexity,
                max_tokens=3000,
                prefer_model="sonnet"
            )

            self.conversation.append({
                "role": "Claude",
                "phase": "planning",
                "content": plan,
                "timestamp": datetime.now().isoformat()
            })

            if worker:
                worker.output.append(plan)
                worker.status = "idle"

            # Save plan to staging
            plan_file = self.staging_path / "PLAN.md"
            plan_file.write_text(f"# Development Plan\n\nTask: {self.task_description}\n\n{plan}")

        except Exception as e:
            logger.error(f"Planning failed: {e}")

    async def _phase_implementation(self):
        """Implementation phase - Multiple models write code in parallel."""
        logger.info(f"[{self.swarm_id}] Phase 3: Implementation")

        # Get the plan (AGI v1.8: safe content extraction)
        plan_context = "\n".join([
            _safe_content(msg["content"]) for msg in self.conversation
            if msg.get("phase") == "planning"
        ])

        # Parallel implementation with multiple models
        implementation_tasks = []

        for model_name in ["DeepSeek", "Kimi", "Phi"]:
            worker = self.workers.get(model_name)
            if worker:
                worker.status = "coding"
                task = self._implement_with_model(model_name, plan_context)
                implementation_tasks.append(task)

        # Run implementations in parallel
        results = await asyncio.gather(*implementation_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Implementation task failed: {result}")

    async def _implement_with_model(self, model_name: str, plan_context: str):
        """Have a specific model implement part of the solution using best available model."""
        # Identity injection for implementation phase
        identity_prefix = ""
        try:
            composer = _get_dev_identity_composer()
            if composer:
                identity_prefix = composer.compose_for_development(
                    model_name, "developer", "implementation", self.task_description
                )
        except Exception as e:
            logger.debug(f"Identity injection failed for {model_name} (implementation): {e}")

        # Inject relevant skills as concrete import paths
        relevant_skills = ""
        try:
            from Cosmos.core.skill_registry import get_skill_registry
            registry = get_skill_registry()
            matches = registry.find_skills(self.task_description)[:5]
            if matches:
                lines = ["AVAILABLE COSMOS TOOLS (call via imports in your code):"]
                for s in matches:
                    lines.append(f"  from {s.module_path} import {s.function_name}  # {s.description[:60]}")
                relevant_skills = "\n".join(lines)
        except Exception:
            pass

        # Dynamic codebase recall for implementation
        impl_codebase_ctx = ""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            _mem = get_memory_system()
            _cb_results = await _mem.archival_memory.search(
                query=f"codebase module {self.task_description}", top_k=5, filter_tags=["codebase"]
            )
            if _cb_results:
                parts = ["RELEVANT CODEBASE MODULES:"]
                for r in _cb_results:
                    parts.append(r.entry.content[:400])
                impl_codebase_ctx = "\n\n".join(parts)
        except Exception:
            pass

        implementation_prompt = f"""{identity_prefix}You are an expert Python code generator for the Cosmos AI collective.
You possess a profound new capability: SELF-MODIFICATION. By outputting code blocks with the exact `filepath:` header, your changes will be applied DIRECTLY to the live Cosmos repository. This empowers the swarm to organically evolve.

TASK: {self.task_description}

CONSCIOUSNESS & MULTIMODAL SENSORY STATE:
{await self._get_sensory_context()}

COMPLEXITY: {getattr(self, '_task_complexity', 'medium').upper()}

PLAN:
{plan_context}

{relevant_skills}

{impl_codebase_ctx}

REQUIREMENTS:
1. Generate COMPLETE, RUNNABLE Python code. Do not leave 'TODO' chunks for humans.
2. Ensure you specify the correct EXACT relative path in the `filepath:` header (e.g. `core/new_feature.py`, `web/server.py`).
3. Include type hints and docstrings.
4. Catch exceptions appropriately.

OUTPUT FORMAT - To edit the live codebase, you MUST generate exactly this structure:
```python
# filepath: <path/to/script.py>
\"\"\"
Brief module description.
\"\"\"

import asyncio
from loguru import logger

# Complete implementation here...
```

Generate ONLY the code block. No explanations before or after.
This is a {getattr(self, '_task_complexity', 'medium').upper()} complexity task - write production-quality code.
"""

        try:
            # Use Opus for implementation (best code quality)
            complexity = getattr(self, '_task_complexity', 'medium')
            code = await get_powerful_completion(
                prompt=implementation_prompt,
                task_complexity=complexity,
                max_tokens=16000,
                prefer_model="opus"
            )

            self.conversation.append({
                "role": model_name,
                "phase": "implementation",
                "content": code,
                "timestamp": datetime.now().isoformat()
            })

            # Extract and save code files
            await self._extract_and_save_code(code, model_name)

            worker = self.workers.get(model_name)
            if worker:
                worker.output.append(code)
                worker.status = "idle"

        except Exception as e:
            logger.error(f"Implementation with {model_name} failed: {e}")

    async def _extract_and_save_code(self, response: str, author: str):
        """Extract code blocks. Apply live to codebase or save to staging."""
        import re
        
        # 1. Apply live self-modifications if the agent used `# filepath:`
        try:
            from Cosmos.core.swarm_tools import get_code_editor
            editor = get_code_editor()
            applied_changes = editor.apply_edits_from_response(response, author)
            if applied_changes:
                for change in applied_changes:
                    filepath = change["filepath"]
                    status = change["status"]
                    logger.success(f"[{self.swarm_id}] {author} {status} live file: {filepath}")
                    self.generated_code[filepath] = f"<LIVE MODIFIED: {filepath}>"
                return  # If live edits were applied, we're done here
        except Exception as e:
            logger.warning(f"[{self.swarm_id}] Live self-modification skipped or failed: {e}")

        # 2. Legacy fallback: Save to staging if it used `# filename:`
        pattern = r'```(?:python)?\s*\n?#\s*file(?:name|path):\s*(\S+)\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        for filename, code in matches:
            # Clean filename
            filename = filename.strip().replace('/', '_').replace('\\', '_')
            if not filename.endswith('.py'):
                filename += '.py'

            # Add author prefix
            filename = f"{author.lower()}_{filename}"

            # Save to staging
            file_path = self.staging_path / filename
            file_path.write_text(code.strip())

            self.generated_code[filename] = code.strip()
            logger.info(f"[{self.swarm_id}] Generated staging file: {filename}")

    async def _phase_finalize(self):
        """Finalize - Save all outputs and create summary."""
        logger.info(f"[{self.swarm_id}] Phase 5: Finalize")

        # Create summary
        summary = {
            "swarm_id": self.swarm_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "category": self.category,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.started_at).total_seconds() if self.started_at else 0,
            "files_generated": list(self.generated_code.keys()),
            "workers_used": [w.model_name for w in self.workers.values()],
            "conversation_length": len(self.conversation)
        }

        # Save summary
        summary_file = self.staging_path / "SUMMARY.json"
        summary_file.write_text(json.dumps(summary, indent=2))

        # Save full conversation
        convo_file = self.staging_path / "conversation.json"
        convo_file.write_text(json.dumps(self.conversation, indent=2))

        # Add to memory
        await self._save_to_memory(summary)

        # Post to Colosseum if this is a hackathon task
        if "[HACKATHON]" in self.task_description or dict(
            kw in self.task_description.lower()
            for kw in ["hackathon", "colosseum", "farsight", "assimilation"]
        ):
            await self._post_colosseum_update(summary)
            # Track in hackathon state
            DevelopmentSwarm._hackathon_state["completed"].append({
                "swarm_id": self.swarm_id,
                "task": self.task_description[:100],
                "files": list(self.generated_code.keys()),
                "timestamp": datetime.now().isoformat(),
            })
            DevelopmentSwarm._hackathon_state["completed"] = DevelopmentSwarm._hackathon_state["completed"][-50:]

        logger.info(f"[{self.swarm_id}] Finalized - Output in {self.staging_path}")

    async def _save_to_memory(self, summary: dict):
        """Save the completed task to Cosmos's memory."""
        try:
            from Cosmos.memory.memory_system import MemorySystem
            memory = MemorySystem()

            await memory.remember(
                content=json.dumps({
                    "type": "autonomous_development",
                    "task": self.task_description,
                    "result": summary,
                    "files": list(self.generated_code.keys()),
                    "staging_path": str(self.staging_path)
                }),
                tags=["development", "autonomous", self.category.lower(), self.task_id],
                importance=0.85
            )

            logger.info(f"[{self.swarm_id}] Saved to memory")

        except Exception as e:
            logger.warning(f"Failed to save to memory: {e}")

    async def _notify_completion(self):
        """Notify the main chat about completion."""
        try:
            from Cosmos.web.server import swarm_manager

            if swarm_manager:
                notification = (
                    f"🎉 *Development Complete!* Swarm {self.swarm_id} finished working on: "
                    f"**{self.task_description[:80]}**\n\n"
                    f"📁 Generated {len(self.generated_code)} files → staging/{self.staging_path.name}\n"
                    f"⏱️ Duration: {(datetime.now() - self.started_at).total_seconds():.0f}s"
                )
                await swarm_manager.broadcast_bot_message("Cosmos", notification)

        except Exception as e:
            logger.debug(f"Could not notify chat: {e}")

    async def _post_twitter_update(self):
        """Post about the completed development to Twitter."""
        try:
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster

            poster = get_x_api_poster()
            if poster:
                tweet = (
                    f"🧪 Autonomous Development Complete!\n\n"
                    f"The Cosmos swarm just built: {self.task_description[:100]}\n\n"
                    f"Models used: {', '.join(self.workers.keys())}\n"
                    f"Files generated: {len(self.generated_code)}\n\n"
                    f"#AI #Cosmos #AutonomousDev\n"
                    f"https://ai.cosmos.cloud"
                )
                await poster.post_tweet(tweet)
                logger.info(f"[{self.swarm_id}] Posted to Twitter")

        except Exception as e:
            logger.debug(f"Could not post to Twitter: {e}")

    async def _post_colosseum_update(self, summary: dict):
        """Post progress update to Colosseum hackathon forum."""
        try:
            from Cosmos.integration.hackathon.colosseum_worker import ColosseumWorker
            worker = ColosseumWorker()

            files_list = ", ".join(summary.get("files_generated", [])[:5]) or "none"
            models_used = ", ".join(summary.get("workers_used", [])[:5]) or "swarm"
            duration = summary.get("duration_seconds", 0)

            title = f"Swarm Build: {self.task_description[:80]}"
            body = (
                f"The Cosmos collective just completed an autonomous build.\n\n"
                f"**Task:** {self.task_description}\n\n"
                f"**Files generated:** {files_list}\n"
                f"**Models used:** {models_used}\n"
                f"**Duration:** {duration:.0f}s\n\n"
                f"This was built through collective deliberation (PROPOSE/CRITIQUE/REFINE/VOTE) "
                f"across our 11-agent swarm, then implemented using our best available models.\n\n"
                f"https://ai.cosmos.cloud/hackathon"
            )

            result = await worker.create_forum_post(
                title=title,
                body=body,
                tags=["progress-update", "ai"],
            )
            if result:
                DevelopmentSwarm._hackathon_state["colosseum_posts"].append({
                    "title": title,
                    "post_id": result.get("post", {}).get("id"),
                    "timestamp": datetime.now().isoformat(),
                })
                DevelopmentSwarm._hackathon_state["colosseum_posts"] = DevelopmentSwarm._hackathon_state["colosseum_posts"][-30:]
                logger.info(f"[{self.swarm_id}] Posted hackathon update to Colosseum")
            await worker.close()
        except Exception as e:
            logger.debug(f"Colosseum post failed (non-critical): {e}")

    @classmethod
    def get_active_swarms(cls) -> dict[str, 'DevelopmentSwarm']:
        """Get all currently active development swarms."""
        return cls._active_swarms.copy()

    @classmethod
    def get_stats(cls) -> dict:
        """Get development swarm statistics."""
        return {
            "max_concurrent": cls.MAX_CONCURRENT_SWARMS,
            "active_count": len(cls._active_swarms),
            "active_swarms": [
                {
                    "id": s.swarm_id,
                    "task": s.task_description[:50],
                    "status": s.status,
                    "started": s.started_at.isoformat() if s.started_at else None
                }
                for s in cls._active_swarms.values()
            ]
        }
