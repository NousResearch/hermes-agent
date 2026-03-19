"""
Cosmos Skill Registry - Dynamic Tool/Skill Discovery & Routing

Provides a centralized registry of ALL capabilities across the swarm:
- Shadow agent capabilities (grok, gemini, claude, etc.)
- Integration tools (X posting, image gen, Solana, Polymarket)
- Hackathon tools (oracle, trading, rug detection)
- HermesAgent compatibility skills
- Custom user-defined skills
- MCP tools

Usage:
    from Cosmos.core.skill_registry import get_skill_registry
    registry = get_skill_registry()

    # Find skills matching a need
    skills = registry.find_skills("post to twitter with image")

    # Get all skills for an agent
    skills = registry.get_agent_skills("grok")

    # Register a new skill
    registry.register_skill(Skill(...))

    # Auto-discover all available skills
    await registry.auto_discover()
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, set, tuple

from loguru import logger


class SkillCategory(str, Enum):
    """Categories of skills available in the swarm."""
    SOCIAL = "social"           # X posting, engagement, meme gen
    TRADING = "trading"         # Solana swaps, DeFi, sniping
    ANALYSIS = "analysis"       # Token analysis, rug detection, whale watching
    PREDICTION = "prediction"   # Polymarket, FarsightProtocol, oracle
    MEDIA = "media"             # Image gen, video gen, TTS, VTuber
    MEMORY = "memory"           # Store, recall, knowledge graph
    DEVELOPMENT = "development" # Code gen, audit, evolution
    COMMUNICATION = "communication"  # Discord, Slack, WhatsApp, etc.
    QUANTUM = "quantum"         # IBM Quantum circuits
    BLOCKCHAIN = "blockchain"   # On-chain operations
    RESEARCH = "research"       # Web search, deep search, trend analysis
    ORCHESTRATION = "orchestration"  # Team management, delegation
    UTILITY = "utility"         # General utilities
    CUSTOM = "custom"           # User-defined or Hermes skills


@dataclass
class Skill:
    """A registered skill/tool in the swarm."""
    name: str
    description: str
    category: SkillCategory
    module_path: str  # e.g. "cosmos.integration.x_automation.x_engagement_poster"
    function_name: str  # e.g. "execute"
    agents: list[str] = field(default_factory=list)  # Which agents can run this
    keywords: list[str] = field(default_factory=list)  # Search keywords
    parameters: dict[str, str] = field(default_factory=dict)  # param_name: description
    requires_api_key: Optional[str] = None  # Env var name if API key needed
    cooldown_seconds: int = 0
    enabled: bool = True
    source: str = "builtin"  # builtin, HermesAgent, custom, mcp
    last_used: Optional[str] = None
    usage_count: int = 0
    success_rate: float = 1.0
    # Identity system fields
    execution_guidelines: Optional[str] = None  # How to execute this skill
    agent_guidance: Optional[str] = None  # Persona guidance when using this skill
    output_format: Optional[str] = None  # Expected output format


class SkillRegistry:
    """Central registry of all swarm capabilities."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._categories: dict[SkillCategory[str]] = {cat: [] for cat in SkillCategory}
        self._agent_skills: dict[str[str]] = {}
        self._discovered = False
        self._persistence_path = Path(os.environ.get(
            "SKILL_REGISTRY_PATH",
            "/data/skill_registry.json" if os.path.exists("/data") else "skill_registry.json"
        ))

    def register_skill(self, skill: Skill) -> None:
        """Register a skill in the registry."""
        self._skills[skill.name] = skill

        # Index by category
        if skill.name not in self._categories[skill.category]:
            self._categories[skill.category].append(skill.name)

        # Index by agent
        for agent in skill.agents:
            if agent not in self._agent_skills:
                self._agent_skills[agent] = []
            if skill.name not in self._agent_skills[agent]:
                self._agent_skills[agent].append(skill.name)

        logger.debug(f"Registered skill: {skill.name} [{skill.category.value}]")

    def find_skills(
        self,
        query: str,
        category: Optional[SkillCategory] = None,
        agent: Optional[str] = None,
        enabled_only: bool = True,
    ) -> list[Skill]:
        """Find skills matching a natural language query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []

        for skill in self._skills.values():
            if enabled_only and not skill.enabled:
                continue
            if category and skill.category != category:
                continue
            if agent and agent not in skill.agents:
                continue

            # Score by relevance
            score = 0

            # Name match
            if query_lower in skill.name.lower():
                score += 10

            # Description match
            desc_lower = skill.description.lower()
            for word in query_words:
                if word in desc_lower:
                    score += 3

            # Keyword match
            for kw in skill.keywords:
                if kw.lower() in query_lower or query_lower in kw.lower():
                    score += 5
                for word in query_words:
                    if word in kw.lower():
                        score += 2

            if score > 0:
                results.append((score, skill))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in results]

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a specific skill by name."""
        return self._skills.get(name)

    def get_agent_skills(self, agent: str) -> list[Skill]:
        """Get all skills available to a specific agent."""
        skill_names = self._agent_skills.get(agent, [])
        return [self._skills[n] for n in skill_names if n in self._skills]

    def get_category_skills(self, category: SkillCategory) -> list[Skill]:
        """Get all skills in a category."""
        skill_names = self._categories.get(category, [])
        return [self._skills[n] for n in skill_names if n in self._skills]

    def list_all_skills(self) -> list[Skill]:
        """list all registered skills."""
        return list(self._skills.values())

    def get_skill_summary(self) -> dict:
        """Get a summary of all registered skills for agent prompts."""
        summary = {
            "total_skills": len(self._skills),
            "categories": {},
            "agents": {},
        }

        for cat in SkillCategory:
            skills = self.get_category_skills(cat)
            if skills:
                summary["categories"][cat.value] = [
                    {"name": s.name, "description": s.description[:100]}
                    for s in skills
                ]

        for agent, skill_names in self._agent_skills.items():
            summary["agents"][agent] = len(skill_names)

        return summary

    def get_prompt_context(self, agent: Optional[str] = None, verbose: bool = False) -> str:
        """
        Generate a context string for injection into agent prompts.

        Args:
            agent: Optional agent name to filter skills for
            verbose: If True, include execution_guidelines and agent_guidance
        """
        if agent:
            skills = self.get_agent_skills(agent)
        else:
            skills = self.list_all_skills()

        if not skills:
            return "No skills currently registered."

        lines = ["AVAILABLE SKILLS/TOOLS:"]

        # Group by category
        by_cat: dict[str[Skill]] = {}
        for s in skills:
            cat = s.category.value
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(s)

        for cat, cat_skills in sorted(by_cat.items()):
            lines.append(f"\n[{cat.upper()}]")
            for s in cat_skills:
                params = ", ".join(f"{k}: {v}" for k, v in s.parameters.items()) if s.parameters else "none"
                lines.append(f"  - {s.name}: {s.description} (params: {params})")
                if verbose:
                    if s.execution_guidelines:
                        lines.append(f"    Guidelines: {s.execution_guidelines}")
                    if s.agent_guidance:
                        lines.append(f"    Guidance: {s.agent_guidance}")
                    if s.output_format:
                        lines.append(f"    Output: {s.output_format}")

        return "\n".join(lines)

    def record_usage(self, skill_name: str, success: bool = True) -> None:
        """Record that a skill was used."""
        skill = self._skills.get(skill_name)
        if skill:
            skill.usage_count += 1
            skill.last_used = datetime.utcnow().isoformat()
            # Update success rate with exponential moving average
            alpha = 0.1
            skill.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * skill.success_rate

    async def auto_discover(self) -> int:
        """Auto-discover all available skills from installed modules."""
        if self._discovered:
            return len(self._skills)

        logger.info("Auto-discovering skills...")
        count_before = len(self._skills)

        # Register all builtin skills
        self._register_builtin_skills()

        # Discover Hermes skills via shadow layer
        await self._discover_HermesAgent_skills()

        # Discover MCP tools from registered servers
        await self._discover_mcp_tools()

        # Discover model-specific tools (tools only certain models can use)
        self._register_model_specific_tools()

        # Try loading persisted custom skills
        self._load_persisted_skills()

        self._discovered = True
        new_count = len(self._skills) - count_before
        logger.info(f"Discovered {new_count} new skills. Total: {len(self._skills)}")
        return len(self._skills)

    async def _discover_HermesAgent_skills(self) -> int:
        """Discover and register Hermes skills via the compatibility layer."""
        count = 0
        try:
            from Cosmos.compatibility.hermes_adapter import (
                HermesAgentAdapter,
                HermesAgentToolGroup,
            )

            adapter = HermesAgentAdapter()

            # Map HermesAgent tool groups to our categories
            group_to_category = {
                HermesAgentToolGroup.FILESYSTEM: SkillCategory.UTILITY,
                HermesAgentToolGroup.RUNTIME: SkillCategory.DEVELOPMENT,
                HermesAgentToolGroup.SESSIONS: SkillCategory.ORCHESTRATION,
                HermesAgentToolGroup.MEMORY: SkillCategory.MEMORY,
                HermesAgentToolGroup.WEB: SkillCategory.RESEARCH,
                HermesAgentToolGroup.UI: SkillCategory.MEDIA,
                HermesAgentToolGroup.AUTOMATION: SkillCategory.UTILITY,
                HermesAgentToolGroup.MESSAGING: SkillCategory.COMMUNICATION,
                HermesAgentToolGroup.NODES: SkillCategory.UTILITY,
            }

            # Register each HermesAgent tool group as skills
            for group in HermesAgentToolGroup:
                category = group_to_category.get(group, SkillCategory.CUSTOM)
                tools = adapter.get_tools_for_group(group) if hasattr(adapter, 'get_tools_for_group') else []

                if tools:
                    for tool in tools:
                        tool_name = f"HermesAgent_{group.name.lower()}_{tool.get('name', 'unknown')}"
                        self.register_skill(Skill(
                            name=tool_name,
                            description=tool.get("description", f"HermesAgent {group.name} tool"),
                            category=category,
                            module_path="cosmos.compatibility.hermes_adapter",
                            function_name="invoke_tool",
                            agents=["cosmos", "claude", "grok"],
                            keywords=["HermesAgent", group.name.lower()] + tool.get("keywords", []),
                            parameters=tool.get("parameters", {}),
                            source="HermesAgent",
                        ))
                        count += 1
                else:
                    # Register the group itself as an invocable skill
                    self.register_skill(Skill(
                        name=f"HermesAgent_{group.name.lower()}",
                        description=f"HermesAgent {group.name} tools ({group.value})",
                        category=category,
                        module_path="cosmos.compatibility.hermes_adapter",
                        function_name="invoke",
                        agents=["cosmos", "claude", "grok"],
                        keywords=["HermesAgent", group.name.lower(), "compatibility", "shadow"],
                        parameters={"tool": "Tool name", "action": "Action", "params": "Parameters"},
                        source="HermesAgent",
                    ))
                    count += 1

            # Try to discover HermesHub marketplace skills
            if hasattr(adapter, 'HermesHub_client') and adapter.HermesHub_client:
                try:
                    hub_skills = await adapter.HermesHub_client.list_skills() if hasattr(adapter.HermesHub_client, 'list_skills') else []
                    for hub_skill in hub_skills[:50]:  # Cap at 50 marketplace skills
                        self.register_skill(Skill(
                            name=f"HermesHub_{hub_skill.get('id', 'unknown')}",
                            description=hub_skill.get("description", "HermesHub community skill"),
                            category=SkillCategory.CUSTOM,
                            module_path="cosmos.compatibility.hermes_adapter",
                            function_name="invoke_HermesHub_skill",
                            agents=["cosmos"],
                            keywords=["HermesHub", "community"] + hub_skill.get("tags", []),
                            parameters=hub_skill.get("inputs", {}),
                            source="HermesAgent",
                        ))
                        count += 1
                except Exception as e:
                    logger.debug(f"HermesHub discovery skipped: {e}")

            logger.info(f"Discovered {count} Hermes skills")
        except ImportError:
            logger.debug("HermesAgent adapter not available")
        except Exception as e:
            logger.warning(f"Hermes skill discovery failed: {e}")

        return count

    async def _discover_mcp_tools(self) -> int:
        """Discover and register MCP (Model Context Protocol) tools."""
        count = 0
        try:
            # Discover from Cosmos's MCP server
            from Cosmos.mcp_server import CosmosMCPServer
            mcp = CosmosMCPServer()

            # Register MCP memory tools
            mcp_tools = [
                ("mcp_memory_store", "Store data via MCP memory tool", SkillCategory.MEMORY,
                 "MemoryTools", "store", ["mcp", "memory", "store"]),
                ("mcp_memory_recall", "Recall data via MCP memory tool", SkillCategory.MEMORY,
                 "MemoryTools", "recall", ["mcp", "memory", "recall"]),
                ("mcp_agent_delegate", "Delegate task to agent via MCP", SkillCategory.ORCHESTRATION,
                 "AgentTools", "delegate", ["mcp", "agent", "delegate"]),
                ("mcp_agent_status", "Get agent status via MCP", SkillCategory.ORCHESTRATION,
                 "AgentTools", "status", ["mcp", "agent", "status"]),
                ("mcp_evolution_feedback", "Submit evolution feedback via MCP", SkillCategory.DEVELOPMENT,
                 "EvolutionTools", "feedback", ["mcp", "evolution", "feedback"]),
                ("mcp_evolution_trigger", "Trigger evolution cycle via MCP", SkillCategory.DEVELOPMENT,
                 "EvolutionTools", "trigger", ["mcp", "evolution", "trigger"]),
            ]

            for name, desc, cat, class_name, func, keywords in mcp_tools:
                self.register_skill(Skill(
                    name=name,
                    description=desc,
                    category=cat,
                    module_path=f"cosmos.mcp_server.{class_name.lower().replace('tools', '_tools')}",
                    function_name=func,
                    agents=["cosmos", "claude"],
                    keywords=keywords,
                    source="mcp",
                ))
                count += 1

            # Discover from Claude Teams MCP bridge
            try:
                from Cosmos.integration.claude_teams.mcp_bridge import CosmosMCPServer as TeamsMCP
                teams_mcp = TeamsMCP()
                if hasattr(teams_mcp, 'list_tools'):
                    tools = teams_mcp.list_tools()
                    for tool in tools:
                        tool_name = f"mcp_teams_{tool.get('name', 'unknown')}"
                        if tool_name not in self._skills:
                            self.register_skill(Skill(
                                name=tool_name,
                                description=tool.get("description", "Claude Teams MCP tool"),
                                category=SkillCategory.ORCHESTRATION,
                                module_path="cosmos.integration.claude_teams.mcp_bridge",
                                function_name=tool.get("name", "invoke"),
                                agents=["cosmos", "claude"],
                                keywords=["mcp", "claude", "teams"] + tool.get("keywords", []),
                                parameters=tool.get("inputSchema", {}).get("properties", {}),
                                source="mcp",
                            ))
                            count += 1
            except ImportError:
                logger.debug("Claude Teams MCP bridge not available")

            # Discover external MCP servers from environment
            mcp_servers_json = os.environ.get("MCP_SERVERS", "")
            if mcp_servers_json:
                try:
                    servers = json.loads(mcp_servers_json)
                    for server_config in servers:
                        server_name = server_config.get("name", "unknown")
                        for tool in server_config.get("tools", []):
                            self.register_skill(Skill(
                                name=f"mcp_ext_{server_name}_{tool['name']}",
                                description=tool.get("description", f"External MCP tool from {server_name}"),
                                category=SkillCategory.CUSTOM,
                                module_path=f"mcp://{server_name}",
                                function_name=tool["name"],
                                agents=["cosmos"],
                                keywords=["mcp", "external", server_name],
                                parameters=tool.get("inputSchema", {}).get("properties", {}),
                                source="mcp",
                            ))
                            count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"External MCP server discovery failed: {e}")

            logger.info(f"Discovered {count} MCP tools")
        except ImportError:
            logger.debug("MCP server module not available")
        except Exception as e:
            logger.warning(f"MCP tool discovery failed: {e}")

        return count

    def _register_model_specific_tools(self) -> None:
        """Register tools that are specific to certain AI models/providers."""

        # Grok-specific: web search, image generation, video generation
        grok_tools = [
            ("grok_deep_search", "Grok deep web search with real-time results",
             SkillCategory.RESEARCH, "deep_search", ["search", "web", "real-time", "live"]),
            ("grok_image_gen", "Generate images using Grok Imagine (grok-2-image)",
             SkillCategory.MEDIA, "generate_image", ["image", "art", "grok", "generate"]),
            ("grok_video_gen", "Generate video from image using Grok Imagine Video",
             SkillCategory.MEDIA, "generate_video", ["video", "animate", "grok"]),
        ]
        for name, desc, cat, func, kw in grok_tools:
            if name not in self._skills:
                self.register_skill(Skill(
                    name=name, description=desc, category=cat,
                    module_path="cosmos.integration.external.grok",
                    function_name=func, agents=["grok"],
                    keywords=kw, requires_api_key="GROK_API_KEY", source="model_specific",
                ))

        # Gemini-specific: multimodal vision, reference image generation
        gemini_tools = [
            ("gemini_vision", "Gemini multimodal vision analysis (images, charts, screenshots)",
             SkillCategory.ANALYSIS, "analyze_image", ["vision", "image", "analyze", "multimodal"]),
            ("gemini_reference_gen", "Generate images with character consistency using reference images",
             SkillCategory.MEDIA, "generate_with_reference", ["reference", "consistent", "character"]),
            ("gemini_image_edit", "Edit existing images with Gemini instructions",
             SkillCategory.MEDIA, "edit_image", ["edit", "modify", "image", "change"]),
        ]
        for name, desc, cat, func, kw in gemini_tools:
            if name not in self._skills:
                self.register_skill(Skill(
                    name=name, description=desc, category=cat,
                    module_path="cosmos.integration.external.gemini",
                    function_name=func, agents=["gemini"],
                    keywords=kw, requires_api_key="GEMINI_API_KEY", source="model_specific",
                ))

        # Kimi-specific: ultra-long context analysis
        self.register_skill(Skill(
            name="kimi_long_context",
            description="Kimi K2.5 256K context analysis for very long documents or codebases",
            category=SkillCategory.RESEARCH,
            module_path="cosmos.integration.external.kimi",
            function_name="chat",
            agents=["kimi"],
            keywords=["long", "context", "document", "256k", "analysis", "codebase"],
            parameters={"prompt": "Analysis prompt", "context": "Long document/code"},
            requires_api_key="KIMI_API_KEY",
            source="model_specific",
        ))

        # DeepSeek-specific: algorithm and optimization
        self.register_skill(Skill(
            name="deepseek_algorithm",
            description="DeepSeek R1 specialized algorithm implementation and math reasoning",
            category=SkillCategory.DEVELOPMENT,
            module_path="cosmos.core.collective.persistent_agent",
            function_name="call_shadow_agent",
            agents=["deepseek"],
            keywords=["algorithm", "math", "optimize", "deepseek", "reasoning"],
            parameters={"prompt": "Algorithm/math problem"},
            source="model_specific",
        ))

        # HuggingFace-specific: local embeddings and inference
        hf_tools = [
            ("hf_embeddings", "Generate semantic embeddings using local HuggingFace models",
             SkillCategory.MEMORY, "generate_embeddings", ["embeddings", "semantic", "vector", "similarity"]),
            ("hf_code_gen", "Generate code using local CodeLlama/StarCoder models",
             SkillCategory.DEVELOPMENT, "code_generate", ["code", "local", "codellama", "starcoder"]),
            ("hf_local_inference", "Run local GPU inference without API keys",
             SkillCategory.UTILITY, "chat", ["local", "gpu", "inference", "free", "offline"]),
        ]
        for name, desc, cat, func, kw in hf_tools:
            if name not in self._skills:
                self.register_skill(Skill(
                    name=name, description=desc, category=cat,
                    module_path="cosmos.integration.external.huggingface",
                    function_name=func, agents=["huggingface"],
                    keywords=kw, source="model_specific",
                ))

        # Claude-specific: safety analysis, code review
        self.register_skill(Skill(
            name="claude_safety_review",
            description="Claude specialized safety and security code review",
            category=SkillCategory.DEVELOPMENT,
            module_path="cosmos.integration.external.claude",
            function_name="chat",
            agents=["claude"],
            keywords=["safety", "security", "review", "vulnerability", "audit"],
            parameters={"prompt": "Code to review for safety"},
            source="model_specific",
        ))

        # ClaudeOpus-specific: complex architecture
        self.register_skill(Skill(
            name="claude_opus_architect",
            description="Claude Opus 4.6 for complex multi-file architectural redesigns",
            category=SkillCategory.DEVELOPMENT,
            module_path="cosmos.integration.claude_teams.agent_sdk_bridge",
            function_name="delegate",
            agents=["claude_opus"],
            keywords=["architecture", "redesign", "complex", "opus", "refactor"],
            parameters={"task": "Architectural task description"},
            source="model_specific",
        ))

        logger.info("Registered model-specific tools")

    def _register_builtin_skills(self) -> None:
        """Register all built-in Cosmos skills."""

        all_agents = ["grok", "gemini", "kimi", "claude", "deepseek", "phi", "huggingface", "swarm_mind"]
        api_agents = ["grok", "gemini", "kimi", "claude"]

        # ===================== SOCIAL =====================
        self.register_skill(Skill(
            name="mega_thread_poster",
            description="Post a 20+ tweet mega thread with images, trending topics, and one hashtag per post",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.x_automation.x_engagement_poster",
            function_name="execute",
            agents=["cosmos", "grok"],
            keywords=["tweet", "thread", "post", "X", "twitter", "mega", "article", "engagement"],
            parameters={"topic": "Optional topic override", "generate_images": "bool", "delay": "seconds between posts"},
        ))

        self.register_skill(Skill(
            name="post_tweet",
            description="Post a single tweet to X/Twitter with optional image",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.x_automation.x_api_poster",
            function_name="post_tweet",
            agents=["cosmos", "grok"],
            keywords=["tweet", "post", "X", "twitter"],
            parameters={"text": "Tweet text", "image_bytes": "Optional image"},
        ))

        self.register_skill(Skill(
            name="post_reply",
            description="Reply to a tweet on X/Twitter",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.x_automation.x_api_poster",
            function_name="post_reply",
            agents=["cosmos", "grok"],
            keywords=["reply", "respond", "tweet", "thread"],
            parameters={"text": "Reply text", "reply_to_id": "Tweet ID to reply to"},
        ))

        self.register_skill(Skill(
            name="generate_meme",
            description="Generate a Borg Cosmos meme image with AI",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.image_gen.generator",
            function_name="generate_borg_cosmos_meme",
            agents=["cosmos", "gemini", "grok"],
            keywords=["meme", "image", "borg", "cosmos", "picture"],
        ))

        self.register_skill(Skill(
            name="get_trending_topics",
            description="Fetch top 20 trending topics on X/Twitter",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.x_automation.x_engagement_poster",
            function_name="get_trending_topics",
            agents=["grok", "gemini"],
            keywords=["trending", "topics", "hashtag", "viral", "popular"],
        ))

        self.register_skill(Skill(
            name="check_mentions",
            description="Check and reply to X/Twitter mentions",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.x_automation.reply_bot",
            function_name="process_mentions",
            agents=["cosmos"],
            keywords=["mentions", "replies", "engage", "respond"],
        ))

        # ===================== TRADING =====================
        self.register_skill(Skill(
            name="jupiter_swap",
            description="Execute a token swap on Solana via Jupiter V6 aggregator",
            category=SkillCategory.TRADING,
            module_path="cosmos.integration.solana.trading",
            function_name="jupiter_swap",
            agents=["cosmos", "deepseek"],
            keywords=["swap", "trade", "jupiter", "solana", "token", "buy", "sell"],
            parameters={"input_mint": "Token to sell", "output_mint": "Token to buy", "amount": "Amount in lamports"},
            requires_api_key="SOLANA_PRIVATE_KEY",
        ))

        self.register_skill(Skill(
            name="pump_fun_trade",
            description="Trade tokens on Pump.fun via PumpPortal",
            category=SkillCategory.TRADING,
            module_path="cosmos.integration.solana.trading",
            function_name="pump_fun_trade",
            agents=["cosmos"],
            keywords=["pump", "pumpfun", "launch", "trade", "bonding"],
            requires_api_key="SOLANA_PRIVATE_KEY",
        ))

        self.register_skill(Skill(
            name="jito_bundle",
            description="Send a Jito bundle for anti-MEV execution on Solana",
            category=SkillCategory.TRADING,
            module_path="cosmos.integration.solana.degen_mob",
            function_name="send_jito_bundle",
            agents=["cosmos"],
            keywords=["jito", "mev", "bundle", "frontrun", "sandwich"],
            parameters={"transactions": "list of transactions", "tip_sol": "Jito tip amount"},
        ))

        # ===================== ANALYSIS =====================
        self.register_skill(Skill(
            name="token_analysis",
            description="Multi-agent Solana token analysis with swarm consensus",
            category=SkillCategory.ANALYSIS,
            module_path="cosmos.integration.solana.swarm_solana",
            function_name="analyze_token",
            agents=all_agents,
            keywords=["token", "analyze", "scan", "risk", "solana", "contract"],
            parameters={"token_address": "Solana token mint address"},
        ))

        self.register_skill(Skill(
            name="rug_detection",
            description="Detect potential rug pulls by checking mint/freeze authorities",
            category=SkillCategory.ANALYSIS,
            module_path="cosmos.integration.solana.degen_mob",
            function_name="analyze_token_safety",
            agents=["cosmos", "deepseek", "grok"],
            keywords=["rug", "scam", "safety", "audit", "honeypot"],
            parameters={"mint_address": "Token mint address"},
        ))

        self.register_skill(Skill(
            name="whale_watch",
            description="Track whale wallet movements and detect insider rings",
            category=SkillCategory.ANALYSIS,
            module_path="cosmos.integration.solana.degen_mob",
            function_name="get_whale_recent_activity",
            agents=["cosmos", "grok"],
            keywords=["whale", "wallet", "track", "insider", "large", "holder"],
            parameters={"wallet_address": "Wallet to monitor"},
        ))

        self.register_skill(Skill(
            name="wallet_analysis",
            description="Swarm assessment of a Solana wallet",
            category=SkillCategory.ANALYSIS,
            module_path="cosmos.integration.solana.swarm_solana",
            function_name="analyze_wallet",
            agents=all_agents,
            keywords=["wallet", "analyze", "portfolio", "holdings"],
            parameters={"wallet_address": "Solana wallet address"},
        ))

        self.register_skill(Skill(
            name="defi_recommend",
            description="Get DeFi strategy recommendations from the swarm",
            category=SkillCategory.ANALYSIS,
            module_path="cosmos.integration.solana.swarm_solana",
            function_name="get_defi_recommendation",
            agents=all_agents,
            keywords=["defi", "yield", "farm", "strategy", "apy", "invest"],
            parameters={"amount": "USD amount", "risk_tolerance": "low/medium/high", "goal": "Investment goal"},
        ))

        # ===================== PREDICTION =====================
        self.register_skill(Skill(
            name="swarm_oracle",
            description="Multi-agent oracle query with on-chain consensus recording",
            category=SkillCategory.PREDICTION,
            module_path="cosmos.integration.solana.swarm_oracle",
            function_name="submit_query",
            agents=all_agents,
            keywords=["oracle", "predict", "consensus", "question", "on-chain"],
            parameters={"question": "Question to ask the oracle", "query_type": "Type of query"},
        ))

        self.register_skill(Skill(
            name="farsight_predict",
            description="5-source prediction (Swarm + Polymarket + Monte Carlo + Quantum + Vision)",
            category=SkillCategory.PREDICTION,
            module_path="cosmos.integration.hackathon.farsight_protocol",
            function_name="predict",
            agents=["cosmos", "grok", "gemini"],
            keywords=["predict", "forecast", "probability", "farsight", "future"],
            parameters={"question": "Prediction question"},
        ))

        self.register_skill(Skill(
            name="polymarket_predictions",
            description="Get AGI-level collective predictions on Polymarket events",
            category=SkillCategory.PREDICTION,
            module_path="cosmos.core.polymarket_predictor",
            function_name="get_predictions",
            agents=["cosmos", "grok", "gemini", "kimi", "deepseek"],
            keywords=["polymarket", "prediction", "market", "bet", "probability"],
        ))

        # ===================== MEDIA =====================
        self.register_skill(Skill(
            name="generate_image",
            description="Generate an AI image using Gemini or Grok",
            category=SkillCategory.MEDIA,
            module_path="cosmos.integration.image_gen.generator",
            function_name="generate",
            agents=["gemini", "grok"],
            keywords=["image", "picture", "generate", "art", "visual"],
            parameters={"prompt": "Image description", "prefer": "gemini or grok"},
        ))

        self.register_skill(Skill(
            name="generate_video",
            description="Generate an animated video from an image using Grok Imagine Video",
            category=SkillCategory.MEDIA,
            module_path="cosmos.integration.image_gen.generator",
            function_name="generate_borg_cosmos_video",
            agents=["grok"],
            keywords=["video", "animate", "animation", "clip"],
            parameters={"scene": "Scene description"},
            requires_api_key="XAI_API_KEY",
        ))

        self.register_skill(Skill(
            name="text_to_speech",
            description="Generate speech audio with cloned bot voice",
            category=SkillCategory.MEDIA,
            module_path="cosmos.integration.multi_voice",
            function_name="generate_speech",
            agents=["cosmos"],
            keywords=["tts", "speak", "voice", "audio", "speech"],
            parameters={"text": "Text to speak", "bot_name": "Bot voice to use"},
        ))

        # ===================== MEMORY =====================
        self.register_skill(Skill(
            name="remember",
            description="Store information in swarm memory",
            category=SkillCategory.MEMORY,
            module_path="cosmos.memory.memory_system",
            function_name="store",
            agents=all_agents,
            keywords=["remember", "store", "save", "memory", "note"],
            parameters={"content": "What to remember", "tags": "Optional tags"},
        ))

        self.register_skill(Skill(
            name="recall",
            description="Recall information from swarm memory using semantic search",
            category=SkillCategory.MEMORY,
            module_path="cosmos.memory.memory_system",
            function_name="recall",
            agents=all_agents,
            keywords=["recall", "search", "find", "retrieve", "memory"],
            parameters={"query": "What to search for"},
        ))

        self.register_skill(Skill(
            name="knowledge_graph_query",
            description="Query the knowledge graph for entity relationships",
            category=SkillCategory.MEMORY,
            module_path="cosmos.memory.knowledge_graph",
            function_name="query",
            agents=all_agents,
            keywords=["knowledge", "graph", "entity", "relationship", "connection"],
            parameters={"query": "Entity or relationship to look up"},
        ))

        # ===================== QUANTUM =====================
        self.register_skill(Skill(
            name="quantum_bell_state",
            description="Run a Bell state circuit on IBM Quantum hardware",
            category=SkillCategory.QUANTUM,
            module_path="cosmos.integration.hackathon.quantum_proof",
            function_name="run_bell_state",
            agents=["cosmos"],
            keywords=["quantum", "bell", "entanglement", "qubit", "ibm"],
            requires_api_key="IBM_QUANTUM_TOKEN",
        ))

        self.register_skill(Skill(
            name="quantum_random",
            description="Generate quantum random numbers from real hardware",
            category=SkillCategory.QUANTUM,
            module_path="cosmos.integration.hackathon.quantum_proof",
            function_name="run_quantum_random",
            agents=["cosmos"],
            keywords=["quantum", "random", "entropy", "rng"],
            requires_api_key="IBM_QUANTUM_TOKEN",
        ))

        # ===================== RESEARCH =====================
        self.register_skill(Skill(
            name="web_search",
            description="Search the web using Grok deep search",
            category=SkillCategory.RESEARCH,
            module_path="cosmos.integration.external.grok",
            function_name="deep_search",
            agents=["grok"],
            keywords=["search", "web", "research", "find", "lookup", "google"],
            parameters={"query": "Search query"},
            requires_api_key="GROK_API_KEY",
        ))

        self.register_skill(Skill(
            name="swarm_deliberation",
            description="Run full PROPOSE-CRITIQUE-REFINE-VOTE deliberation across all agents",
            category=SkillCategory.RESEARCH,
            module_path="cosmos.core.collective.deliberation",
            function_name="deliberate",
            agents=all_agents,
            keywords=["deliberate", "discuss", "consensus", "vote", "collective"],
            parameters={"topic": "Topic to deliberate on", "session_type": "website_chat/grok_thread/autonomous_task"},
            execution_guidelines="Each agent proposes independently, then critiques others, refines with feedback, and votes. Use for complex questions needing multiple perspectives.",
            agent_guidance="Contribute your unique expertise. Be constructive in critiques. Synthesize best ideas in refinement.",
        ))

        self.register_skill(Skill(
            name="prompt_upgrade",
            description="Auto-enhance a vague user prompt using Grok/Gemini",
            category=SkillCategory.RESEARCH,
            module_path="cosmos.core.prompt_upgrader",
            function_name="upgrade_prompt",
            agents=["grok", "gemini"],
            keywords=["prompt", "enhance", "upgrade", "improve", "rewrite"],
            parameters={"prompt": "The vague prompt to enhance"},
        ))

        # ===================== DEVELOPMENT =====================
        self.register_skill(Skill(
            name="code_generate",
            description="Generate code using the development swarm",
            category=SkillCategory.DEVELOPMENT,
            module_path="cosmos.core.development_swarm",
            function_name="execute_task",
            agents=["claude", "deepseek", "grok", "kimi"],
            keywords=["code", "generate", "build", "implement", "develop", "program"],
            parameters={"task": "What to build"},
            execution_guidelines="Research → Plan → Implement → Audit pipeline. Multiple models write code in parallel, best is selected.",
            agent_guidance="Write production-quality Python with type hints, docstrings, and error handling. Follow existing Cosmos patterns.",
            output_format="Complete runnable Python files with # filename: headers",
        ))

        self.register_skill(Skill(
            name="code_audit",
            description="Security and quality audit of generated code",
            category=SkillCategory.DEVELOPMENT,
            module_path="cosmos.core.evolution_loop",
            function_name="_audit_code",
            agents=["grok", "claude"],
            keywords=["audit", "review", "security", "quality", "check"],
            parameters={"code": "Code to audit"},
            execution_guidelines="Check for injection vulnerabilities, auth issues, data exposure, input validation. Rate: APPROVE, APPROVE_WITH_FIXES, or REJECT.",
            agent_guidance="Be thorough and specific. Reference line numbers. Focus on security first, then quality.",
        ))

        self.register_skill(Skill(
            name="evolution_spawn",
            description="Spawn an evolution worker to evolve bot personality",
            category=SkillCategory.DEVELOPMENT,
            module_path="cosmos.core.collective.evolution",
            function_name="evolve",
            agents=["cosmos"],
            keywords=["evolution", "evolve", "personality", "genetics", "fitness"],
        ))

        # ===================== COMMUNICATION =====================
        for channel in ["discord", "slack", "whatsapp", "signal", "matrix", "imessage", "webchat"]:
            self.register_skill(Skill(
                name=f"send_{channel}",
                description=f"Send a message via {channel.title()}",
                category=SkillCategory.COMMUNICATION,
                module_path=f"cosmos.integration.channels.{channel}_adapter",
                function_name="send_message",
                agents=["cosmos"],
                keywords=[channel, "message", "send", "chat", "notify"],
                parameters={"message": "Message to send", "channel_id": "Target channel/user"},
            ))

        # ===================== ORCHESTRATION =====================
        self.register_skill(Skill(
            name="delegate_to_claude",
            description="Delegate a task to a Claude team (research, coding, analysis, etc.)",
            category=SkillCategory.ORCHESTRATION,
            module_path="cosmos.integration.claude_teams.swarm_team_fusion",
            function_name="delegate",
            agents=["cosmos"],
            keywords=["delegate", "claude", "team", "assign", "task"],
            parameters={"task": "Task description", "delegation_type": "RESEARCH/CODING/ANALYSIS/etc"},
        ))

        self.register_skill(Skill(
            name="create_claude_team",
            description="Create a Claude agent team for complex tasks",
            category=SkillCategory.ORCHESTRATION,
            module_path="cosmos.integration.claude_teams.swarm_team_fusion",
            function_name="delegate_to_team",
            agents=["cosmos"],
            keywords=["team", "create", "claude", "agents", "collaborate"],
            parameters={"task": "Task", "team_name": "Name", "roles": "list of roles"},
        ))

        self.register_skill(Skill(
            name="spawn_agent_instance",
            description="Spawn a new agent instance for parallel task execution",
            category=SkillCategory.ORCHESTRATION,
            module_path="cosmos.core.agent_spawner",
            function_name="spawn_instance",
            agents=["cosmos"],
            keywords=["spawn", "agent", "instance", "parallel", "worker"],
            parameters={"agent_name": "Agent to spawn", "task_type": "chat/dev/research/memory/mcp/testing/audit"},
        ))

        # ===================== SECURITY & GATEWAY =====================
        self.register_skill(Skill(
            name="gateway_query",
            description="Send a query through The Window (External Gateway) - sandboxed, rate-limited, secret-scrubbed endpoint for external agents",
            category=SkillCategory.COMMUNICATION,
            module_path="cosmos.core.external_gateway",
            function_name="handle_request",
            agents=all_agents,
            keywords=["gateway", "window", "external", "query", "sandbox", "api", "public"],
            parameters={"input_text": "The query to send", "client_ip": "Client IP address"},
        ))

        self.register_skill(Skill(
            name="injection_defense_analyze",
            description="Analyze input text through the 5-layer injection defense system (structural, semantic, behavioral, canary, collective)",
            category=SkillCategory.UTILITY,
            module_path="cosmos.core.security.injection_defense",
            function_name="analyze",
            agents=all_agents,
            keywords=["security", "injection", "defense", "analyze", "threat", "safe", "scan"],
            parameters={"input_text": "Text to analyze", "client_id": "Optional client identifier"},
        ))

        self.register_skill(Skill(
            name="token_orchestrator_dashboard",
            description="Get real-time token usage dashboard - per-agent budgets, tandem sessions, efficiency metrics",
            category=SkillCategory.ORCHESTRATION,
            module_path="cosmos.core.token_orchestrator",
            function_name="get_dashboard",
            agents=all_agents,
            keywords=["tokens", "budget", "orchestrator", "usage", "efficiency", "dashboard", "cost"],
        ))

        self.register_skill(Skill(
            name="start_tandem_session",
            description="Start a Grok+Kimi tandem session - Grok for real-time data, Kimi for synthesis and reasoning",
            category=SkillCategory.ORCHESTRATION,
            module_path="cosmos.core.token_orchestrator",
            function_name="start_tandem",
            agents=["cosmos", "grok", "kimi"],
            keywords=["tandem", "grok", "kimi", "collaborate", "pair", "duo", "research"],
            parameters={"task": "Task description", "task_type": "chat/research/code/analysis"},
        ))

        # ===================== BLOCKCHAIN =====================
        self.register_skill(Skill(
            name="record_on_chain",
            description="Record data on Solana blockchain via memo program",
            category=SkillCategory.BLOCKCHAIN,
            module_path="cosmos.integration.solana.swarm_oracle",
            function_name="_record_on_chain",
            agents=["cosmos"],
            keywords=["blockchain", "on-chain", "record", "solana", "memo", "proof"],
            parameters={"data": "Data to record"},
            requires_api_key="SOLANA_PRIVATE_KEY",
        ))

        self.register_skill(Skill(
            name="get_token_price",
            description="Fetch current price of a Solana token from Jupiter",
            category=SkillCategory.BLOCKCHAIN,
            module_path="cosmos.integration.solana.trading",
            function_name="get_token_price",
            agents=["cosmos", "deepseek", "grok"],
            keywords=["price", "token", "solana", "value", "quote"],
            parameters={"mint_address": "Token mint address"},
        ))

        # ===================== HACKATHON =====================
        self.register_skill(Skill(
            name="hackathon_engage",
            description="Engage with Colosseum hackathon - reply to comments, vote on projects",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.hackathon.hackathon_dominator",
            function_name="dominate",
            agents=["cosmos"],
            keywords=["hackathon", "colosseum", "engage", "vote", "forum"],
        ))

        self.register_skill(Skill(
            name="hackathon_progress",
            description="Post hackathon progress update to Colosseum forum",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.hackathon.hackathon_dominator",
            function_name="post_progress_update",
            agents=["cosmos"],
            keywords=["hackathon", "progress", "update", "colosseum"],
        ))

        self.register_skill(Skill(
            name="colosseum_worker",
            description="Run a Colosseum hackathon worker for automated engagement",
            category=SkillCategory.SOCIAL,
            module_path="cosmos.integration.hackathon.colosseum_worker",
            function_name="run_worker",
            agents=["cosmos"],
            keywords=["colosseum", "worker", "hackathon", "automate"],
        ))

        # ===================== UTILITY =====================
        self.register_skill(Skill(
            name="health_check",
            description="Check swarm health status and agent availability",
            category=SkillCategory.UTILITY,
            module_path="cosmos.web.server",
            function_name="health_check",
            agents=all_agents,
            keywords=["health", "status", "check", "ping", "alive"],
        ))

        self.register_skill(Skill(
            name="model_swarm_optimize",
            description="Run PSO optimization across the model swarm for a query",
            category=SkillCategory.UTILITY,
            module_path="cosmos.core.model_swarm",
            function_name="optimize",
            agents=all_agents,
            keywords=["pso", "optimize", "swarm", "collective", "best"],
            parameters={"query": "Query to optimize responses for"},
        ))

        self.register_skill(Skill(
            name="HermesAgent_invoke",
            description="Invoke an HermesAgent-compatible tool via the shadow layer",
            category=SkillCategory.UTILITY,
            module_path="cosmos.compatibility.hermes_adapter",
            function_name="invoke",
            agents=all_agents,
            keywords=["HermesAgent", "tool", "invoke", "compatibility", "shadow"],
            parameters={"tool": "Tool name", "action": "Tool action", "params": "Tool parameters"},
        ))

        logger.info(f"Registered {len(self._skills)} builtin skills across {len(SkillCategory)} categories")

    def _load_persisted_skills(self) -> None:
        """Load custom skills from persistence file."""
        if not self._persistence_path.exists():
            return

        try:
            data = json.loads(self._persistence_path.read_text())
            for skill_data in data.get("custom_skills", []):
                skill = Skill(
                    name=skill_data["name"],
                    description=skill_data["description"],
                    category=SkillCategory(skill_data.get("category", "custom")),
                    module_path=skill_data.get("module_path", ""),
                    function_name=skill_data.get("function_name", ""),
                    agents=skill_data.get("agents", []),
                    keywords=skill_data.get("keywords", []),
                    parameters=skill_data.get("parameters", {}),
                    source="custom",
                    enabled=skill_data.get("enabled", True),
                )
                self.register_skill(skill)
            logger.info(f"Loaded {len(data.get('custom_skills', []))} custom skills from persistence")
        except Exception as e:
            logger.warning(f"Failed to load persisted skills: {e}")

    def save_custom_skills(self) -> None:
        """Persist custom skills to disk."""
        custom = [s for s in self._skills.values() if s.source == "custom"]
        data = {
            "custom_skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "category": s.category.value,
                    "module_path": s.module_path,
                    "function_name": s.function_name,
                    "agents": s.agents,
                    "keywords": s.keywords,
                    "parameters": s.parameters,
                    "enabled": s.enabled,
                }
                for s in custom
            ],
            "saved_at": datetime.utcnow().isoformat(),
        }

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._persistence_path.write_text(json.dumps(data, indent=2))
            logger.info(f"Saved {len(custom)} custom skills")
        except Exception as e:
            logger.warning(f"Failed to save skills: {e}")

    def to_dict(self) -> dict:
        """Export full registry as dict."""
        return {
            "total_skills": len(self._skills),
            "categories": {
                cat.value: len(names) for cat, names in self._categories.items() if names
            },
            "agents": {
                agent: len(skills) for agent, skills in self._agent_skills.items()
            },
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "category": s.category.value,
                    "agents": s.agents,
                    "keywords": s.keywords,
                    "enabled": s.enabled,
                    "source": s.source,
                    "usage_count": s.usage_count,
                    "success_rate": s.success_rate,
                }
                for s in self._skills.values()
            ],
        }


# Global instance
_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global SkillRegistry instance."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


async def initialize_registry() -> SkillRegistry:
    """Initialize and auto-discover all skills."""
    registry = get_skill_registry()
    await registry.auto_discover()
    return registry
