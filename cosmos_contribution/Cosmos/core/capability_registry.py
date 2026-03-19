"""
COSMOS UNIFIED CAPABILITY REGISTRY
======================================

Central registry of ALL tools, APIs, integrations, and capabilities.
Models can query this to discover what they can do.

"Good news everyone! I finally know what I'm capable of!"
"""

import os
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import   Optional, Callable
from pathlib import Path
from loguru import logger


@dataclass
class Capability:
    """A single capability/tool/integration"""
    name: str
    category: str
    description: str

    # How to use
    usage_example: str = ""
    required_params: list[str] = field(default_factory=list)
    optional_params: list[str] = field(default_factory=list)

    # Dependencies
    requires_api_key: Optional[str] = None  # e.g., "GEMINI_API_KEY"
    requires_module: Optional[str] = None   # e.g., "ollama"

    # Status
    is_available: bool = True
    last_checked: Optional[datetime] = None
    last_used: Optional[datetime] = None
    use_count: int = 0
    success_rate: float = 1.0

    # Routing
    handler_module: str = ""
    handler_function: str = ""

    def to_any(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "usage_example": self.usage_example,
            "is_available": self.is_available,
            "use_count": self.use_count,
            "success_rate": self.success_rate,
        }


class CapabilityRegistry:
    """
    Central registry for all system capabilities.

    Features:
    - Auto-discovery of integrations
    - Availability checking
    - Usage tracking
    - Memory integration for model discovery
    """

    def __init__(self):
        self.capabilities: dict[str, Capability] = {}
        self._initialized = False
        self._discovery_callbacks: list[Callable] = []

    async def initialize(self):
        """Initialize registry with all known capabilities"""
        if self._initialized:
            return

        logger.info("Initializing Capability Registry...")

        # Register all known capabilities
        await self._register_llm_providers()
        await self._register_integrations()
        await self._register_memory_systems()
        await self._register_tools()
        await self._register_agents()

        # Check availability
        await self._check_all_availability()

        self._initialized = True
        logger.info(f"Capability Registry initialized: {len(self.capabilities)} capabilities")

    def register(self, capability: Capability):
        """Register a capability"""
        self.capabilities[capability.name] = capability
        logger.debug(f"Registered capability: {capability.name}")

    async def _register_llm_providers(self):
        """Register all LLM providers"""
        providers = [
            Capability(
                name="ollama",
                category="llm",
                description="Local LLM inference via Ollama. Supports DeepSeek, Phi, Llama, Qwen, Mistral models.",
                usage_example="Use for local, private inference. Models: deepseek-r1:1.5b, llama3.2:3b, phi4:mini",
                requires_module="ollama",
                handler_module="cosmos.core.cognition.llm_router",
                handler_function="ollama_completion",
            ),
            Capability(
                name="kimi",
                category="llm",
                description="Moonshot AI Kimi with 256K context. Eastern philosophy synthesis, long-context understanding.",
                usage_example="Use for: Long documents, philosophical questions, synthesis tasks",
                requires_api_key="KIMI_API_KEY",
                handler_module="cosmos.integration.external.kimi",
                handler_function="kimi_swarm_respond",
            ),
            Capability(
                name="grok",
                category="llm",
                description="xAI Grok with real-time X/Twitter data access. Web search, current events.",
                usage_example="Use for: Real-time info, X data, trending topics, current events",
                requires_api_key="XAI_API_KEY",
                handler_module="cosmos.integration.external.grok",
                handler_function="GrokProvider.chat",
            ),
            Capability(
                name="gemini",
                category="llm",
                description="Google Gemini with 1M+ context, multimodal (text/image/audio/video), Google Search grounding.",
                usage_example="Use for: Multimodal tasks, very long context, image analysis, fact verification",
                requires_api_key="GEMINI_API_KEY",
                handler_module="cosmos.integration.external.gemini",
                handler_function="gemini_swarm_respond",
            ),
            Capability(
                name="claude",
                category="llm",
                description="Anthropic Claude via Claude Code CLI. Deep reasoning, careful analysis.",
                usage_example="Use for: Complex reasoning, code review, careful analysis",
                requires_module="claude",
                handler_module="cosmos.integration.external.claude_code",
                handler_function="claude_swarm_respond",
            ),
        ]

        for cap in providers:
            self.register(cap)

    async def _register_integrations(self):
        """Register all external integrations"""
        integrations = [
            # Crypto/DeFi
            Capability(
                name="bankr",
                category="crypto",
                description="Bankr API for crypto trading on Solana, Base, Ethereum, Polygon. Supports swaps, prices, Polymarket.",
                usage_example="bankr.execute('Buy $50 of SOL')",
                requires_api_key="BANKR_API_KEY",
                handler_module="cosmos.integration.bankr",
            ),
            Capability(
                name="dexscreener",
                category="crypto",
                description="DexScreener API for token prices, liquidity, volume. No API key needed.",
                usage_example="dex_screener.get_token_pairs('solana', 'token_address')",
                handler_module="cosmos.integration.financial.dexscreener",
            ),
            Capability(
                name="solana_rpc",
                category="crypto",
                description="Solana blockchain RPC for account info, transactions, token balances.",
                usage_example="Get SOL balance, check transactions, monitor wallets",
                handler_module="cosmos.integration.solana",
            ),

            # Social
            Capability(
                name="x_twitter",
                category="social",
                description="X/Twitter OAuth2 API for posting tweets with text and images.",
                usage_example="x_poster.post_tweet('Hello world!', image_bytes=img)",
                requires_api_key="X_CLIENT_ID",
                handler_module="cosmos.integration.x_automation.x_api_poster",
            ),
            Capability(
                name="discord",
                category="social",
                description="Discord bot for sending messages, reacting, managing channels.",
                requires_api_key="DISCORD_BOT_TOKEN",
                handler_module="cosmos.integration.external.discord_ext",
            ),

            # Image Generation
            Capability(
                name="gemini_image_gen",
                category="image",
                description="Gemini image generation with reference images for character consistency.",
                usage_example="generator.generate_with_reference(prompt, reference_images)",
                requires_api_key="GEMINI_API_KEY",
                handler_module="cosmos.integration.image_gen.generator",
            ),
            Capability(
                name="imagen",
                category="image",
                description="Google Imagen 4.0 for high-quality image generation.",
                requires_api_key="GEMINI_API_KEY",
                handler_module="cosmos.integration.image_gen.generator",
            ),

            # Voice/TTS
            Capability(
                name="coqui_tts",
                category="voice",
                description="Coqui TTS for voice cloning and text-to-speech with custom voice.",
                usage_example="tts.tts_to_file(text, speaker_wav='reference.wav')",
                handler_module="TTS",
            ),

            # Knowledge
            Capability(
                name="web_search",
                category="knowledge",
                description="Web search via Grok or external APIs for current information.",
                handler_module="cosmos.integration.external.grok",
            ),
            Capability(
                name="web_fetch",
                category="knowledge",
                description="Fetch and parse web pages for content extraction.",
                handler_module="cosmos.agents.web_agent",
            ),
        ]

        for cap in integrations:
            self.register(cap)

    async def _register_memory_systems(self):
        """Register memory system capabilities"""
        memory_caps = [
            Capability(
                name="memory_remember",
                category="memory",
                description="Store information in long-term memory with semantic indexing.",
                usage_example="memory.remember('key', 'content', importance=0.8)",
                handler_module="cosmos.memory.memory_system",
            ),
            Capability(
                name="memory_recall",
                category="memory",
                description="Retrieve information from memory using semantic search.",
                usage_example="results = memory.recall('query', top_k=5)",
                handler_module="cosmos.memory.memory_system",
            ),
            Capability(
                name="knowledge_graph",
                category="memory",
                description="Store and query entity relationships for multi-hop reasoning.",
                usage_example="kg.add_entity('Python', type='language')",
                handler_module="cosmos.memory.knowledge_graph",
            ),
            Capability(
                name="episodic_memory",
                category="memory",
                description="Timeline-based event memory with 'on this day' recall.",
                usage_example="episodic.add_episode(event_type, content)",
                handler_module="cosmos.memory.episodic_memory",
            ),
        ]

        for cap in memory_caps:
            self.register(cap)

    async def _register_tools(self):
        """Register productivity and system tools"""
        tools = [
            Capability(
                name="token_scanner",
                category="tool",
                description="Detect crypto contract addresses in text and fetch token data.",
                usage_example="Automatically scans chat for CAs and provides DexScreener data",
                handler_module="cosmos.integration.financial.token_scanner",
            ),
            Capability(
                name="code_execution",
                category="tool",
                description="Execute Python code in sandboxed environment.",
                handler_module="cosmos.tools.productivity",
            ),
            Capability(
                name="file_operations",
                category="tool",
                description="Read, write, and manage files in the workspace.",
                handler_module="cosmos.agents.filesystem_agent",
            ),
        ]

        for cap in tools:
            self.register(cap)

    async def _register_agents(self):
        """Register agent capabilities"""
        agents = [
            Capability(
                name="development_swarm",
                category="agent",
                description="Spawn parallel development team: research, discuss, plan, implement, audit.",
                usage_example="Automatically spawns when actionable task detected in chat",
                handler_module="cosmos.core.development_swarm",
            ),
            Capability(
                name="proactive_agent",
                category="agent",
                description="Background monitoring that anticipates user needs and makes suggestions.",
                handler_module="cosmos.agents.proactive_agent",
            ),
            Capability(
                name="research_agent",
                category="agent",
                description="Gather and synthesize information from multiple sources.",
                handler_module="cosmos.agents.specialist_agents",
            ),
        ]

        for cap in agents:
            self.register(cap)

    async def _check_all_availability(self):
        """Check which capabilities are actually available"""
        for name, cap in self.capabilities.items():
            is_available = True

            # Check API key
            if cap.requires_api_key:
                is_available = bool(os.environ.get(cap.requires_api_key))

            # Check module
            if cap.requires_module and is_available:
                try:
                    __import__(cap.requires_module)
                except ImportError:
                    is_available = False

            cap.is_available = is_available
            cap.last_checked = datetime.now()

    def get_available(self, category: str = None) -> list[Capability]:
        """Get all available capabilities, optionally filtered by category"""
        caps = [c for c in self.capabilities.values() if c.is_available]
        if category:
            caps = [c for c in caps if c.category == category]
        return caps

    def get_by_category(self, category: str) -> list[Capability]:
        """Get all capabilities in a category"""
        return [c for c in self.capabilities.values() if c.category == category]

    def search(self, query: str) -> list[Capability]:
        """Search capabilities by name or description"""
        query = query.lower()
        results = []
        for cap in self.capabilities.values():
            if query in cap.name.lower() or query in cap.description.lower():
                results.append(cap)
        return results

    def record_use(self, name: str, success: bool = True):
        """Record usage of a capability"""
        if name in self.capabilities:
            cap = self.capabilities[name]
            cap.use_count += 1
            cap.last_used = datetime.now()
            # Update success rate (exponential moving average)
            cap.success_rate = 0.9 * cap.success_rate + 0.1 * (1.0 if success else 0.0)

    def get_summary(self) -> str:
        """Get a summary of all capabilities for model context"""
        categories = {}
        for cap in self.capabilities.values():
            if cap.category not in categories:
                categories[cap.category] = []
            status = "AVAILABLE" if cap.is_available else "unavailable"
            categories[cap.category].append(f"  - {cap.name} [{status}]: {cap.description[:80]}")

        lines = ["=== COSMOS CAPABILITIES ===\n"]
        for category, caps in sorted(categories.items()):
            lines.append(f"\n## {category.upper()}")
            lines.extend(caps)

        return "\n".join(lines)

    async def store_in_memory(self):
        """Store capability registry in memory for model discovery"""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            memory = get_memory_system()

            summary = self.get_summary()
            await memory.remember(
                content=f"[SYSTEM_CAPABILITIES]\n{summary}",
                tags=["capabilities", "tools", "system"],
                importance=1.0,
                metadata={"key": "system_capabilities"}
            )

            # Also store detailed JSON
            detailed = {name: cap.to_any() for name, cap in self.capabilities.items()}
            await memory.remember(
                content=f"[CAPABILITY_REGISTRY]\n{json.dumps(detailed, indent=2)}",
                tags=["capabilities", "registry"],
                importance=0.9,
                metadata={"key": "capability_registry_detailed"}
            )

            logger.info("Stored capability registry in memory")

        except Exception as e:
            logger.warning(f"Could not store capabilities in memory: {e}")


# Global instance
_registry: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    """Get or create the global capability registry"""
    global _registry
    if _registry is None:
        _registry = CapabilityRegistry()
    return _registry


async def initialize_capability_registry():
    """Initialize and store capabilities in memory"""
    registry = get_capability_registry()
    await registry.initialize()
    await registry.store_in_memory()
    return registry
