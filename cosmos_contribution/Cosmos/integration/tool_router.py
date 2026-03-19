"""
cosmos Tool Router - Dynamic Tool Management and Routing

Provides intelligent tool routing with:
- Capability-based tool matching
- Dynamic tool discovery
- Tool composition for complex tasks
- Usage tracking and optimization

Novel Features:
- Semantic tool matching using embeddings
- Automatic tool chaining for multi-step tasks
- Tool performance tracking and selection
- Composio integration for 500+ external tools
"""

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Optional, Any, Callable, Awaitable, List, Dict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from loguru import logger


class ToolCategory(Enum):
    """Tool categorization."""
    FILE_SYSTEM = "file_system"
    CODE = "code"
    WEB = "web"
    DATABASE = "database"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    UTILITY = "utility"
    CUSTOM = "custom"
    ACTION = "action"


@dataclass
class ToolDefinition:
    """Definition of a tool available to the system."""
    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, Any]
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    examples: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    priority: int = 5
    enabled: bool = True

    # Performance tracking
    usage_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    last_used: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "examples": self.examples,
            "capabilities": self.capabilities,
            "priority": self.priority,
            "enabled": self.enabled,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(1, self.usage_count),
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class ToolRouter:
    """
    Intelligent tool routing system.

    Provides:
    - Tool registration and discovery
    - Capability-based matching
    - Automatic tool chaining
    - Performance optimization
    """

    def __init__(
        self,
        embedder=None,
        enable_composio: bool = False,
        composio_api_key: Optional[str] = None,
    ):
        self.tools: dict[str, ToolDefinition] = {}
        self.embedder = embedder
        self.tool_embeddings: dict[str, list[float]] = {}

        # Composio integration
        self.composio_enabled = enable_composio
        self.composio_client = None
        if enable_composio and composio_api_key:
            self._init_composio(composio_api_key)

        # Tool chains for complex tasks
        self.tool_chains: dict[str, list[str]] = {}

        # Register built-in tools
        self._register_builtin_tools()

    def _init_composio(self, api_key: str):
        """Initialize Composio integration for 500+ tools."""
        try:
            # Placeholder for Composio integration
            # In production: from composio import ComposioClient
            logger.info("Composio integration enabled")
        except ImportError:
            logger.warning("Composio library not installed")
            self.composio_enabled = False

    def _register_builtin_tools(self):
        """Register built-in tools."""
        # File system tools
        self.register_tool(ToolDefinition(
            name="read_file",
            description="Read contents of a file from the filesystem",
            category=ToolCategory.FILE_SYSTEM,
            parameters={
                "path": {"type": "string", "required": True, "description": "File path to read"},
            },
            examples=["read_file path=/home/user/document.txt"],
            capabilities=["file_read", "text_extraction"],
            handler=self._handle_read_file,
        ))

        self.register_tool(ToolDefinition(
            name="write_file",
            description="Write contents to a file on the filesystem",
            category=ToolCategory.FILE_SYSTEM,
            parameters={
                "path": {"type": "string", "required": True, "description": "File path to write"},
                "content": {"type": "string", "required": True, "description": "Content to write"},
            },
            examples=["write_file path=/home/user/output.txt content='Hello World'"],
            capabilities=["file_write", "file_create"],
            handler=self._handle_write_file,
        ))

        self.register_tool(ToolDefinition(
            name="list_directory",
            description="List contents of a directory",
            category=ToolCategory.FILE_SYSTEM,
            parameters={
                "path": {"type": "string", "required": True, "description": "Directory path"},
            },
            examples=["list_directory path=/home/user"],
            capabilities=["directory_listing", "file_discovery"],
            handler=self._handle_list_directory,
        ))

        # Code tools
        self.register_tool(ToolDefinition(
            name="execute_python",
            description="Execute Python code in a sandboxed environment",
            category=ToolCategory.CODE,
            parameters={
                "code": {"type": "string", "required": True, "description": "Python code to execute"},
            },
            examples=["execute_python code='print(2 + 2)'"],
            capabilities=["code_execution", "python", "computation"],
            handler=self._handle_execute_python,
        ))

        self.register_tool(ToolDefinition(
            name="analyze_code",
            description="Analyze code for issues and suggestions",
            category=ToolCategory.CODE,
            parameters={
                "code": {"type": "string", "required": True, "description": "Code to analyze"},
                "language": {"type": "string", "required": False, "description": "Programming language"},
            },
            examples=["analyze_code code='def foo(): pass' language=python"],
            capabilities=["code_analysis", "linting", "suggestions"],
            handler=self._handle_analyze_code,
        ))

        # Web tools
        self.register_tool(ToolDefinition(
            name="web_search",
            description="Search the web for information",
            category=ToolCategory.WEB,
            parameters={
                "query": {"type": "string", "required": True, "description": "Search query"},
                "num_results": {"type": "integer", "required": False, "description": "Number of results"},
            },
            examples=["web_search query='Python async programming'"],
            capabilities=["web_search", "information_retrieval"],
            handler=self._handle_web_search,
        ))

        self.register_tool(ToolDefinition(
            name="fetch_url",
            description="Fetch content from a URL",
            category=ToolCategory.WEB,
            parameters={
                "url": {"type": "string", "required": True, "description": "URL to fetch"},
            },
            examples=["fetch_url url='https://example.com'"],
            capabilities=["web_fetch", "html_parsing"],
            handler=self._handle_fetch_url,
        ))

        # Analysis tools
        self.register_tool(ToolDefinition(
            name="summarize_text",
            description="Summarize a long text into key points",
            category=ToolCategory.ANALYSIS,
            parameters={
                "text": {"type": "string", "required": True, "description": "Text to summarize"},
                "max_length": {"type": "integer", "required": False, "description": "Max summary length"},
            },
            examples=["summarize_text text='Long document...' max_length=100"],
            capabilities=["summarization", "text_analysis"],
            handler=self._handle_summarize,
        ))

        self.register_tool(ToolDefinition(
            name="extract_entities",
            description="Extract named entities from text",
            category=ToolCategory.ANALYSIS,
            parameters={
                "text": {"type": "string", "required": True, "description": "Text to analyze"},
            },
            examples=["extract_entities text='John works at Google in California'"],
            capabilities=["ner", "entity_extraction", "text_analysis"],
            handler=self._handle_extract_entities,
        ))

        self.register_tool(ToolDefinition(
            name="analyze_multimodal_content",
            description="Analyze image, audio, or text using the 12D Sensory Cortex",
            category=ToolCategory.ANALYSIS,
            parameters={
                "image_path": {"type": "string", "required": False, "description": "Path to image file"},
                "audio_path": {"type": "string", "required": False, "description": "Path to audio file"},
                "text_context": {"type": "string", "required": False, "description": "Text context to fuse"},
            },
            capabilities=["multimodal", "vision", "audio", "12d_analysis"],
            handler=self._handle_multimodal_analysis,
        ))

        # Generation tools
        self.register_tool(ToolDefinition(
            name="generate_image",
            description="Generate an image from a text description",
            category=ToolCategory.GENERATION,
            parameters={
                "prompt": {"type": "string", "required": True, "description": "Image description"},
                "size": {"type": "string", "required": False, "description": "Image size"},
            },
            examples=["generate_image prompt='A sunset over mountains'"],
            capabilities=["image_generation", "creative"],
            handler=self._handle_generate_image,
        ))

        # Utility tools
        self.register_tool(ToolDefinition(
            name="calculate",
            description="Perform mathematical calculations",
            category=ToolCategory.UTILITY,
            parameters={
                "expression": {"type": "string", "required": True, "description": "Math expression"},
            },
            examples=["calculate expression='sqrt(16) + 5 * 2'"],
            capabilities=["math", "calculation"],
            handler=self._handle_calculate,
        ))

        self.register_tool(ToolDefinition(
            name="datetime_info",
            description="Get current date, time, or timezone information",
            category=ToolCategory.UTILITY,
            parameters={
                "timezone": {"type": "string", "required": False, "description": "Timezone name"},
            },
            examples=["datetime_info timezone='America/New_York'"],
            capabilities=["datetime", "timezone"],
            handler=self._handle_datetime,
        ))

        # --- New Skills (Grok, Remotion, Parallel) ---
        self.register_tool(ToolDefinition(
            name="grok_search",
            description="Perform a real-time search on X (Twitter) using Grok AI",
            category=ToolCategory.WEB,
            parameters={
                "query": {"type": "string", "required": True, "description": "Search query"},
            },
            capabilities=["x_search", "grok", "realtime"],
            handler=self._handle_grok_search,
        ))

        self.register_tool(ToolDefinition(
            name="render_video",
            description="Render a programmatic video using Remotion/React",
            category=ToolCategory.GENERATION,
            parameters={
                "narrative": {"type": "string", "required": True, "description": "Script/Narrative for the video"},
                "composition_id": {"type": "string", "required": False, "description": "ID of the remotion composition"},
            },
            capabilities=["video_render", "remotion", "react"],
            handler=self._handle_remotion_render,
        ))

        self.register_tool(ToolDefinition(
            name="parallel_ai_dispatch",
            description="Dispatch a prompt to multiple models in parallel and fuse results",
            category=ToolCategory.ANALYSIS,
            parameters={
                "prompt": {"type": "string", "required": True, "description": "The prompt to dispatch"},
            },
            capabilities=["parallel_processing", "consensus", "high_reliability"],
            handler=self._handle_parallel_ai,
        ))

        self.register_tool(ToolDefinition(
            name="discord_broadcast",
            description="Send a message to a specific Discord channel",
            category=ToolCategory.COMMUNICATION,
            parameters={
                "channel_id": {"type": "integer", "required": True, "description": "Target Discord channel ID"},
                "content": {"type": "string", "required": True, "description": "Message content"},
            },
            capabilities=["chatops", "discord", "broadcast"],
            handler=self._handle_discord_send,
        ))

        self.register_tool(ToolDefinition(
            name="generate_mermaid_chart",
            description="Generate a Mermaid diagram for technical visualization",
            category=ToolCategory.GENERATION,
            parameters={
                "diagram_type": {"type": "string", "required": True, "description": "flowchart / sequence / gantt"},
                "data": {"type": "object", "required": True, "description": "Nodes and edges definition"},
            },
            capabilities=["visual_logic", "mermaid", "documentation"],
            handler=self._handle_mermaid_gen,
        ))

        self.register_tool(ToolDefinition(
            name="system_diagnostic",
            description="Get deep system health, load, and active processes",
            category=ToolCategory.UTILITY,
            parameters={},
            capabilities=["os_level", "diagnostics", "agentic_os"],
            handler=self._handle_system_diag,
        ))

        # --- Community Fold Skills (YouTube, DB, Thinking) ---
        self.register_tool(ToolDefinition(
            name="youtube_analyze",
            description="Extract transcript and main insights from a YouTube video",
            category=ToolCategory.ANALYSIS,
            parameters={
                "video_url": {"type": "string", "required": True, "description": "Full YouTube URL"},
            },
            capabilities=["video_understanding", "transcription", "youtube"],
            handler=self._handle_youtube_analyze,
        ))

        self.register_tool(ToolDefinition(
            name="sequential_thought",
            description="Perform a step in a systematic reasoning chain",
            category=ToolCategory.ANALYSIS,
            parameters={
                "thought": {"type": "string", "required": True, "description": "The current logical step"},
                "verification": {"type": "string", "required": False, "description": "How this step was verified"},
                "is_new_chain": {"type": "boolean", "required": False, "description": "Start a fresh reasoning path"},
            },
            capabilities=["reasoning", "chain_of_thought", "systematic"],
            handler=self._handle_sequential_thought,
        ))

        self.register_tool(ToolDefinition(
            name="database_query",
            description="Execute a read-only SQL query on a connected database",
            category=ToolCategory.DATABASE,
            parameters={
                "query": {"type": "string", "required": True, "description": "SQL SELECT query"},
            },
            capabilities=["sql", "data_extraction", "analytics"],
            handler=self._handle_db_query,
        ))

        # --- Financial Intelligence Skills ---
        self.register_tool(ToolDefinition(
            name="dex_screener_search",
            description="Search for on-chain tokens and trading pairs on DexScreener",
            category=ToolCategory.ANALYSIS,
            parameters={
                "query": {"type": "string", "required": True, "description": "Token name, symbol, or address"},
            },
            capabilities=["crypto", "dex", "token_data"],
            handler=self._handle_dex_search,
        ))

        self.register_tool(ToolDefinition(
            name="polymarket_scan",
            description="Scan prediction markets and betting odds on Polymarket",
            category=ToolCategory.ANALYSIS,
            parameters={
                "query": {"type": "string", "required": True, "description": "Search term for events/markets"},
            },
            capabilities=["prediction_markets", "betting", "forecasting"],
            handler=self._handle_poly_scan,
        ))

        self.register_tool(ToolDefinition(
            name="market_sentiment_check",
            description="Fetch global market conditions and Fear & Greed Index",
            category=ToolCategory.UTILITY,
            parameters={},
            capabilities=["market_sentiment", "macro", "crypto"],
            handler=self._handle_sentiment_check,
        ))

        # --- High-Degeneracy Memecoin Skills ---
        self.register_tool(ToolDefinition(
            name="pump_fun_track",
            description="Track bonding curve and token data on Pump.fun",
            category=ToolCategory.ANALYSIS,
            parameters={
                "mint_address": {"type": "string", "required": True, "description": "Solana mint address of the token"},
            },
            capabilities=["pump.fun", "solana", "bonding_curve", "memecoins"],
            handler=self._handle_pump_track,
        ))

        self.register_tool(ToolDefinition(
            name="bags_fm_track",
            description="Track token data and trending analytics on Bags.fm",
            category=ToolCategory.ANALYSIS,
            parameters={
                "token_address": {"type": "string", "required": True, "description": "Token address or name"},
            },
            capabilities=["bags.fm", "memecoins", "social_finance"],
            handler=self._handle_bags_track,
        ))

        self.register_tool(ToolDefinition(
            name="bags_fm_trending",
            description="Get trending tokens on Bags.fm",
            category=ToolCategory.ANALYSIS,
            parameters={},
            capabilities=["bags.fm", "memecoins", "trending"],
            handler=self._handle_bags_trending,
        ))

        self.register_tool(ToolDefinition(
            name="bags_fm_trade_quote",
            description="Get a trade quote for buying/selling tokens on Bags.fm",
            category=ToolCategory.ANALYSIS,
            parameters={
                "token_mint": {"type": "string", "required": True, "description": "Token mint address"},
                "amount": {"type": "number", "required": True, "description": "Amount to trade"},
                "side": {"type": "string", "required": True, "description": "'buy' or 'sell'"},
                "slippage_bps": {"type": "integer", "required": False, "description": "Slippage in basis points (default 100 = 1%)"},
            },
            capabilities=["bags.fm", "trading", "quote"],
            handler=self._handle_bags_quote,
        ))

        self.register_tool(ToolDefinition(
            name="bags_fm_swap",
            description="Create a swap transaction on Bags.fm",
            category=ToolCategory.ACTION,
            parameters={
                "token_mint": {"type": "string", "required": True, "description": "Token mint address"},
                "amount": {"type": "number", "required": True, "description": "Amount to swap"},
                "side": {"type": "string", "required": True, "description": "'buy' or 'sell'"},
                "slippage_bps": {"type": "integer", "required": False, "description": "Slippage in basis points (default 100)"},
            },
            capabilities=["bags.fm", "trading", "swap", "solana"],
            handler=self._handle_bags_swap,
        ))

        self.register_tool(ToolDefinition(
            name="bags_fm_token_creators",
            description="Get creators associated with a Bags.fm token launch",
            category=ToolCategory.ANALYSIS,
            parameters={
                "token_mint": {"type": "string", "required": True, "description": "Token mint address"},
            },
            capabilities=["bags.fm", "token_launch", "creators"],
            handler=self._handle_bags_creators,
        ))

        self.register_tool(ToolDefinition(
            name="bags_fm_token_fees",
            description="Get lifetime fee information for a Bags.fm token",
            category=ToolCategory.ANALYSIS,
            parameters={
                "token_mint": {"type": "string", "required": True, "description": "Token mint address"},
            },
            capabilities=["bags.fm", "fees", "analytics"],
            handler=self._handle_bags_fees,
        ))

        self.register_tool(ToolDefinition(
            name="bags_fm_claimable",
            description="Get claimable fee positions for a wallet on Bags.fm",
            category=ToolCategory.ANALYSIS,
            parameters={
                "wallet_address": {"type": "string", "required": True, "description": "Solana wallet address"},
            },
            capabilities=["bags.fm", "fees", "claimable", "wallet"],
            handler=self._handle_bags_claimable,
        ))

        self.register_tool(ToolDefinition(
            name="bags_fm_partner_stats",
            description="Get partner statistics on Bags.fm",
            category=ToolCategory.ANALYSIS,
            parameters={
                "wallet_address": {"type": "string", "required": True, "description": "Partner wallet address"},
            },
            capabilities=["bags.fm", "partner", "stats"],
            handler=self._handle_bags_partner_stats,
        ))

        # --- Solana Trading Skills ---
        self.register_tool(ToolDefinition(
            name="solana_get_balance",
            description="Get SOL balance for a wallet",
            category=ToolCategory.UTILITY,
            parameters={
                "pubkey": {"type": "string", "required": False, "description": "Public key to check (defaults to user wallet)"},
            },
            capabilities=["solana", "wallet", "balance"],
            handler=self._handle_solana_balance,
        ))

        self.register_tool(ToolDefinition(
            name="jupiter_swap",
            description="Swap tokens on Solana via Jupiter Aggregator",
            category=ToolCategory.ACTION,
            parameters={
                "input_mint": {"type": "string", "required": True, "description": "Source token mint"},
                "output_mint": {"type": "string", "required": True, "description": "Target token mint"},
                "amount": {"type": "integer", "required": True, "description": "Amount in smallest unit (indices)"},
            },
            capabilities=["solana", "dex", "trading", "jupiter"],
            handler=self._handle_jup_swap,
        ))

        self.register_tool(ToolDefinition(
            name="pump_fun_trade",
            description="Buy or Sell tokens on Pump.fun bonding curves",
            category=ToolCategory.ACTION,
            parameters={
                "action": {"type": "string", "required": True, "description": "'buy' or 'sell'"},
                "mint": {"type": "string", "required": True, "description": "Token mint address"},
                "amount": {"type": "number", "required": True, "description": "Amount (SOL for buy, tokens for sell)"},
            },
            capabilities=["solana", "memecoins", "trading", "pump.fun"],
            handler=self._handle_pump_trade,
        ))

        self.register_tool(ToolDefinition(
            name="meteora_pool_info",
            description="Get information about a Meteora Dynamic/DLMM pool",
            category=ToolCategory.ANALYSIS,
            parameters={
                "pair_address": {"type": "string", "required": True, "description": "Pool pair address"},
            },
            capabilities=["solana", "dex", "liquidity", "meteora"],
            handler=self._handle_meteora_info,
        ))

        self.register_tool(ToolDefinition(
            name="evaluate_trade_signal",
            description="Perform a deep cognitive analysis of a token signal (Reasoning + Learning)",
            category=ToolCategory.ANALYSIS,
            parameters={
                "token_symbol": {"type": "string", "required": True, "description": "Token symbol or mint"},
                "liquidity": {"type": "number", "required": False, "description": "Current liquidity in USD"},
                "volume_24h": {"type": "number", "required": False, "description": "24h trading volume"},
            },
            capabilities=["reasoning", "trading_alpha", "learning"],
            handler=self._handle_trade_evaluation,
        ))

        # --- DeGen Mob Elite Skills ---
        self.register_tool(ToolDefinition(
            name="solana_rug_check",
            description="Perform a security scan on a token mint to detect rugs/honeypots",
            category=ToolCategory.ANALYSIS,
            parameters={
                "mint_address": {"type": "string", "required": True, "description": "Solana token address"},
            },
            capabilities=["security", "solana", "rug_check", "safety"],
            handler=self._handle_rug_check,
        ))

        self.register_tool(ToolDefinition(
            name="solana_whale_watch",
            description="Track real-time activity and rotations of specific whale wallets",
            category=ToolCategory.ANALYSIS,
            parameters={
                "wallet_address": {"type": "string", "required": True, "description": "Solana wallet to monitor"},
                "action": {"type": "string", "required": False, "description": "add / list / history"},
            },
            capabilities=["whale_tracking", "copy_trading", "alpha"],
            handler=self._handle_whale_watch,
        ))

        self.register_tool(ToolDefinition(
            name="solana_snipe_toggle",
            description="Activate or deactivate the automated launch sniper",
            category=ToolCategory.ACTION,
            parameters={
                "enabled": {"type": "boolean", "required": True, "description": "True to start, False to stop"},
                "query": {"type": "string", "required": False, "description": "Filters for tokens to snipe"},
            },
            capabilities=["sniping", "automation", "hft"],
            handler=self._handle_snipe_toggle,
        ))

        self.register_tool(ToolDefinition(
            name="ct_sentiment_swarm",
            description="Run high-scale sentiment analysis on CT (Twitter) for narrative rotations",
            category=ToolCategory.ANALYSIS,
            parameters={
                "keywords": {"type": "array", "items": {"type": "string"}, "required": True, "description": "Narratives to track"},
            },
            capabilities=["sentiment", "narratives", "twitter", "alpha"],
            handler=self._handle_ct_swarm,
        ))

        self.register_tool(ToolDefinition(
            name="show_trading_dashboard",
            description="Visualize the DeGen Mob Trading Dashboard (Stats, Portfolio, Activity)",
            category=ToolCategory.UTILITY,
            parameters={},
            capabilities=["visualization", "dashboard", "ui"],
            handler=self._handle_show_dashboard,
        ))

        self.register_tool(ToolDefinition(
            name="solana_cluster_scan",
            description="Detect insider rings by analyzing correlation between multiple wallets",
            category=ToolCategory.ANALYSIS,
            parameters={
                "wallets": {"type": "array", "items": {"type": "string"}, "required": True, "description": "List of wallet addresses"},
            },
            capabilities=["cluster_analysis", "insider_trading", "forensics"],
            handler=self._handle_cluster_scan,
        ))

        self.register_tool(ToolDefinition(
            name="create_trade_video",
            description="Generate a 9:16 'Shorts' style video recapping a successful trade",
            category=ToolCategory.GENERATION,
            parameters={
                "ticker": {"type": "string", "required": True, "description": "Token Ticker (e.g. SOL)"},
                "pnl_str": {"type": "string", "required": True, "description": "Profit string (e.g. +$500)"},
                "roi_str": {"type": "string", "required": False, "description": "ROI string (e.g. +40%)"},
            },
            capabilities=["video_generation", "content_creation", "marketing"],
            handler=self._handle_trade_video,
        ))

    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a new tool.

        Args:
            tool: Tool definition to register
        """
        self.tools[tool.name] = tool

        # Generate embedding for semantic matching
        if self.embedder:
            description_text = f"{tool.name}: {tool.description}. Capabilities: {', '.join(tool.capabilities)}"
            try:
                self.tool_embeddings[tool.name] = self.embedder.embed(description_text)
            except Exception as e:
                logger.warning(f"Failed to embed tool {tool.name}: {e}")

        logger.debug(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if tool was removed
        """
        if name in self.tools:
            del self.tools[name]
            if name in self.tool_embeddings:
                del self.tool_embeddings[name]
            return True
        return False

    async def route(
        self,
        query: str,
        context: Optional[dict] = None,
        required_capabilities: Optional[list[str]] = None,
    ) -> list[ToolDefinition]:
        """
        Route a query to the most appropriate tools.

        Uses semantic matching and capability filtering.

        Args:
            query: User query or task description
            context: Additional context
            required_capabilities: Required tool capabilities

        Returns:
            List of matching tools, sorted by relevance
        """
        matching_tools = []

        for name, tool in self.tools.items():
            if not tool.enabled:
                continue

            score = 0.0

            # Capability matching
            if required_capabilities:
                capability_match = sum(
                    1 for cap in required_capabilities
                    if cap in tool.capabilities
                )
                if capability_match == 0:
                    continue
                score += capability_match * 0.3

            # Keyword matching
            query_lower = query.lower()
            if tool.name.lower() in query_lower:
                score += 0.4
            if any(cap in query_lower for cap in tool.capabilities):
                score += 0.2

            # Semantic matching (if embedder available)
            if self.embedder and name in self.tool_embeddings:
                try:
                    query_embedding = self.embedder.embed(query)
                    tool_embedding = self.tool_embeddings[name]
                    similarity = self._cosine_similarity(query_embedding, tool_embedding)
                    score += similarity * 0.3
                except Exception:
                    pass

            # Performance-based adjustment
            if tool.usage_count > 0:
                success_rate = tool.success_count / tool.usage_count
                score += success_rate * 0.1

            # Priority adjustment
            score += tool.priority * 0.01

            if score > 0:
                matching_tools.append((tool, score))

        # Sort by score descending
        matching_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in matching_tools]

    async def execute(
        self,
        tool_name: str,
        parameters: dict,
        context: Optional[dict] = None,
    ) -> ToolExecutionResult:
        """
        Execute a tool with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            context: Execution context

        Returns:
            ToolExecutionResult
        """
        if tool_name not in self.tools:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}",
            )

        tool = self.tools[tool_name]

        if not tool.enabled:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Tool disabled: {tool_name}",
            )

        if tool.handler is None:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Tool has no handler: {tool_name}",
            )

        # Validate required parameters
        for param_name, param_def in tool.parameters.items():
            if param_def.get("required", False) and param_name not in parameters:
                return ToolExecutionResult(
                    success=False,
                    output=None,
                    error=f"Missing required parameter: {param_name}",
                )

        # Execute with timing
        start_time = datetime.now()

        try:
            result = await tool.handler(**parameters)
            latency = (datetime.now() - start_time).total_seconds() * 1000

            # Update statistics
            tool.usage_count += 1
            tool.success_count += 1
            tool.avg_latency_ms = (
                tool.avg_latency_ms * (tool.usage_count - 1) + latency
            ) / tool.usage_count
            tool.last_used = datetime.now()

            return ToolExecutionResult(
                success=True,
                output=result,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            tool.usage_count += 1
            tool.last_used = datetime.now()

            logger.error(f"Tool execution failed: {tool_name}: {e}")

            return ToolExecutionResult(
                success=False,
                output=None,
                error=str(e),
                latency_ms=latency,
            )

    async def execute_chain(
        self,
        chain_name: str,
        initial_input: dict,
        context: Optional[dict] = None,
    ) -> list[ToolExecutionResult]:
        """
        Execute a chain of tools.

        Args:
            chain_name: Name of the tool chain
            initial_input: Initial input parameters
            context: Execution context

        Returns:
            List of results from each tool
        """
        if chain_name not in self.tool_chains:
            return [ToolExecutionResult(
                success=False,
                output=None,
                error=f"Chain not found: {chain_name}",
            )]

        results = []
        current_input = initial_input

        for tool_name in self.tool_chains[chain_name]:
            result = await self.execute(tool_name, current_input, context)
            results.append(result)

            if not result.success:
                break

            # Pass output to next tool
            if isinstance(result.output, dict):
                current_input.update(result.output)
            else:
                current_input["_previous_output"] = result.output

        return results

    def register_chain(self, name: str, tools: list[str]) -> bool:
        """
        Register a tool chain.

        Args:
            name: Chain name
            tools: List of tool names in order

        Returns:
            True if chain was registered
        """
        # Validate all tools exist
        for tool_name in tools:
            if tool_name not in self.tools:
                logger.warning(f"Cannot register chain: tool {tool_name} not found")
                return False

        self.tool_chains[name] = tools
        logger.info(f"Registered tool chain: {name} with {len(tools)} tools")
        return True

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        enabled_only: bool = True,
    ) -> list[ToolDefinition]:
        """
        List available tools.

        Args:
            category: Filter by category
            enabled_only: Only return enabled tools

        Returns:
            List of matching tools
        """
        tools = []
        for tool in self.tools.values():
            if enabled_only and not tool.enabled:
                continue
            if category and tool.category != category:
                continue
            tools.append(tool)
        return tools

    def get_statistics(self) -> dict:
        """Get usage statistics for all tools."""
        stats = {
            "total_tools": len(self.tools),
            "enabled_tools": sum(1 for t in self.tools.values() if t.enabled),
            "total_executions": sum(t.usage_count for t in self.tools.values()),
            "total_successes": sum(t.success_count for t in self.tools.values()),
            "by_category": {},
            "top_used": [],
        }

        # By category
        for cat in ToolCategory:
            cat_tools = [t for t in self.tools.values() if t.category == cat]
            if cat_tools:
                stats["by_category"][cat.value] = {
                    "count": len(cat_tools),
                    "total_usage": sum(t.usage_count for t in cat_tools),
                }

        # Top used
        sorted_tools = sorted(
            self.tools.values(),
            key=lambda t: t.usage_count,
            reverse=True,
        )[:10]
        stats["top_used"] = [
            {"name": t.name, "usage_count": t.usage_count, "success_rate": t.success_count / max(1, t.usage_count)}
            for t in sorted_tools
        ]

        return stats

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    # Built-in tool handlers

    async def _handle_read_file(self, path: str) -> str:
        """Read file contents."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    async def _handle_write_file(self, path: str, content: str) -> dict:
        """Write file contents."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"written": True, "path": path, "bytes": len(content.encode())}
        except Exception as e:
            raise RuntimeError(f"Failed to write file: {e}")

    async def _handle_list_directory(self, path: str) -> list[dict]:
        """List directory contents."""
        try:
            p = Path(path)
            items = []
            for item in p.iterdir():
                items.append({
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                })
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to list directory: {e}")

    async def _handle_execute_python(self, code: str) -> dict:
        """Execute Python code (sandboxed)."""
        # In production, this would use a proper sandbox
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Very basic sandboxing - in production use proper isolation
            exec_globals = {"__builtins__": {"print": print, "len": len, "range": range, "str": str, "int": int, "float": float}}
            exec(code, exec_globals)
            output = sys.stdout.getvalue()
            return {"output": output, "success": True}
        except Exception as e:
            return {"output": str(e), "success": False, "error": str(e)}
        finally:
            sys.stdout = old_stdout

    async def _handle_analyze_code(self, code: str, language: str = "python") -> dict:
        """Analyze code for issues."""
        # Placeholder - in production, use proper linting tools
        issues = []
        suggestions = []

        if language == "python":
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if "pass" in line and ":" in lines[max(0, i-1)]:
                    suggestions.append(f"Line {i+1}: Consider implementing the function body")
                if len(line) > 100:
                    issues.append(f"Line {i+1}: Line too long ({len(line)} chars)")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "lines_analyzed": len(code.split("\n")),
        }

    async def _handle_web_search(self, query: str, num_results: int = 5) -> list[dict]:
        """
        Search the web using DuckDuckGo (no API key required).

        Falls back to placeholder if network unavailable.
        """
        try:
            # Use DuckDuckGo HTML search (no API key needed)
            import urllib.request
            import urllib.parse
            import re

            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8")

            # Parse results from HTML
            results = []
            # Find result blocks
            result_pattern = r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]+)</a>'
            matches = re.findall(result_pattern, html, re.DOTALL)

            for match in matches[:num_results]:
                href, title, snippet = match
                # Clean up the redirect URL
                if "uddg=" in href:
                    actual_url = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
                else:
                    actual_url = href

                results.append({
                    "title": title.strip(),
                    "url": actual_url,
                    "snippet": snippet.strip()[:300],
                })

            if results:
                return results

            # Fallback if parsing failed
            logger.warning("Web search parsing returned no results, using fallback")

        except Exception as e:
            logger.warning(f"Web search failed: {e}, using fallback")

        # Fallback placeholder
        return [{
            "title": f"Search result for: {query}",
            "url": f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}",
            "snippet": "Live search unavailable. Click the URL to search manually.",
        }]

    async def _handle_fetch_url(self, url: str) -> dict:
        """Fetch URL content."""
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode("utf-8")
                return {
                    "url": url,
                    "status": response.status,
                    "content": content[:10000],  # Limit size
                }
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL: {e}")

    async def _handle_summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text using LLM if available, otherwise extractive summary.
        """
        # Try LLM-based summarization first
        if hasattr(self, 'llm_fn') and self.llm_fn:
            try:
                prompt = f"""Summarize the following text in {max_length} characters or less.
Be concise and capture the key points.

Text:
{text[:3000]}

Summary:"""
                if asyncio.iscoroutinefunction(self.llm_fn):
                    summary = await self.llm_fn(prompt)
                else:
                    summary = self.llm_fn(prompt)
                return summary.strip()[:max_length]
            except Exception as e:
                logger.debug(f"LLM summarization failed: {e}, using extractive")

        # Extractive summarization fallback
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 2:
            return text[:max_length]

        # Score sentences by position and length
        scored = []
        for i, sent in enumerate(sentences):
            # First and last sentences often contain key info
            position_score = 1.0 if i == 0 else (0.8 if i == len(sentences) - 1 else 0.5)
            # Prefer medium-length sentences
            length_score = min(len(sent) / 100, 1.0) * 0.5
            scored.append((sent, position_score + length_score))

        # Sort by score and take top sentences
        scored.sort(key=lambda x: x[1], reverse=True)

        summary = []
        current_length = 0
        for sent, _ in scored:
            if current_length + len(sent) > max_length:
                break
            summary.append(sent)
            current_length += len(sent) + 1

        # Reorder by original position
        summary_set = set(summary)
        ordered = [s for s in sentences if s in summary_set]

        return " ".join(ordered)[:max_length]

    async def _handle_extract_entities(self, text: str) -> list[dict]:
        """
        Extract named entities using pattern matching and heuristics.
        Uses LLM if available for better accuracy.
        """
        import re

        # Try LLM-based NER first
        if hasattr(self, 'llm_fn') and self.llm_fn:
            try:
                prompt = f"""Extract named entities from this text. Return JSON array with objects containing "text" and "type".
Types: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, EVENT, OTHER

Text: {text[:2000]}

Return ONLY the JSON array:"""
                if asyncio.iscoroutinefunction(self.llm_fn):
                    response = await self.llm_fn(prompt)
                else:
                    response = self.llm_fn(prompt)

                # Parse JSON from response
                import json
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    entities = json.loads(response[start:end])
                    return entities
            except Exception as e:
                logger.debug(f"LLM NER failed: {e}, using pattern matching")

        # Pattern-based entity extraction fallback
        entities = []
        seen = set()

        # Email patterns
        emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
        for email in emails:
            if email not in seen:
                entities.append({"text": email, "type": "EMAIL"})
                seen.add(email)

        # URL patterns
        urls = re.findall(r'https?://\S+', text)
        for url in urls:
            if url not in seen:
                entities.append({"text": url[:100], "type": "URL"})
                seen.add(url)

        # Date patterns
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text, re.I)
        for date in dates:
            if date not in seen:
                entities.append({"text": date, "type": "DATE"})
                seen.add(date)

        # Money patterns
        money = re.findall(r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP)\b', text, re.I)
        for m in money:
            if m not in seen:
                entities.append({"text": m, "type": "MONEY"})
                seen.add(m)

        # Capitalized phrases (potential names/organizations)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        for cap in caps:
            if cap not in seen and len(cap) > 3:
                # Heuristic: if contains common name patterns
                if any(title in cap for title in ['Mr', 'Mrs', 'Dr', 'Prof']):
                    entities.append({"text": cap, "type": "PERSON"})
                elif any(suffix in cap for suffix in ['Inc', 'Corp', 'LLC', 'Ltd', 'Company']):
                    entities.append({"text": cap, "type": "ORGANIZATION"})
                else:
                    entities.append({"text": cap, "type": "UNKNOWN"})
                seen.add(cap)

        return entities[:50]  # Limit results

    async def _handle_generate_image(self, prompt: str, size: str = "512x512") -> dict:
        """
        Generate an image description or placeholder.
        Returns metadata about what would be generated.
        """
        import os

        # Check for API keys
        openai_key = os.environ.get("OPENAI_API_KEY")
        stability_key = os.environ.get("STABILITY_API_KEY")

        if openai_key:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/images/generations",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        json={
                            "prompt": prompt,
                            "n": 1,
                            "size": size if size in ["256x256", "512x512", "1024x1024"] else "512x512",
                        },
                        timeout=60.0,
                    )
                    result = response.json()
                    if "data" in result and result["data"]:
                        return {
                            "prompt": prompt,
                            "size": size,
                            "url": result["data"][0].get("url"),
                            "provider": "openai",
                        }
            except Exception as e:
                logger.warning(f"OpenAI image generation failed: {e}")

        # Return descriptive placeholder
        return {
            "prompt": prompt,
            "size": size,
            "url": None,
            "description": f"Image generation requested: '{prompt[:100]}' at {size}",
            "note": "Set OPENAI_API_KEY or STABILITY_API_KEY environment variable to enable image generation",
            "provider": "none",
        }

    async def _handle_calculate(self, expression: str) -> float:
        """Perform calculation."""
        import math

        # Safe evaluation with limited builtins
        safe_dict = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Calculation error: {e}")

    async def _handle_datetime(self, timezone: str = "UTC") -> dict:
        """Get datetime information."""
        from datetime import timezone as tz

        now = datetime.now()
        return {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": timezone,
            "unix_timestamp": now.timestamp(),
        }

    async def _handle_grok_search(self, query: str) -> dict:
        """Handle Grok search."""
        import os
        from Cosmos.integration.external.grok import create_grok_provider
        
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            return {"error": "XAI_API_KEY not set in environment."}
            
        provider = create_grok_provider(api_key)
        return await provider.execute_action("grok_search", {"query": query})

    async def _handle_remotion_render(self, narrative: str, composition_id: str = "Main") -> dict:
        """Handle Remotion rendering."""
        from Cosmos.integration.video_gen import remotion_skill
        output = f"output_{int(datetime.now().timestamp())}.mp4"
        success = await remotion_skill.render_video(composition_id, {"text": narrative}, output)
        return {"success": success, "output": output}

    async def _handle_parallel_ai(self, prompt: str) -> dict:
        """Handle Parallel AI dispatch."""
        from Cosmos.core.parallel_orchestrator import create_parallel_orchestrator
        from Cosmos.core.llm_backend import llm_backend # Assume this exists and has a generate method
        
        # In actual usage, we'd pull these from the model_manager
        backends = [llm_backend.generate] # placeholder for multi-backend list
        orchestrator = create_parallel_orchestrator(backends)
        result = await orchestrator.fused_consensus(prompt)
        return {"consensus_result": result}

    async def _handle_discord_send(self, channel_id: int, content: str) -> dict:
        """Handle Discord message sending."""
        from Cosmos.integration.external.discord_ext import discord_bridge
        await discord_bridge.send_message(channel_id, content)
        return {"status": "dispatched", "channel": channel_id}

    async def _handle_mermaid_gen(self, diagram_type: str, data: dict) -> dict:
        """Handle Mermaid diagram generation."""
        from Cosmos.integration.diagrams import diagram_skill
        if diagram_type == "flowchart":
            code = diagram_skill.generate_mermaid_flowchart(data.get("nodes", []), data.get("edges", []))
        elif diagram_type == "sequence":
            code = diagram_skill.generate_sequence_diagram(data.get("participants", []), data.get("messages", []))
        else:
            return {"error": f"Diagram type {diagram_type} not supported."}
        return {"code": code}

    async def _handle_system_diag(self) -> dict:
        """Handle System Diagnostics."""
        from Cosmos.os_integration.agentic_os import agentic_os
        return {
            "load": agentic_os.get_system_load(),
            "processes_top": agentic_os.list_processes()[:5],
            "network": agentic_os.get_network_stats()
        }

    async def _handle_youtube_analyze(self, video_url: str) -> dict:
        """Handle YouTube analysis."""
        from Cosmos.integration.external.youtube import youtube_skill
        vid_id = youtube_skill.extract_id(video_url)
        if not vid_id:
            return {"error": "Invalid YouTube URL."}
        transcript = await youtube_skill.get_transcript(vid_id)
        return {"video_id": vid_id, "transcript_preview": transcript[:1000]}

    async def _handle_sequential_thought(self, thought: str, verification: str = "", is_new_chain: bool = False) -> dict:
        """Handle Sequential Thinking."""
        from Cosmos.core.cognition.sequential_thinking import sequential_thinker
        if is_new_chain:
            sequential_thinker.start_new_chain()
        step = sequential_thinker.add_step(thought, verification)
        return {"step": step.step_number, "chain_summary": sequential_thinker.get_summary()}

    async def _handle_db_query(self, query: str) -> dict:
        """Handle Database Querying."""
        from Cosmos.integration.external.db_manager import db_skill
        results = await db_skill.execute_query(query)
        return {"results": results}

    async def _handle_dex_search(self, query: str) -> dict:
        """Handle DexScreener search."""
        from Cosmos.integration.financial.dexscreener import dex_screener
        pairs = await dex_screener.search_pairs(query)
        return {"pairs": pairs[:5]} # Top 5 results

    async def _handle_poly_scan(self, query: str) -> dict:
        """Handle Polymarket scanning."""
        from Cosmos.integration.financial.polymarket import polymarket
        events = await polymarket.search_markets(query)
        return {"events": events[:5]}

    async def _handle_sentiment_check(self) -> dict:
        """Handle Market Sentiment check."""
        from Cosmos.integration.financial.market_sentiment import market_sentiment
        fng = await market_sentiment.get_fear_and_greed()
        global_data = await market_sentiment.get_global_market_cap()
        btc_price = await market_sentiment.get_token_price("bitcoin")
        return {
            "fear_and_greed": fng,
            "global_market": global_data,
            "bitcoin": btc_price
        }

    async def _handle_pump_track(self, mint_address: str) -> dict:
        """Handle Pump.fun tracking."""
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker
        return await memecoin_tracker.get_pump_token(mint_address)

    async def _handle_bags_track(self, token_address: str) -> dict:
        """Handle Bags.fm tracking."""
        import os
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker
        api_key = os.environ.get("BAGS_API_KEY")
        if api_key:
            memecoin_tracker.set_bags_api_key(api_key)
        return await memecoin_tracker.get_bags_token(token_address)

    async def _handle_bags_trending(self) -> dict:
        """Handle Bags.fm trending tokens."""
        import os
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker
        api_key = os.environ.get("BAGS_API_KEY")
        if api_key:
            memecoin_tracker.set_bags_api_key(api_key)
        tokens = await memecoin_tracker.get_bags_trending()
        return {"trending": tokens, "count": len(tokens)}

    async def _handle_bags_quote(self, token_mint: str, amount: float, side: str, slippage_bps: int = 100) -> dict:
        """Handle Bags.fm trade quote."""
        import os
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker
        api_key = os.environ.get("BAGS_API_KEY")
        if api_key:
            memecoin_tracker.set_bags_api_key(api_key)
        return await memecoin_tracker.get_trade_quote(token_mint, amount, side, slippage_bps)

    async def _handle_bags_swap(self, token_mint: str, amount: float, side: str, slippage_bps: int = 100) -> dict:
        """Handle Bags.fm swap transaction."""
        import os
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker, BagsSwapParams
        api_key = os.environ.get("BAGS_API_KEY")
        if api_key:
            memecoin_tracker.set_bags_api_key(api_key)
        swap_params = BagsSwapParams(
            token_mint=token_mint,
            amount=amount,
            side=side,
            slippage_bps=slippage_bps
        )
        return await memecoin_tracker.create_swap_transaction(swap_params)

    async def _handle_bags_creators(self, token_mint: str) -> dict:
        """Handle Bags.fm token creators lookup."""
        import os
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker
        api_key = os.environ.get("BAGS_API_KEY")
        if api_key:
            memecoin_tracker.set_bags_api_key(api_key)
        return await memecoin_tracker.get_token_launch_creators(token_mint)

    async def _handle_bags_fees(self, token_mint: str) -> dict:
        """Handle Bags.fm token fees lookup."""
        import os
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker
        api_key = os.environ.get("BAGS_API_KEY")
        if api_key:
            memecoin_tracker.set_bags_api_key(api_key)
        return await memecoin_tracker.get_token_lifetime_fees(token_mint)

    async def _handle_bags_claimable(self, wallet_address: str) -> dict:
        """Handle Bags.fm claimable positions lookup."""
        import os
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker
        api_key = os.environ.get("BAGS_API_KEY")
        if api_key:
            memecoin_tracker.set_bags_api_key(api_key)
        return await memecoin_tracker.get_claimable_positions(wallet_address)

    async def _handle_bags_partner_stats(self, wallet_address: str) -> dict:
        """Handle Bags.fm partner stats lookup."""
        import os
        from Cosmos.integration.financial.memecoin_tracker import memecoin_tracker
        api_key = os.environ.get("BAGS_API_KEY")
        if api_key:
            memecoin_tracker.set_bags_api_key(api_key)
        return await memecoin_tracker.get_partner_stats(wallet_address)

    async def _handle_solana_balance(self, pubkey: str = None) -> dict:
        """Handle Solana balance check."""
        from Cosmos.integration.solana.trading import solana_trader
        balance = await solana_trader.get_balance(pubkey)
        return {"balance_sol": balance}

    async def _handle_jup_swap(self, input_mint: str, output_mint: str, amount: int) -> dict:
        """Handle Jupiter swap with exception mapping."""
        from Cosmos.integration.solana.trading import solana_trader
        from Cosmos.core.cognition.trading_cognition import exception_manager
        
        return await exception_manager.handle_execution(
            solana_trader.jupiter_swap(input_mint, output_mint, amount)
        )

    async def _handle_pump_trade(self, action: str, mint: str, amount: float) -> dict:
        """Handle Pump.fun trading with exception mapping."""
        from Cosmos.integration.solana.trading import solana_trader
        from Cosmos.core.cognition.trading_cognition import exception_manager

        return await exception_manager.handle_execution(
            solana_trader.pump_fun_trade(action, mint, amount)
        )

    async def _handle_meteora_info(self, pair_address: str) -> dict:
        """Handle Meteora info."""
        from Cosmos.integration.solana.trading import solana_trader
        return await solana_trader.meteora_info(pair_address)

    async def _handle_trade_evaluation(self, token_symbol: str, liquidity: float = 0, volume_24h: float = 0) -> dict:
        """Handle cognitive signal evaluation."""
        from Cosmos.core.cognition.trading_cognition import trading_cognition
        token_data = {
            "symbol": token_symbol,
            "liquidity": {"usd": liquidity},
            "volume": {"h24": volume_24h}
        }
        return await trading_cognition.evaluate_token(token_data)

    async def _handle_rug_check(self, mint_address: str) -> dict:
        """Handle Rug Check."""
        from Cosmos.integration.solana.degen_mob import degen_mob
        return await degen_mob.analyze_token_safety(mint_address)

    async def _handle_whale_watch(self, wallet_address: str, action: str = "history") -> dict:
        """Handle Whale Watching."""
        from Cosmos.integration.solana.degen_mob import degen_mob
        if action == "add":
            degen_mob.add_whale_wallet(wallet_address)
            return {"status": "added", "wallet": wallet_address}
        
        history = await degen_mob.get_whale_recent_activity(wallet_address)
        return {"wallet": wallet_address, "recent_activity": history}

    async def _handle_snipe_toggle(self, enabled: bool, query: str = None) -> dict:
        """Handle Sniper Toggle."""
        from Cosmos.integration.solana.degen_mob import degen_mob
        if enabled:
            await degen_mob.start_sniper(query)
            return {"status": "Sniper STARTED"}
        else:
            degen_mob.stop_sniper()
            return {"status": "Sniper STOPPED"}

    async def _handle_ct_swarm(self, keywords: List[str]) -> dict:
        """Handle CT Alpha analysis."""
        from Cosmos.integration.solana.degen_mob import degen_mob
        await degen_mob.run_alpha_leak_loop(keywords)
        return {"status": "Alpha swarm dispatched", "tracking": keywords}

    async def _handle_show_dashboard(self) -> dict:
        """Handle showing the dashboard."""
        import os
        ui_path = os.path.abspath("cosmos/ui/dashboard.html")
        return {
            "status": "Visualization Ready",
            "ui_url": f"file://{ui_path}",
            "message": "The Professor's specialized trading dashboard has been generated."
        }

    async def _handle_cluster_scan(self, wallets: List[str]) -> dict:
        """Handle Cluster Analysis."""
        from Cosmos.integration.solana.degen_mob import degen_mob
        return await degen_mob.analyze_wallet_cluster(wallets)

    async def _handle_trade_video(self, ticker: str, pnl_str: str, roi_str: str = "0%") -> dict:
        """Handle Trade Recap Video Generation."""
        from Cosmos.integration.video_gen import remotion_skill
        trade_data = {"ticker": ticker, "pnl_str": pnl_str, "roi_str": roi_str}
        path = await remotion_skill.generate_trade_recap(trade_data)
        return {"status": "Video Rendered", "path": path}

    async def _handle_multimodal_analysis(self, image_path: str = None, audio_path: str = None, text_context: str = None) -> dict:
        """Handle 12D Multimodal Analysis."""
        from Cosmos.core.multimodal import get_multimodal_system
        from PIL import Image
        import numpy as np
        
        system = get_multimodal_system()
        if not system:
            return {"error": "Multimodal system unavailable."}
            
        image_data = None
        # Load image if provided
        if image_path:
            try:
                img = Image.open(image_path).convert("RGB")
                image_data = np.array(img)
            except Exception as e:
                return {"error": f"Failed to load image: {e}"}
        
        token, emotion, thought = system.process_multimodal_input(
            image=image_data,
            text=text_context
        )
        
        return {
            "thought": thought,
            "emotional_state": {
                "valence": float(emotion.valence),
                "arousal": float(emotion.arousal),
                "dominance": float(emotion.dominance),
                "label": emotion.classify_emotion()
            },
            "12d_embedding_preview": token.embedding_12d.tolist()[:3] # First 3 dims
        }











