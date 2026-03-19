"""
CLI Bridge — Turn free AI CLIs into a unified swarm API.

"Any CLI that speaks, the swarm can hear." - The Collective

This package provides:
- CLIBridge ABC for wrapping CLI tools as async providers
- ClaudeCodeBridge for Claude Code CLI (Max subscription)
- GeminiCLIBridge for Gemini CLI (1,000 free req/day + Google Search)
- CLICapabilityRouter for smart routing + fallback chains
- RateTracker for per-CLI daily/minute quota tracking
- InteractiveCLISession for tmux-based persistent agents

Usage:
    from Cosmos.integration.cli_bridge import create_router

    router = await create_router()
    response = await router.query_with_fallback("What's trending in crypto?")
"""

from .base import CLIBridge, CLICapability, CLIHealth, CLIResponse
from .capability_router import CLICapabilityRouter, get_cli_router
from .rate_tracker import RateTracker, get_rate_tracker

__all__ = [
    "CLIBridge",
    "CLICapability",
    "CLIHealth",
    "CLIResponse",
    "CLICapabilityRouter",
    "RateTracker",
    "create_router",
    "get_cli_router",
    "get_rate_tracker",
]


async def create_router() -> CLICapabilityRouter:
    """
    Factory function to create and initialize the CLI capability router.

    Discovers available CLIs, health-checks them, and returns
    a singleton router ready for queries.

    Usage:
        router = await create_router()
        response = await router.query_with_fallback("Hello world")
    """
    return await get_cli_router()
