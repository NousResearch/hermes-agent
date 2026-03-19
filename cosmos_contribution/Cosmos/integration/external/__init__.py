"""
cosmos External AI Provider Integrations.

This module provides integrations with external AI services:
- Claude Code CLI (Anthropic, via authenticated CLI)
- Kimi (Moonshot AI, long-context reasoning)
- Grok (xAI)
- And more...

These providers participate in the multi-model swarm orchestration.
"""

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus

# Lazy imports for optional providers
__all__ = [
    "ExternalProvider",
    "IntegrationConfig",
    "ConnectionStatus",
]
