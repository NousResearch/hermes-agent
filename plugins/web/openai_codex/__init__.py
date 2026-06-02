"""OpenAI Codex OAuth web search + extract plugin — bundled, auto-loaded."""
from __future__ import annotations

from plugins.web.openai_codex.provider import OpenAICodexWebSearchProvider


def register(ctx) -> None:
    """Register the OpenAI Codex OAuth web provider with the plugin context."""
    ctx.register_web_search_provider(OpenAICodexWebSearchProvider())
