"""LLMLayer web search + scrape plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.llmlayer.provider import LLMLayerWebSearchProvider


def register(ctx) -> None:
    """Register the LLMLayer provider with the plugin context."""
    ctx.register_web_search_provider(LLMLayerWebSearchProvider())
