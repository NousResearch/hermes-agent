"""Perplexity web search plugin — bundled, auto-loaded."""

from __future__ import annotations

from tools.web_providers.perplexity import PerplexitySearchProvider


def register(ctx) -> None:
    """Register the Perplexity Search API provider with the plugin context."""
    ctx.register_web_search_provider(PerplexitySearchProvider())
