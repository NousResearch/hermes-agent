"""OpenRouter agentic web search + content fetch plugin — bundled, auto-loaded.

Leverages OpenRouter's server-side tools (``openrouter:web_search`` /
``openrouter:web_fetch``) so any model can search and fetch the web
through a single lightweight chat completions call. Uses your existing
OpenRouter credits — no separate Exa / Firecrawl / Parallel API key needed.
"""

from __future__ import annotations

from plugins.web.openrouter.provider import OpenRouterWebSearchProvider


def register(ctx) -> None:
    """Register the OpenRouter web search provider with the plugin context."""
    ctx.register_web_search_provider(OpenRouterWebSearchProvider())