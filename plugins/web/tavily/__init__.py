"""Tavily web search + extract + crawl plugin — bundled, auto-loaded.

First plugin in this codebase to advertise ``supports_crawl=True``. The
crawl method maps to Tavily's ``/crawl`` endpoint, which accepts a seed
URL plus optional instructions and extract depth.
"""

from __future__ import annotations

from plugins.web.tavily.provider import TavilyWebSearchProvider


def register(ctx) -> None:
    """Register the Tavily provider with the plugin context."""
    ctx.register_web_search_provider(TavilyWebSearchProvider())
