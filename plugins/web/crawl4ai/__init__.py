"""Crawl4AI local-read + Firecrawl-fallback plugin — bundled, auto-loaded.

``provider.py`` holds the provider class; ``register(ctx)`` registers an
instance into the web search registry (extract capability only).
"""

from __future__ import annotations

from plugins.web.crawl4ai.provider import Crawl4aiWebSearchProvider


def register(ctx) -> None:
    """Register the Crawl4AI local-read provider with the plugin context."""
    ctx.register_web_search_provider(Crawl4aiWebSearchProvider())
