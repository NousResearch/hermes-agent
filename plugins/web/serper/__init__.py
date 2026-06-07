"""Serper web search plugin — bundled, auto-loaded.

Search-only backend powered by Serper.dev's Google Search API.
"""

from __future__ import annotations

from plugins.web.serper.provider import SerperWebSearchProvider


def register(ctx) -> None:
    """Register the Serper provider with the plugin context."""
    ctx.register_web_search_provider(SerperWebSearchProvider())
