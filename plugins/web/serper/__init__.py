"""Serper.dev web search plugin — bundled, auto-loaded.

Provides Google Search API results via serper.dev free tier.
"""

from __future__ import annotations

from plugins.web.serper.provider import SerperWebSearchProvider


def register(ctx) -> None:
    """Register the Serper.dev provider with the plugin context."""
    ctx.register_web_search_provider(SerperWebSearchProvider())
