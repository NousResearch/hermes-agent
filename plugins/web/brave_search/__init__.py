"""Brave Search API plugin, bundled and auto-loaded."""

from __future__ import annotations

from plugins.web.brave_search.provider import BraveSearchWebProvider


def register(ctx) -> None:
    """Register the Brave Search API provider with the plugin context."""
    ctx.register_web_search_provider(BraveSearchWebProvider())
