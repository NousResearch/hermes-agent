"""Bocha web search plugin - bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.bocha.provider import BochaWebSearchProvider


def register(ctx) -> None:
    """Register the Bocha provider with the plugin context."""
    ctx.register_web_search_provider(BochaWebSearchProvider())
