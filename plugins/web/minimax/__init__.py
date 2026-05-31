"""MiniMax web search plugin."""

from __future__ import annotations

from plugins.web.minimax.provider import MiniMaxWebSearchProvider


def register(ctx) -> None:
    """Register the MiniMax provider with the plugin context."""
    ctx.register_web_search_provider(MiniMaxWebSearchProvider())
