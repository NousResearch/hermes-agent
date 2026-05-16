"""MiniMax web search plugin — bundled, auto-loaded.

Uses direct API calls to perform web searches via the MiniMax AI platform.
"""

from __future__ import annotations

from plugins.web.minimax.provider import MiniMaxWebSearchProvider


def register(ctx) -> None:
    """Register the MiniMax provider with the plugin context."""
    ctx.register_web_search_provider(MiniMaxWebSearchProvider())
