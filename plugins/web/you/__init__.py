"""You.com Search plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.you.provider import YouWebSearchProvider


def register(ctx) -> None:
    """Register the You.com Search provider with the plugin context."""
    ctx.register_web_search_provider(YouWebSearchProvider())
