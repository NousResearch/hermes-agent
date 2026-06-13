"""fastCRW web search + extract plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.crw.provider import CrwWebSearchProvider


def register(ctx) -> None:
    """Register the fastCRW provider with the plugin context."""
    ctx.register_web_search_provider(CrwWebSearchProvider())
