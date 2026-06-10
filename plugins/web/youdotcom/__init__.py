"""You.com web search + content extraction plugin."""

from __future__ import annotations

from plugins.web.youdotcom.provider import YoudotcomWebSearchProvider


def register(ctx) -> None:
    """Register the You.com provider with the plugin context."""
    ctx.register_web_search_provider(YoudotcomWebSearchProvider())
