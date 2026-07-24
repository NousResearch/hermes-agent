"""Bundled local sxng-search web provider plugin."""
from __future__ import annotations

from plugins.web.sxng.provider import SxngWebSearchProvider


def register(ctx) -> None:
    """Register the sxng provider with the plugin context."""
    ctx.register_web_search_provider(SxngWebSearchProvider())
