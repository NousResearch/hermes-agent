"""AnySearch search plugin — bundled, auto-loaded."""
from __future__ import annotations

from plugins.web.anysearch.provider import AnySearchProvider


def register(ctx) -> None:
    """Register the AnySearch provider with the plugin context."""
    ctx.register_web_search_provider(AnySearchProvider())
