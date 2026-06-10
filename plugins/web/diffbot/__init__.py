from __future__ import annotations

from .provider import DiffbotWebSearchProvider


def register(ctx) -> None:
    """Register the Diffbot provider with the plugin context."""
    ctx.register_web_search_provider(DiffbotWebSearchProvider())
