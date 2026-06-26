"""Plugin entrypoint — registers the provider with Hermes's plugin manager."""

from __future__ import annotations

from plugins.web.bocha.provider import BochaWebSearchProvider


def register(ctx) -> None:
    """Register the provider with the plugin context."""
    ctx.register_web_search_provider(BochaWebSearchProvider())
