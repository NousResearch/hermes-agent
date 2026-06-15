"""iFlow Search plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.iflow.provider import IFlowWebSearchProvider


def register(ctx) -> None:
    """Register the iFlow Search provider with the plugin context."""
    ctx.register_web_search_provider(IFlowWebSearchProvider())
