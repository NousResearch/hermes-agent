"""Local web extraction plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.local.provider import LocalWebExtractProvider


def register(ctx) -> None:
    """Register the local HTML extraction provider with the plugin context."""
    ctx.register_web_search_provider(LocalWebExtractProvider())
