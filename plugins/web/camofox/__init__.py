"""Camofox web extract plugin — bundled, auto-loaded.

Backed by a self-hosted Camofox browser server. Extract-only — pair with a
search provider (searxng) for ``web_search`` calls.
"""

from __future__ import annotations

from plugins.web.camofox.provider import CamofoxWebExtractProvider


def register(ctx) -> None:
    """Register the Camofox extract provider with the plugin context."""
    ctx.register_web_search_provider(CamofoxWebExtractProvider())
