"""Camofox web provider — bundled, auto-loaded.

Backed by the local Camofox/Camoufox browser server. This lets the generic
``web_search`` and ``web_extract`` tools use the same anti-detection browser
stack as the ``browser_*`` tools when ``CAMOFOX_URL`` is configured.
"""

from __future__ import annotations

from plugins.web.camofox.provider import CamofoxWebSearchProvider


def register(ctx) -> None:
    """Register the Camofox provider with the plugin context."""
    ctx.register_web_search_provider(CamofoxWebSearchProvider())
