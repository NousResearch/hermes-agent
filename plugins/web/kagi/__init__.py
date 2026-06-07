"""Kagi web search plugin — bundled, auto-loaded.

Calls Kagi Search API v1 (POST /api/v1/search) and Extract API
(POST /api/v1/extract) directly. Requires ``KAGI_API_KEY`` in
``~/.hermes/.env``.
"""

from __future__ import annotations

from plugins.web.kagi.provider import KagiWebSearchProvider


def register(ctx) -> None:
    """Register the Kagi provider with the plugin context."""
    ctx.register_web_search_provider(KagiWebSearchProvider())
