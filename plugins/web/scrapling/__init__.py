"""Scrapling web search + extraction — local, no API key, no rate limit.

Bundled, auto-loaded. Registers as a keyless-tier provider and is also the
5th rung of the keyless ring (``plugins/web/keyless_mcp.py``).
"""
from __future__ import annotations

from plugins.web.scrapling.provider import ScraplingWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(ScraplingWebSearchProvider())
