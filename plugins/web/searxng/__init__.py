"""SearXNG search-only plugin — bundled, auto-loaded (``SEARXNG_URL``)."""
from __future__ import annotations
from plugins.web.searxng.provider import SearXNGWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(SearXNGWebSearchProvider())
