"""Firecrawl web search + extract plugin — bundled, auto-loaded."""
from __future__ import annotations
from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(FirecrawlWebSearchProvider())
