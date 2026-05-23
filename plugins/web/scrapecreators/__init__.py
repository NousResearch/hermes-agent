"""ScrapeCreators web search plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.scrapecreators.provider import ScrapeCreatorsWebSearchProvider


def register(ctx) -> None:
    """Register the ScrapeCreators Google Search provider."""
    ctx.register_web_search_provider(ScrapeCreatorsWebSearchProvider())
