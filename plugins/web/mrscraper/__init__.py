"""MrScraper web provider and native tool plugin."""

from __future__ import annotations

from plugins.mrscraper_client import is_mrscraper_available
from plugins.web.mrscraper.provider import MrScraperWebSearchProvider
from plugins.web.mrscraper.tools import MRSCRAPER_TOOLS


def register(ctx) -> None:
    """Register the web provider and fourteen non-rendered native tools."""
    ctx.register_web_search_provider(MrScraperWebSearchProvider())
    for name, schema, handler in MRSCRAPER_TOOLS:
        ctx.register_tool(
            name=name,
            toolset="mrscraper",
            schema=schema,
            handler=handler,
            check_fn=is_mrscraper_available,
            requires_env=["MRSCRAPER_API_TOKEN"],
            emoji="🕷️",
        )
