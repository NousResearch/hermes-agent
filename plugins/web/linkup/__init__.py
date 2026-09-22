"""Linkup web search + page fetch plugin — bundled, auto-loaded."""
from __future__ import annotations
from plugins.web.linkup.provider import LinkupWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(LinkupWebSearchProvider())
