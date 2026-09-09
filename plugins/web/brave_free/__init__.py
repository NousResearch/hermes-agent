"""Brave Search (free tier) plugin — bundled, auto-loaded."""
from __future__ import annotations
from plugins.web.brave_free.provider import BraveFreeWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(BraveFreeWebSearchProvider())
