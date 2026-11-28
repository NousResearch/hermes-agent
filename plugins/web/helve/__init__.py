"""Helve web search + extract plugin — bundled, auto-loaded."""
from __future__ import annotations

from plugins.web.helve.provider import HelveWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(HelveWebSearchProvider())
