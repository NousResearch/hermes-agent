"""Keenable web search + extract plugin — bundled, auto-loaded; keyless-ring member."""
from __future__ import annotations
from plugins.web.keenable.provider import KeenableWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(KeenableWebSearchProvider())
