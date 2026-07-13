"""Kimi Router web search plugin — auto-loaded, bundled."""
from __future__ import annotations

from plugins.web.kimi_router.provider import KimiRouterProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(KimiRouterProvider())
