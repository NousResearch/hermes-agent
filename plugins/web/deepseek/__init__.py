"""DeepSeek web search + fetch plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.deepseek.provider import DeepSeekWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(DeepSeekWebSearchProvider())
