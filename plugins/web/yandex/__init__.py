"""Yandex Cloud Search API plugin — bundled web search backend."""

from __future__ import annotations

from plugins.web.yandex.provider import YandexWebSearchProvider


def register(ctx) -> None:
    """Register the Yandex web search provider with the plugin context."""
    ctx.register_web_search_provider(YandexWebSearchProvider())
