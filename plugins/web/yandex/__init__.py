"""Yandex Cloud AI Studio web search plugin — bundled, auto-loaded."""
from __future__ import annotations
from plugins.web.yandex.provider import YandexWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(YandexWebSearchProvider())
