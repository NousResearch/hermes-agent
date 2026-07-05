"""Yandex Search API plugin — bundled, auto-loaded.

Mirrors the ``plugins/web/brave_free/`` layout: ``provider.py`` holds the
provider class, ``__init__.py::register(ctx)`` registers an instance.
"""

from __future__ import annotations

from plugins.web.yandex.provider import YandexWebSearchProvider


def register(ctx) -> None:
    """Register the Yandex Search provider with the plugin context."""
    ctx.register_web_search_provider(YandexWebSearchProvider())
