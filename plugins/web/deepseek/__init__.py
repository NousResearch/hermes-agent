"""DeepSeek web search plugin — user plugin, adapted from bundled web-xai.

``provider.py`` holds the provider class, ``__init__.py::register(ctx)``
registers an instance with the plugin context (same layout as
``plugins/web/brave_free/``).
"""

from __future__ import annotations

from .provider import DeepSeekWebSearchProvider


def register(ctx) -> None:
    """Register the DeepSeek Web Search provider with the plugin context."""
    ctx.register_web_search_provider(DeepSeekWebSearchProvider())
