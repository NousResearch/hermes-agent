"""MiniMax Token Plan web search — user plugin.

Mirrors the ``plugins/web/brave_free/`` layout:

    provider.py    WebSearchProvider subclass + helpers
    __init__.py    register(ctx) hook
    plugin.yaml    kind: backend, provides_web_providers: [minimax]

Hermes auto-loads this from ``~/.hermes/plugins/web/minimax/`` once it is
listed under ``plugins.enabled`` in ``config.yaml``.
"""

from __future__ import annotations

from plugins.web.minimax.provider import MiniMaxWebSearchProvider


def register(ctx) -> None:
    """Register the MiniMax provider with the plugin context."""
    ctx.register_web_search_provider(MiniMaxWebSearchProvider())
