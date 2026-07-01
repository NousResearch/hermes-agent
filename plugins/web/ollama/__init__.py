"""Ollama Cloud web search + fetch plugin — bundled, auto-loaded.

Registers the Ollama Cloud provider so ``web_search`` and ``web_extract`` can
route through Ollama's official web_search and web_fetch endpoints.
"""

from __future__ import annotations

from plugins.web.ollama.provider import OllamaWebSearchProvider


def register(ctx) -> None:
    """Register the Ollama Cloud provider with the plugin context."""
    ctx.register_web_search_provider(OllamaWebSearchProvider())
