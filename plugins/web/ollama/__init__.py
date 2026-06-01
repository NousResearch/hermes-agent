"""Ollama Cloud web search + extract plugin — bundled, auto-loaded.

Backed by Ollama Cloud API (requires ``OLLAMA_API_KEY`` from
https://ollama.com/settings/keys). Search and content extraction in
one provider.
"""

from __future__ import annotations

from plugins.web.ollama.provider import OllamaWebSearchProvider


def register(ctx) -> None:
    """Register the Ollama provider with the plugin context."""
    ctx.register_web_search_provider(OllamaWebSearchProvider())
