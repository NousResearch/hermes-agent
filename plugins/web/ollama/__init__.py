"""Ollama Cloud web search + extract plugin — bundled, auto-loaded.

Backed by the Ollama Cloud API (https://ollama.com/api). Both search and
extract are sync; the dispatcher in :mod:`tools.web_tools` handles the
wrap when the caller is async.
"""

from __future__ import annotations

from plugins.web.ollama.provider import OllamaWebSearchProvider


def register(ctx) -> None:
    """Register the Ollama Cloud provider with the plugin context."""
    ctx.register_web_search_provider(OllamaWebSearchProvider())