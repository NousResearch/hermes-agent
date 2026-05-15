"""Gemini web search + extract plugin — bundled, auto-loaded.

Backed by the official Google GenAI SDK (``google-genai``) using Google Search Grounding.
"""

from __future__ import annotations

from plugins.web.gemini.provider import GeminiWebSearchProvider


def register(ctx) -> None:
    """Register the Gemini provider with the plugin context."""
    ctx.register_web_search_provider(GeminiWebSearchProvider())
