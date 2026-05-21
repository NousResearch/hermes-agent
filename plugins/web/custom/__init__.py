"""Custom OpenAI-compatible web search + extract plugin — bundled, auto-loaded.

Routes search/extract calls through any chat-completions endpoint that
serves a model with built-in web access (e.g. Perplexity Sonar, OpenAI
gpt-4o with browsing). Search-result extraction prefers a structured
``search_results`` field, then a ``citations`` list, then falls back to
the answer text itself.
"""

from __future__ import annotations

from plugins.web.custom.provider import CustomWebSearchProvider


def register(ctx) -> None:
    """Register the custom provider with the plugin context."""
    ctx.register_web_search_provider(CustomWebSearchProvider())
