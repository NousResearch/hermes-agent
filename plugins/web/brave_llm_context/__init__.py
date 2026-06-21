"""Brave Search LLM Context plugin -- bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.brave_llm_context.provider import BraveLLMContextWebSearchProvider


def register(ctx) -> None:
    """Register the Brave LLM Context provider with the plugin context."""
    ctx.register_web_search_provider(BraveLLMContextWebSearchProvider())
