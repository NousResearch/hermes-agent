"""OpenAI Codex web search plugin — bundled. Mirrors plugins/web/xai/ layout."""
from __future__ import annotations

from plugins.web.openai_codex.provider import OpenAICodexWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(OpenAICodexWebSearchProvider())
