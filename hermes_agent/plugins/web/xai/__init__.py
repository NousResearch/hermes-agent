"""xAI web search plugin — bundled, auto-loaded."""
from __future__ import annotations
from hermes_agent.plugins.web.xai.provider import XAIWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(XAIWebSearchProvider())
