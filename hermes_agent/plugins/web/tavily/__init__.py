"""Tavily web search + extract plugin — bundled, auto-loaded."""
from __future__ import annotations
from hermes_agent.plugins.web.tavily.provider import TavilyWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(TavilyWebSearchProvider())
