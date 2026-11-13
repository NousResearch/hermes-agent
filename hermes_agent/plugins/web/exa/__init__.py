"""Exa web search + extract plugin — bundled, auto-loaded (sync SDK; dispatcher wraps async callers)."""
from __future__ import annotations
from hermes_agent.plugins.web.exa.provider import ExaWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(ExaWebSearchProvider())
