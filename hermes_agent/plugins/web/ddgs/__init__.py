"""DuckDuckGo search plugin (``ddgs`` package, optional dep) — bundled, auto-loaded."""
from __future__ import annotations
from hermes_agent.plugins.web.ddgs.provider import DDGSWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(DDGSWebSearchProvider())
