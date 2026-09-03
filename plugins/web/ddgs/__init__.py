"""DuckDuckGo search plugin — bundled, auto-loaded.

Backed by the community ``ddgs`` package on desktop platforms and a core-httpx
HTML fallback on Termux, where ddgs's native transport cannot run. No API key
is required.
"""

from __future__ import annotations

from plugins.web.ddgs.provider import DDGSWebSearchProvider


def register(ctx) -> None:
    """Register the DDGS provider with the plugin context."""
    ctx.register_web_search_provider(DDGSWebSearchProvider())
