"""DuckDuckGo search plugin — bundled, auto-loaded.

Backed by the community ``ddgs`` Python package which scrapes DDG's HTML
results page. No API key required, but the package itself must be installed
(it's an optional dep — gated via :meth:`is_available`).
"""

from __future__ import annotations

from plugins.web.ddgs.provider import DDGSWebSearchProvider


def register(ctx) -> None:
    """Register the DDGS provider with the plugin context."""
    ctx.register_web_search_provider(DDGSWebSearchProvider())
