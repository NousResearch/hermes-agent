"""CloakBrowser web search + extract plugin — bundled, auto-loaded.

Stealth Chromium backend for ``web_search`` and ``web_extract``. Uses the
``cloakbrowser`` Python package (Playwright-compatible) to bypass bot
detection on search and crawl targets.

Source: https://github.com/zapabob/CloakBrowser
"""

from __future__ import annotations

from plugins.web.cloakbrowser.provider import CloakBrowserWebSearchProvider


def register(ctx) -> None:
    """Register the CloakBrowser provider with the plugin context."""
    ctx.register_web_search_provider(CloakBrowserWebSearchProvider())
