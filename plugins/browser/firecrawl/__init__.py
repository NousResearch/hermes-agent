"""Firecrawl cloud browser plugin — bundled, auto-loaded (distinct from plugins/web/firecrawl/)."""

from __future__ import annotations

from plugins.browser.firecrawl.provider import FirecrawlBrowserProvider


def register(ctx) -> None:
    ctx.register_browser_provider(FirecrawlBrowserProvider())
