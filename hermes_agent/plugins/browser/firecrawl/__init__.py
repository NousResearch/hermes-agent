"""Firecrawl cloud browser plugin — bundled, auto-loaded (distinct from hermes_agent/plugins/web/firecrawl/)."""

from __future__ import annotations

from hermes_agent.plugins.browser.firecrawl.provider import FirecrawlBrowserProvider


def register(ctx) -> None:
    ctx.register_browser_provider(FirecrawlBrowserProvider())
