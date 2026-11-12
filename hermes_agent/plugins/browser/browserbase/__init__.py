"""Browserbase cloud browser plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.browser.browserbase.provider import BrowserbaseBrowserProvider


def register(ctx) -> None:
    ctx.register_browser_provider(BrowserbaseBrowserProvider())
