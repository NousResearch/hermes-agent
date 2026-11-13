"""Browser Use cloud browser plugin — bundled, auto-loaded."""

from __future__ import annotations

from hermes_agent.plugins.browser.browser_use.provider import BrowserUseBrowserProvider


def register(ctx) -> None:
    ctx.register_browser_provider(BrowserUseBrowserProvider())
