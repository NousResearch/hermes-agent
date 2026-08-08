"""Remote Browser cloud browser plugin."""

from __future__ import annotations

from plugins.browser.remote_browser.provider import RemoteBrowserProvider


def register(ctx) -> None:
    """Register the Remote Browser provider with the plugin context."""
    ctx.register_browser_provider(RemoteBrowserProvider())
