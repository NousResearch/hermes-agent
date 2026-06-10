"""Obscura local browser plugin, bundled, auto-loaded.

Mirrors the other ``plugins/browser/<vendor>/`` plugins: ``provider.py`` holds
the provider class; ``__init__.py::register`` instantiates and registers it via
the plugin context.
"""

from __future__ import annotations

from plugins.browser.obscura.provider import ObscuraBrowserProvider


def register(ctx) -> None:
    """Register the Obscura provider with the plugin context."""
    ctx.register_browser_provider(ObscuraBrowserProvider())
