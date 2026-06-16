"""Steel cloud browser plugin — bundled, auto-loaded.

Mirrors the ``plugins/browser/browserbase/`` and ``plugins/browser/firecrawl/``
layout: ``provider.py`` holds the provider class; ``__init__.py::register``
instantiates and registers it via the plugin context.
"""

from __future__ import annotations

from plugins.browser.steel.provider import SteelBrowserProvider


def register(ctx) -> None:
    """Register the Steel cloud-browser provider with the plugin context."""
    ctx.register_browser_provider(SteelBrowserProvider())
