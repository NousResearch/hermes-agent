"""Kagi web search plugin — bundled, auto-loaded.

Uses the existing ``kagi_search.py`` script which scrapes Kagi's
``/html/search`` endpoint via a session cookie.  Requires ``KAGI_SESSION``
in ``~/.hermes/.env``.
"""

from __future__ import annotations

from plugins.web.kagi.provider import KagiWebSearchProvider


def register(ctx) -> None:
    """Register the Kagi provider with the plugin context."""
    ctx.register_web_search_provider(KagiWebSearchProvider())
