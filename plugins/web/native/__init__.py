"""Local HTTP fetch extract plugin — bundled, auto-loaded.

Uses httpx + readability-lxml + html2text to fetch and extract web page
content directly. No API key required. Register as ``extract_backend: native``.
"""

from __future__ import annotations

from plugins.web.native.provider import WebFetchWebSearchProvider


def register(ctx) -> None:
    """Register the web-fetch provider with the plugin context."""
    ctx.register_web_search_provider(WebFetchWebSearchProvider())