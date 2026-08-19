"""Local web extract plugin — bundled, auto-loaded.

Fetches pages directly from this machine with ``httpx`` (core dependency)
and reduces them to readable text with ``trafilatura`` (Apache-2.0,
lazily installed via ``tools.lazy_deps``). No API key required.
Extract-only — pair with any search provider (ddgs, searxng, …).
"""

from __future__ import annotations

from plugins.web.local.provider import LocalWebSearchProvider


def register(ctx) -> None:
    """Register the local extract provider with the plugin context."""
    ctx.register_web_search_provider(LocalWebSearchProvider())
