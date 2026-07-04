"""Local browser web extraction plugin — bundled, auto-loaded.

Provides a local-first ``web_extract`` backend backed by the agent-browser CLI,
with a deterministic urllib/static-HTML fallback. No paid scraping API calls.
"""

from __future__ import annotations

from plugins.web.local_browser.provider import LocalBrowserExtractProvider


def register(ctx) -> None:
    """Register the local browser extraction provider with the plugin context."""
    ctx.register_web_search_provider(LocalBrowserExtractProvider())
