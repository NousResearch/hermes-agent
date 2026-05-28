"""Oxylabs AI Studio web provider plugin — bundled, auto-loaded.

Talks to the AI Studio REST API directly over ``httpx`` (a Hermes core
dependency) — no vendor SDK. Exposes two ``WebSearchProvider``
capabilities: search (``/search`` endpoints) and extract (``/scrape``).

Crawl, Browser Agent, and AI-Map are not surfaced through this provider —
the ABC has no slot for them. They're candidates for a future sibling
tool plugin under ``plugins/oxylabs/``.
"""

from __future__ import annotations

from plugins.web.oxylabs.provider import OxylabsWebSearchProvider


def register(ctx) -> None:
    """Register the Oxylabs provider with the plugin context."""
    ctx.register_web_search_provider(OxylabsWebSearchProvider())
