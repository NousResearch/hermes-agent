"""Multi-provider web search plugin — Exa + Tavily + web-search-prime.

Registers a single WebSearchProvider (name ``multi-search``) that fans out
to three search backends in parallel, merges and deduplicates results.
"""

from __future__ import annotations

from .provider import MultiSearchProvider


def register(ctx) -> None:
    """Register the multi-provider with the plugin context."""
    ctx.register_web_search_provider(MultiSearchProvider())
