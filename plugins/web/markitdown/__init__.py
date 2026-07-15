"""MarkItDown extract plugin — bundled, auto-loaded.

Extract-only — uses Microsoft's MarkItDown library to convert HTML pages
to clean markdown locally. No API key, no cloud dependency. Pair with any
search provider (brave-free, ddgs, searxng, etc.) for web_search.
"""

from __future__ import annotations

from plugins.web.markitdown.provider import MarkItDownExtractProvider


def register(ctx) -> None:
    """Register the MarkItDown provider with the plugin context."""
    ctx.register_web_search_provider(MarkItDownExtractProvider())
