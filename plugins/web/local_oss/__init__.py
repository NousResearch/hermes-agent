"""Local OSS web backend plugin.

Registers:

- ``local_oss``: SearXNG search + Crawl4AI extract.
- ``crawl4ai``: Crawl4AI extract-only backend for per-capability config.
"""

from __future__ import annotations

from plugins.web.local_oss.provider import (
    Crawl4AIWebSearchProvider,
    LocalOSSWebSearchProvider,
)


def register(ctx) -> None:
    """Register the local OSS web providers with the plugin context."""
    ctx.register_web_search_provider(LocalOSSWebSearchProvider())
    ctx.register_web_search_provider(Crawl4AIWebSearchProvider())
