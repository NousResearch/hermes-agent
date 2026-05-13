"""Firecrawl web search + extract plugin — bundled, auto-loaded.

Largest single plugin in this PR. Captures everything the previous
inline implementation in tools/web_tools.py did:

  - Lazy import of the firecrawl SDK (~200ms cold-start cost) via a
    callable proxy that defers the actual import to first use.
  - Dual client paths: direct (FIRECRAWL_API_KEY / FIRECRAWL_API_URL)
    OR Nous-hosted tool-gateway routing for subscribers, with
    web.use_gateway as the tie-breaker.
  - Per-URL scrape loop with 60s timeout, SSRF re-check after redirect,
    website-policy gating, and format-aware content selection.
  - Robust response shape normalization across SDK / direct API /
    gateway variants (search returns differ by transport).

The plugin re-exports ``Firecrawl`` (the lazy proxy) and
``check_firecrawl_api_key`` for backward-compatibility with tests and
external code that imports those names from ``tools.web_tools``.
"""

from __future__ import annotations

from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider


def register(ctx) -> None:
    """Register the Firecrawl provider with the plugin context."""
    ctx.register_web_search_provider(FirecrawlWebSearchProvider())
