"""DuckDuckGo Instant Answer API plugin (no key, no deps) — bundled, auto-loaded.

Wraps https://api.duckduckgo.com/ — the free, no-auth, no-rate-limit
Instant Answer endpoint that returns entity summaries (``Abstract``) plus
``RelatedTopics``. Distinct from ``ddgs`` (which scrapes HTML and needs the
``ddgs`` package); this one uses only the stdlib so it works out of the box.
"""
from __future__ import annotations
from plugins.web.ddg_instant.provider import DDGInstantWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(DDGInstantWebSearchProvider())