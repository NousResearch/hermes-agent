"""SearXNG search via a user-hosted instance (``/search?format=json``).

Search-only — SearXNG aggregates upstream engines but does not fetch URLs.
Env: ``SEARXNG_URL=http://localhost:8080``; optional ``SEARXNG_TIMEOUT``
(seconds, default 15) overrides the request deadline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)

def _searxng_timeout() -> float:
    """Request deadline in seconds — ``SEARXNG_TIMEOUT`` env override.

    Useful for private SearXNG instances behind slow upstream paths
    (e.g. Tor exits, where an engine round can take 5-25s). Defaults
    to 15s when unset or unparseable; floored at 5s so a nonsensical
    low value can't wedge every query.
    """
    raw = provider_env("SEARXNG_TIMEOUT")
    try:
        return max(5.0, float(raw)) if raw else 15.0
    except (TypeError, ValueError):
        return 15.0


class SearXNGWebSearchProvider(BaseWebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    NAME = "searxng"
    DISPLAY_NAME = "SearXNG"
    KEY_ENV = "SEARXNG_URL"

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        base_url = provider_env("SEARXNG_URL").rstrip("/")
        if not base_url:
            return search_fail("SEARXNG_URL is not set")
        data, failure = http_get_json(
            "SearXNG", f"{base_url}/search", params={"q": query, "format": "json", "pageno": 1},
            headers={"Accept": "application/json"}, timeout=_searxng_timeout(), logger=logger, reach_target=f"SearXNG at {base_url}",
        )
        if failure is not None:
            return failure
        raw_results = data.get("results", [])
        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(raw_results, key=lambda r: float(r.get("score", 0)), reverse=True)[:limit]
        web_results = titled_rows(sorted_results, "content")
        logger.info("SearXNG search '%s': %d results (from %d raw, limit %d)", query, len(web_results), len(raw_results), limit)
        return search_ok(web_results)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "SearXNG", "free · self-hosted", "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "SEARXNG_URL", "SearXNG instance URL (e.g. http://localhost:8080)", "https://searx.space/",
        )
