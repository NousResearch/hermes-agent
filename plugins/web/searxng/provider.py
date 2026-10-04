"""SearXNG search via a user-hosted instance (``/search?format=json``).

Search-only — SearXNG aggregates upstream engines but does not fetch URLs.
Env: ``SEARXNG_URL=http://localhost:8080``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)

# Minimum seconds between consecutive SearXNG searches (SEARXNG_SEARCH_DELAY env).
# Self-hosted instances get suspended by upstream engines (Google, Brave, DuckDuckGo)
# when queried in rapid bursts; pacing requests avoids empty result sets.
# Unset / non-positive → no pacing (stock behavior).
_LAST_SEARCH_TS: float = 0.0


def _min_search_interval() -> float:
    """Pacing interval from ``SEARXNG_SEARCH_DELAY`` (seconds); 0.0 disables pacing."""
    raw = provider_env("SEARXNG_SEARCH_DELAY")
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Ignoring invalid SEARXNG_SEARCH_DELAY=%r", raw)
        return 0.0


class SearXNGWebSearchProvider(BaseWebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    NAME = "searxng"
    DISPLAY_NAME = "SearXNG"
    KEY_ENV = "SEARXNG_URL"

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        global _LAST_SEARCH_TS
        base_url = provider_env("SEARXNG_URL").rstrip("/")
        if not base_url:
            return search_fail("SEARXNG_URL is not set")

        interval = _min_search_interval()
        if interval > 0:
            elapsed = time.monotonic() - _LAST_SEARCH_TS
            if elapsed < interval:
                time.sleep(interval - elapsed)

        data, failure = http_get_json(
            "SearXNG", f"{base_url}/search", params={"q": query, "format": "json", "pageno": 1},
            headers={"Accept": "application/json"}, timeout=15, logger=logger, reach_target=f"SearXNG at {base_url}",
        )
        _LAST_SEARCH_TS = time.monotonic()
        if failure is not None:
            return failure
        raw_results = data.get("results", [])
        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(raw_results, key=lambda r: float(r.get("score", 0)), reverse=True)[:limit]
        web_results = titled_rows(sorted_results, "content")
        logger.info("SearXNG search '%s': %d results (from %d raw, limit %d)", query, len(web_results), len(raw_results), limit)
        return search_ok(web_results)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "SearXNG", "free · self-hosted", "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "SEARXNG_URL", "SearXNG instance URL (e.g. http://localhost:8080)", "https://searx.space/",
        )
