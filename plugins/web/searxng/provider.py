"""SearXNG search via a user-hosted instance (``/search?format=json``).

Search-only — SearXNG aggregates upstream engines but does not fetch URLs.
Env: ``SEARXNG_URL=http://localhost:8080``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)

# SearXNG's limiter answers 429 for its bot-detection refusals as well as its rate limits, and caps
# ``format=json`` at a few requests per hour per IP whatever the headers, so only the instance's
# own configuration can let an API client through. Without this, the bare 429 reads as a quota.
_STATUS_HINTS = {
    429: (
        "likely the instance's limiter (bot detection), which refuses or caps API clients. On an instance you run, "
        "add the IP SearXNG sees this host as to botdetection.ip_lists.pass_ip in limiter.toml, or set server.limiter: false"
    ),
}


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
            headers={"Accept": "application/json"}, timeout=15, logger=logger, reach_target=f"SearXNG at {base_url}",
            status_hints=_STATUS_HINTS,
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
