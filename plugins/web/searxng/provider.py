"""SearXNG search via a user-hosted instance (``/search?format=json``).

Search-only — SearXNG aggregates upstream engines but does not fetch/extract
arbitrary URLs. ``supports_extract()`` returns False.

Config keys this provider responds to::

    web:
      search_backend: "searxng"     # explicit per-capability
      backend: "searxng"            # shared fallback
      searxng:
        url: "https://search.example.com"   # optional, overrides SEARXNG_URL
        method: "get"                       # HTTP method: "get" (default) or "post".
                                            #   POST is recommended when your SearXNG
                                            #   instance has ``server.method = "post"``.
        params:                             # extra query params sent with every search
          categories: "general"
          language: "en-US"
          safesearch: 0
        headers:                            # extra headers, merged with Accept
          Accept-Language: "en-US,en;q=0.9"

Env var::

    SEARXNG_URL=http://localhost:8080
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from plugins.web._common import (
    BaseWebSearchProvider,
    http_get_json,
    search_fail,
    search_ok,
    setup_schema,
    titled_rows,
)

logger = logging.getLogger(__name__)

_DEFAULT_METHOD = "get"


def _searxng_config() -> Dict[str, Any]:
    """Read ``web.searxng`` section from config.yaml. Returns empty dict on miss."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg.get("web", {}).get("searxng", {}) or {}
    except Exception:
        return {}


def _searxng_url() -> str:
    """Resolve the SearXNG URL — config.yaml > .env > process env."""
    try:
        searxng_cfg = _searxng_config()
        url = searxng_cfg.get("url", "").strip()
        if url:
            return url
    except Exception:
        pass

    try:
        from hermes_cli.config import get_env_value

        val = get_env_value("SEARXNG_URL")
    except Exception:
        val = None

    if val is None:
        val = os.getenv("SEARXNG_URL", "")

    return (val or "").strip()


class SearXNGWebSearchProvider(BaseWebSearchProvider):
    """Search via a user-hosted SearXNG instance, configured through ``web.searxng`` in config.yaml."""

    NAME = "searxng"
    DISPLAY_NAME = "SearXNG"
    KEY_ENV = "SEARXNG_URL"

    def is_available(self) -> bool:
        """True when a URL is set, whether it came from env or ``web.searxng.url``."""
        return bool(_searxng_url())

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Execute a search against the configured SearXNG instance.

        Method, extra params and extra headers come from ``web.searxng``.
        GET stays the default so existing instances behave unchanged.
        """
        base_url = _searxng_url().rstrip("/")
        if not base_url:
            return search_fail("SEARXNG_URL is not set")

        cfg = _searxng_config()
        method = (cfg.get("method") or _DEFAULT_METHOD).strip().lower()
        if method not in ("get", "post"):
            method = _DEFAULT_METHOD
        extra_params = cfg.get("params") or {}
        extra_headers = cfg.get("headers") or {}

        # User values may not override the keys the protocol depends on.
        params: dict[str, Any] = {"q": query, "format": "json", "pageno": 1}
        for key, value in extra_params.items():
            params.setdefault(key, value)

        headers = {"Accept": "application/json"}
        headers.update({str(key): str(value) for key, value in extra_headers.items()})

        data, failure = http_get_json(
            "SearXNG",
            f"{base_url}/search",
            params=params,
            headers=headers,
            timeout=15,
            logger=logger,
            reach_target=f"SearXNG at {base_url}",
            method=method.upper(),
        )
        if failure is not None:
            return failure

        raw_results = data.get("results", [])
        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(raw_results, key=lambda r: float(r.get("score", 0)), reverse=True)[:limit]
        web_results = titled_rows(sorted_results, "content")
        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d) [method=%s]",
            query,
            len(web_results),
            len(raw_results),
            limit,
            method,
        )
        return search_ok(web_results)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "SearXNG", "free · self-hosted", "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance, or configure web.searxng in config.yaml.",
            "SEARXNG_URL", "SearXNG instance URL (e.g. http://localhost:8080)", "https://searx.space/",
        )
