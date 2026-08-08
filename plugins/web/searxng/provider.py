"""SearXNG search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Same JSON
API call (``/search?format=json``), same result normalization. The legacy
in-tree module ``tools.web_providers.searxng`` was removed in the same
commit that moved this code under ``plugins/``; this file is now the
canonical implementation.

Search-only — SearXNG aggregates results from upstream engines but does not
fetch/extract arbitrary URLs. ``supports_extract()`` returns False.

Config keys this provider responds to::

    web:
      search_backend: "searxng"     # explicit per-capability
      backend: "searxng"            # shared fallback
      searxng:
        url: "https://search.example.com"   # optional, overrides SEARXNG_URL
        method: "get"                       # HTTP method — "get" (default) or "post".
                                            #   POST is recommended when your SearXNG
                                            #   instance has ``server.method = "post"``.
        params:                             # additional query params sent with every search
          categories: "general"
          language: "en-US"
          safesearch: 0
        headers:                            # additional HTTP headers (merged with
          Accept-Language: "en-US,en;q=0.9" #   Accept: application/json)

Env var::

    SEARXNG_URL=http://localhost:8080
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

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


class SearXNGWebSearchProvider(WebSearchProvider):
    """Search via a user-hosted SearXNG instance, configured through ``web.searxng`` in config.yaml."""

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def display_name(self) -> str:
        return "SearXNG"

    def is_available(self) -> bool:
        """Return True when ``SEARXNG_URL`` (env or config) is set."""
        return bool(_searxng_url())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the configured SearXNG instance.

        Reads method, params, and headers from ``web.searxng`` config.
        Defaults to GET for backward compatibility.
        """
        import httpx

        base_url = _searxng_url().rstrip("/")
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        searxng_cfg = _searxng_config()
        method = (searxng_cfg.get("method") or _DEFAULT_METHOD).strip().lower()
        extra_params = searxng_cfg.get("params", {}) or {}
        extra_headers = searxng_cfg.get("headers", {}) or {}

        # user values cannot override q/format/pageno
        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        for k, v in extra_params.items():
            if k not in params:
                params[k] = v

        headers = {"Accept": "application/json"}
        headers.update(extra_headers)

        try:
            if method == "post":
                resp = httpx.post(
                    f"{base_url}/search",
                    data=params,
                    headers=headers,
                    timeout=15,
                )
            else:
                resp = httpx.get(
                    f"{base_url}/search",
                    params=params,
                    headers=headers,
                    timeout=15,
                )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"SearXNG returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach SearXNG at {base_url}: {exc}",
            }

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SearXNG response parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse SearXNG response as JSON",
            }

        raw_results = data.get("results", [])

        sorted_results = sorted(
            raw_results,
            key=lambda r: float(r.get("score", 0)),
            reverse=True,
        )[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
            }
            for i, r in enumerate(sorted_results)
        ]

        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d) [method=%s]",
            query,
            len(web_results),
            len(raw_results),
            limit,
            method,
        )

        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SearXNG",
            "badge": "free · self-hosted",
            "tag": "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance, or configure web.searxng in config.yaml.",
            "env_vars": [
                {
                    "key": "SEARXNG_URL",
                    "prompt": "SearXNG instance URL (e.g. http://localhost:8080)",
                    "url": "https://searx.space/",
                },
            ],
        }
