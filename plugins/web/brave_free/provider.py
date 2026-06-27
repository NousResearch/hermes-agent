"""Brave Search (free tier) — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider` (the
plugin-facing ABC). The legacy in-tree module
``tools.web_providers.brave_free`` was removed in the same commit that
moved this code under ``plugins/``; this file is now the canonical
implementation.

Config keys this provider responds to::

    web:
      search_backend: "brave-free"     # explicit per-capability
      backend: "brave-free"            # shared fallback
      brave_free:
        min_request_interval_seconds: 1.5

Auth env var::

    BRAVE_SEARCH_API_KEY=...    # https://brave.com/search/api/ (free tier)
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 1.5
_THROTTLE_LOCK = threading.Lock()
_last_request_monotonic: float | None = None


def _load_brave_free_config() -> Dict[str, Any]:
    """Read ``web.brave_free`` from config.yaml without deepcopying it."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        web_section = cfg.get("web") if isinstance(cfg, dict) else None
        brave_section = web_section.get("brave_free") if isinstance(web_section, dict) else None
        return brave_section if isinstance(brave_section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load web.brave_free config: %s", exc)
        return {}


def _min_request_interval_seconds() -> float:
    value = _load_brave_free_config().get(
        "min_request_interval_seconds",
        _DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
    )
    try:
        interval = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_REQUEST_INTERVAL_SECONDS
    if not math.isfinite(interval):
        return _DEFAULT_MIN_REQUEST_INTERVAL_SECONDS
    return max(0.0, interval)


def _throttle_brave_request() -> None:
    """Serialize Brave API requests and enforce the configured process-local gap."""
    global _last_request_monotonic

    interval = _min_request_interval_seconds()
    if interval <= 0:
        return

    with _THROTTLE_LOCK:
        now = time.monotonic()
        if _last_request_monotonic is not None:
            wait_seconds = interval - (now - _last_request_monotonic)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.monotonic()
        _last_request_monotonic = now


def _reset_throttle_for_tests() -> None:
    global _last_request_monotonic
    with _THROTTLE_LOCK:
        _last_request_monotonic = None


class BraveFreeWebSearchProvider(WebSearchProvider):
    """Search-only Brave provider using the free-tier Data-for-Search API.

    Free tier is 2,000 queries/month (1 qps). No content-extraction capability —
    users pair this with Firecrawl/Tavily/Exa for ``web_extract``.
    """

    @property
    def name(self) -> str:
        # Hyphen form preserved for backward compat with the existing
        # ``web.search_backend: "brave-free"`` config keys users have set.
        return "brave-free"

    @property
    def display_name(self) -> str:
        return "Brave Search (Free)"

    def is_available(self) -> bool:
        """Return True when ``BRAVE_SEARCH_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("BRAVE_SEARCH_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the Brave Search API.

        Returns ``{"success": True, "data": {"web": [{"title", "url", "description", "position"}]}}``
        on success, or ``{"success": False, "error": str}`` on failure.
        """
        import httpx

        api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "BRAVE_SEARCH_API_KEY is not set"}

        # Brave's `count` is capped at 20.
        count = max(1, min(int(limit), 20))

        try:
            _throttle_brave_request()
            resp = httpx.get(
                _BRAVE_ENDPOINT,
                params={"q": query, "count": count},
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Brave Search HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"Brave Search returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("Brave Search request error: %s", exc)
            return {"success": False, "error": f"Could not reach Brave Search: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Brave Search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Brave Search response as JSON"}

        raw_results = (data.get("web") or {}).get("results", []) or []
        truncated = raw_results[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("description", "")),
                "position": i + 1,
            }
            for i, r in enumerate(truncated)
        ]

        logger.info(
            "Brave Search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Brave Search (Free)",
            "badge": "free",
            "tag": "Free-tier API key — 2k queries/mo, search only.",
            "env_vars": [
                {
                    "key": "BRAVE_SEARCH_API_KEY",
                    "prompt": "Brave Search API key (free tier)",
                    "url": "https://brave.com/search/api/",
                },
            ],
        }
