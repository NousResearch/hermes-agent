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

Auth env var::

    BRAVE_SEARCH_API_KEY=...    # https://brave.com/search/api/ (free tier)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
_BRAVE_IMAGES_ENDPOINT = "https://api.search.brave.com/res/v1/images/search"


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

    def search(self, query: str, limit: int = 5, categories: Optional[list[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute a search against the Brave Search API.

        When ``categories`` is provided, the first recognized value is
        passed as Brave's ``result_filter`` parameter (``web``, ``news``,
        ``images``, ``video``). Unrecognized values are silently ignored.

        Returns ``{"success": True, "data": {"web": [{"title", "url", "description", "position"}]}}``
        on success, or ``{"success": False, "error": str}`` on failure.
        """
        import httpx

        api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "BRAVE_SEARCH_API_KEY is not set"}

        # Brave's `count` is capped at 20.
        count = max(1, min(int(limit), 20))

        brave_categories = {"web", "news", "images", "video"}
        endpoint = _BRAVE_ENDPOINT
        params: Dict[str, Any] = {"q": query, "count": count}
        selected_category = None
        if categories:
            for c in categories:
                if c in brave_categories:
                    selected_category = c
                    if c == "images":
                        endpoint = _BRAVE_IMAGES_ENDPOINT
                    elif c == "news":
                        endpoint = _BRAVE_NEWS_ENDPOINT
                    elif c == "video":
                        params["result_filter"] = "videos"
                    # "web" uses default web endpoint with no filter
                    break

        try:
            resp = httpx.get(
                endpoint,
                params=params,
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

        # Read results from the appropriate container based on category
        if selected_category in ("images", "news"):
            # Brave's dedicated news/images endpoints return results at top level
            raw_results = data.get("results", []) or []
        else:
            raw_results = (data.get("web") or {}).get("results", []) or []

        # Fallback: try all containers if nothing found in the primary
        if not raw_results:
            for container in ("web", "news", "video"):
                raw_results = (data.get(container) or {}).get("results", []) or []
                if raw_results:
                    break

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
