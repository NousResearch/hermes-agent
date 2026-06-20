"""You.com Search API — plugin form.

Search-only provider backed by You.com's Search API.

Config keys this provider responds to::

    web:
      search_backend: "you"          # explicit per-capability
      backend: "you"                 # shared fallback

Auth env vars::

    YOU_API_KEY=...   # canonical name
    YDC_API_KEY=...   # legacy alias supported for backward compatibility
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_YOU_ENDPOINT = "https://ydc-index.io/v1/search"


def _you_api_key() -> str:
    """Return the configured You.com API key, preferring the modern name."""
    return (
        os.getenv("YOU_API_KEY", "").strip()
        or os.getenv("YDC_API_KEY", "").strip()
    )


class YouWebSearchProvider(WebSearchProvider):
    """Search-only You.com provider using the Search API."""

    @property
    def name(self) -> str:
        return "you"

    @property
    def display_name(self) -> str:
        return "You.com Search"

    def is_available(self) -> bool:
        return bool(_you_api_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = _you_api_key()
        if not api_key:
            return {
                "success": False,
                "error": "YOU_API_KEY or YDC_API_KEY is not set",
            }

        count = max(1, min(int(limit), 100))

        try:
            resp = httpx.get(
                _YOU_ENDPOINT,
                params={"query": query, "count": count},
                headers={
                    "X-API-Key": api_key,
                    "Accept": "application/json",
                },
                timeout=20,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("You.com Search HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"You.com Search returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("You.com Search request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach You.com Search: {exc}",
            }

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("You.com Search response parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse You.com Search response as JSON",
            }

        raw_results = ((data.get("results") or {}).get("web") or [])
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
            "You.com Search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "You.com Search",
            "badge": "paid · search only",
            "tag": "You.com / YDC Search API — search only. Supports YOU_API_KEY or legacy YDC_API_KEY.",
            "env_vars": [
                {
                    "key": "YOU_API_KEY",
                    "prompt": "You.com Search API key",
                    "url": "https://you.com/platform",
                },
            ],
        }
