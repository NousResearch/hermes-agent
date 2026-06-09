"""Serper.dev Google Search API — plugin form.

Search-only provider backed by Serper's Google search endpoint.

Config keys this provider responds to::

    web:
      search_backend: "serper"        # explicit per-capability
      backend: "serper"               # shared fallback

Auth env vars::

    SERPER_API_KEY=...                 # required (https://serper.dev)
    SERPER_BASE_URL=https://google.serper.dev  # optional override
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://google.serper.dev"


class SerperWebSearchProvider(WebSearchProvider):
    """Search-only provider using Serper.dev's Google Search API."""

    @property
    def name(self) -> str:
        return "serper"

    @property
    def display_name(self) -> str:
        return "Serper"

    def is_available(self) -> bool:
        return bool(os.getenv("SERPER_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "SERPER_API_KEY is not set"}

        try:
            safe_limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            safe_limit = 5

        base_url = os.getenv("SERPER_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/") or _DEFAULT_BASE_URL

        try:
            resp = httpx.post(
                f"{base_url}/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": safe_limit},
                timeout=15,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Serper HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"Serper returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("Serper request error: %s", exc)
            return {"success": False, "error": f"Could not reach Serper: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Serper response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Serper response as JSON"}

        raw_results = data.get("organic", []) or []
        truncated = raw_results[:safe_limit]

        web_results = []
        for i, row in enumerate(truncated):
            web_results.append(
                {
                    "title": str(row.get("title", "")),
                    "url": str(row.get("link") or row.get("url") or ""),
                    "description": str(
                        row.get("snippet")
                        or row.get("description")
                        or row.get("text")
                        or ""
                    ),
                    "position": i + 1,
                }
            )

        logger.info(
            "Serper search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            safe_limit,
        )
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Serper",
            "badge": "paid",
            "tag": "Google SERP API (search only) via serper.dev.",
            "env_vars": [
                {
                    "key": "SERPER_API_KEY",
                    "prompt": "Serper API key",
                    "url": "https://serper.dev",
                },
            ],
        }
