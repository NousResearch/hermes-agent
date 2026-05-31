"""Serper Google Search API provider."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_SERPER_ENDPOINT = "https://google.serper.dev/search"


class SerperWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by Serper's Google Search API."""

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

        num = max(1, min(int(limit), 20))
        try:
            resp = httpx.post(
                _SERPER_ENDPOINT,
                json={"q": query, "num": num},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Serper HTTP error: %s", exc)
            return {"success": False, "error": f"Serper returned HTTP {exc.response.status_code}"}
        except httpx.RequestError as exc:
            logger.warning("Serper request error: %s", exc)
            return {"success": False, "error": f"Could not reach Serper: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Serper response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Serper response as JSON"}

        organic = data.get("organic") or []
        web_results = []
        for i, r in enumerate(organic[:limit]):
            web_results.append({
                "title": str(r.get("title", "")),
                "url": str(r.get("link", "")),
                "description": str(r.get("snippet", "")),
                "position": int(r.get("position") or i + 1),
            })

        logger.info("Serper search '%s': %d results", query, len(web_results))
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Serper",
            "badge": "paid · Google SERP",
            "tag": "Google-quality SERP via Serper. Search only; use a reader/extractor for page content.",
            "env_vars": [
                {"key": "SERPER_API_KEY", "prompt": "Serper API key", "url": "https://serper.dev/"},
            ],
        }
