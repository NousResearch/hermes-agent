"""Bocha Web Search provider plugin."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _bocha_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to Bocha Web Search API and return parsed JSON."""
    import httpx

    api_key = os.getenv("BOCHA_API_KEY")
    if not api_key:
        raise ValueError(
            "BOCHA_API_KEY environment variable not set. "
            "Get your API key at https://open.bochaai.com/"
        )

    base_url = os.getenv("BOCHA_BASE_URL", "https://api.bochaai.com")
    url = f"{base_url.rstrip('/')}/v1/web-search"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info("Bocha web search request to %s", url)

    response = httpx.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def _normalize_bocha_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map Bocha response to Hermes web search result shape."""
    web_results = []

    # Bocha commonly returns results under data.webPages.value. Some wrappers
    # flatten this to webPages.value; support both for compatibility.
    payload = response.get("data") if isinstance(response.get("data"), dict) else response
    web_pages = payload.get("webPages") or {}
    results = web_pages.get("value") or []

    for i, result in enumerate(results):
        snippet = result.get("summary") or result.get("snippet") or ""
        web_results.append(
            {
                "title": result.get("name", ""),
                "url": result.get("url", ""),
                "description": snippet,
                "position": i + 1,
            }
        )

    return {"success": True, "data": {"web": web_results}}


class BochaWebSearchProvider(WebSearchProvider):
    """Bocha search-only provider."""

    @property
    def name(self) -> str:
        return "bocha"

    @property
    def display_name(self) -> str:
        return "Bocha"

    def is_available(self) -> bool:
        return bool(os.getenv("BOCHA_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def supports_crawl(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Bocha web search."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Bocha search: '%s' (limit=%d)", query, limit)

            raw = _bocha_request(
                {
                    "query": query,
                    "summary": True,
                    "count": max(1, min(limit, 50)),
                }
            )
            return _normalize_bocha_search_results(raw)

        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bocha search error: %s", exc)
            return {"success": False, "error": f"Bocha search failed: {exc}"}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Bocha",
            "badge": "paid",
            "tag": "Web search provider for AI applications.",
            "env_vars": [
                {
                    "key": "BOCHA_API_KEY",
                    "prompt": "Bocha API key",
                    "url": "https://open.bochaai.com/",
                },
            ],
        }
