"""AnySearch web provider.

Unified real-time search engine supporting general web, vertical domain search,
and batch search. Integrates with Hermes via the `web_search` tool.

Config keys::

    web:
      search_backend: "anysearch"
      backend: "searxng"  # fallback

Env var::

    ANYSEARCH_API_KEY=as_sk_...
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class AnySearchProvider(WebSearchProvider):
    """Search via AnySearch REST API (/v1/search)."""

    @property
    def name(self) -> str:
        return "anysearch"

    @property
    def display_name(self) -> str:
        return "AnySearch"

    def is_available(self) -> bool:
        return bool(os.getenv("ANYSEARCH_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False  # Extract via MCP is unreliable (403s)

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        api_key = os.getenv("ANYSEARCH_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "ANYSEARCH_API_KEY is not set"}

        url = "https://api.anysearch.com/v1/search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "max_results": limit,
            # Auto-detect CN zone for Chinese queries
            "zone": "cn" if any("\u4e00" <= c <= "\u9fff" for c in query) else None,
        }
        # Remove zone key if None to avoid sending it
        if payload["zone"] is None:
            del payload["zone"]

        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("AnySearch HTTP error: %s", exc)
            try:
                detail = exc.response.json()
                msg = detail.get("message", f"HTTP {exc.response.status_code}")
            except Exception:
                msg = f"HTTP {exc.response.status_code}"
            return {"success": False, "error": f"AnySearch returned {msg}"}
        except httpx.RequestError as exc:
            return {"success": False, "error": f"Could not reach AnySearch API: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:
            return {"success": False, "error": f"Could not parse AnySearch response: {exc}"}

        if data.get("code") != 0:
            return {"success": False, "error": f"API error: {data.get('message', 'unknown')}"}

        results = data.get("data", {}).get("results", [])
        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", ""))[:500],  # Snippet
                "position": i + 1,
            }
            for i, r in enumerate(results[:limit])
        ]

        logger.info(
            "AnySearch '%s': %d results (limit %d)",
            query, len(web_results), limit,
        )
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "AnySearch",
            "badge": "real-time · 23 domains",
            "tag": "Unified search with vertical domains (finance, academic, security, etc.).",
            "env_vars": [
                {
                    "key": "ANYSEARCH_API_KEY",
                    "prompt": "AnySearch API key",
                    "url": "https://anysearch.com/console/api-keys",
                },
            ],
        }
