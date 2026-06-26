"""Baidu AI Search API — 100 queries/day free. Requires BAIDU_API_KEY from https://ai.baidu.com."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class BaiduWebSearchProvider(WebSearchProvider):
    """Baidu Search — free tier search provider."""

    @property
    def name(self) -> str:
        return "baidu"

    @property
    def display_name(self) -> str:
        return "Baidu Search"

    def is_available(self) -> bool:
        """Return True when BAIDU_API_KEY is set."""
        return bool(os.getenv("BAIDU_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx
        api_key = os.getenv("BAIDU_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "BAIDU_API_KEY not set. Get a key at https://ai.baidu.com."}
        try:
            r = httpx.post("https://api.baidu.com/search/v1/websearch", params={"q": query, "topn": min(limit, 50)}, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            r.raise_for_status()
            data = r.json()
            raw = data.get("results", [])[:limit]
            results = [
                {"title": str(i.get("title", "")), "url": str(i.get("url", "")),
                 "description": str(i.get("summary", "")), "position": n + 1}
                for n, i in enumerate(raw)
            ]
            return {"success": True, "data": {"web": results}}
        except Exception as e:
            return {"success": False, "error": str(e)}
    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Baidu Search",
            "badge": "free",
            "tag": "100 queries/day free tier, native Chinese content.",
            "env_vars": [
                {
                    "key": "BAIDU_API_KEY",
                    "prompt": "Baidu Search API key",
                    "url": "https://ai.baidu.com",
                },
            ],
        }
