"""SerpAPI — multi-engine (Google, Bing, Baidu, YouTube). 100 queries/month free. Requires SERPAPI_API_KEY from https://serpapi.com."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class SerpapiWebSearchProvider(WebSearchProvider):
    """SerpAPI — free tier search provider."""

    @property
    def name(self) -> str:
        return "serpapi"

    @property
    def display_name(self) -> str:
        return "SerpAPI"

    def is_available(self) -> bool:
        """Return True when SERPAPI_API_KEY is set."""
        return bool(os.getenv("SERPAPI_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import requests
        api_key = os.getenv("SERPAPI_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "SERPAPI_API_KEY not set. Get a key at https://serpapi.com."}
        try:
            r = requests.get("https://serpapi.com/search", params={"q": query, "num": min(limit, 100), "engine": "google", "api_key": api_key}, timeout=10)
            r.raise_for_status()
            data = r.json()
            raw = data.get("organic_results", [])[:limit]
            results = [
                {"title": str(i.get("title", "")), "url": str(i.get("link", "")),
                 "description": str(i.get("snippet", "")), "position": n + 1}
                for n, i in enumerate(raw)
            ]
            return {"success": True, "data": {"web": results}}
        except Exception as e:
            return {"success": False, "error": str(e)}
    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SerpAPI",
            "badge": "free",
            "tag": "100 queries/month free tier, multi-engine (Google/Bing/Baidu/YouTube).",
            "env_vars": [
                {
                    "key": "SERPAPI_API_KEY",
                    "prompt": "SerpAPI API key",
                    "url": "https://serpapi.com",
                },
            ],
        }
