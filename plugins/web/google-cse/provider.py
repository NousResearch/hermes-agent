"""Google Custom Search JSON API — requires GCP project + API key + SE ID. NOTE: blocked by GFW in China."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class GoogleCseWebSearchProvider(WebSearchProvider):
    """Google CSE — free tier search provider."""

    @property
    def name(self) -> str:
        return "google-cse"

    @property
    def display_name(self) -> str:
        return "Google CSE"

    def is_available(self) -> bool:
        """Return True when GOOGLE_CSE_API_KEY is set."""
        return bool(os.getenv("GOOGLE_CSE_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import requests
        api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
        cx = os.getenv("GOOGLE_CSE_CX", "").strip()
        if not api_key or not cx:
            return {"success": False, "error": "GOOGLE_CSE_API_KEY or GOOGLE_CSE_CX not set."}
        try:
            r = requests.get("https://customsearch.googleapis.com/customsearch/v1",
                params={"q": query, "num": min(limit, 10), "key": api_key, "cx": cx}, timeout=10)
            r.raise_for_status()
            data = r.json()
            raw = data.get("items", [])[:limit]
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
            "name": "Google CSE",
            "badge": "free",
            "tag": "100 queries/day free tier. Requires GCP setup. NOT accessible from mainland China without proxy.",
            "env_vars": [
                {
                    "key": "GOOGLE_CSE_API_KEY",
                    "prompt": "Google CSE API key",
                    "url": "https://developers.google.com/custom-search/v1/overview",
                },
            ],
        }
