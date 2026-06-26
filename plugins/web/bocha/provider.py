"""Bochaa AI Search API — leading Chinese search for AI. Free for personal use. Requires BOCHA_API_KEY from https://open.bochaai.com."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class BochaWebSearchProvider(WebSearchProvider):
    """Bocha (博查) — free tier search provider."""

    @property
    def name(self) -> str:
        return "bocha"

    @property
    def display_name(self) -> str:
        return "Bocha (博查)"

    def is_available(self) -> bool:
        """Return True when BOCHA_API_KEY is set."""
        return bool(os.getenv("BOCHA_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import requests
        api_key = os.getenv("BOCHA_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "BOCHA_API_KEY not set. Get a key at https://open.bochaai.com."}
        try:
            r = requests.post("https://api.bochaai.com/v1/web-search", json={"query": query, "count": min(limit, 20), "summary": True}, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=10)
            r.raise_for_status()
            data = r.json()
            raw = list(((data or {}).get("webPages", []) or {}).get("value", []))[:limit]
            results = [
                {"title": str(i.get("name", "")), "url": str(i.get("url", "")),
                 "description": str(i.get("summary", i.get("snippet", ""))), "position": n + 1}
                for n, i in enumerate(raw)
            ]
            return {"success": True, "data": {"web": results}}
        except Exception as e:
            return {"success": False, "error": str(e)}
    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Bocha (博查)",
            "badge": "free",
            "tag": "Free for personal use, best Chinese content quality. 1,000 free queries starter pack.",
            "env_vars": [
                {
                    "key": "BOCHA_API_KEY",
                    "prompt": "Bocha (博查) API key",
                    "url": "https://open.bochaai.com",
                },
            ],
        }
