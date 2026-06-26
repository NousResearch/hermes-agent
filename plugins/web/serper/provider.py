"""Serper.dev — Google Search API. 2,500 free queries (no credit card). Requires SERPER_API_KEY from https://serper.dev."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class SerperWebSearchProvider(WebSearchProvider):
    """Serper (Google Search) — free tier search provider."""

    @property
    def name(self) -> str:
        return "serper"

    @property
    def display_name(self) -> str:
        return "Serper (Google Search)"

    def is_available(self) -> bool:
        """Return True when SERPER_API_KEY is set."""
        return bool(os.getenv("SERPER_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import requests
        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "SERPER_API_KEY not set. Get a key at https://serper.dev."}
        try:
            r = requests.post("https://google.serper.dev/search", json={"q": query, "num": min(limit, 100)}, headers={"X-API-KEY": api_key}, timeout=10)
            r.raise_for_status()
            data = r.json()
            raw = data.get("organic", [])[:limit]
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
            "name": "Serper (Google Search)",
            "badge": "free",
            "tag": "2,500 free queries/month, no credit card — Google SERP backed.",
            "env_vars": [
                {
                    "key": "SERPER_API_KEY",
                    "prompt": "Serper (Google Search) API key",
                    "url": "https://serper.dev",
                },
            ],
        }
