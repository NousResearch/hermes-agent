"""Jina AI Search — full-page content extraction. 10M tokens free for new users. Requires JINA_API_KEY from https://jina.ai. NOTE: blocked by GFW in China."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class JinaWebSearchProvider(WebSearchProvider):
    """Jina AI — free tier search provider."""

    @property
    def name(self) -> str:
        return "jina"

    @property
    def display_name(self) -> str:
        return "Jina AI"

    def is_available(self) -> bool:
        """Return True when JINA_API_KEY is set."""
        return bool(os.getenv("JINA_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx
        api_key = os.getenv("JINA_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "JINA_API_KEY not set. Get a key at https://jina.ai (10M free tokens for new users)."}
        try:
            import urllib.parse
            r = httpx.get(f"https://s.jina.ai/{urllib.parse.quote(query)}",
                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                timeout=15)
            r.raise_for_status()
            data = r.json()
            raw = data.get("data", [])[:limit]
            results = [
                {"title": str(i.get("title", "")), "url": str(i.get("url", "")),
                 "description": str(i.get("description", i.get("content", "")))[:500], "position": n + 1}
                for n, i in enumerate(raw)
            ]
            return {"success": True, "data": {"web": results}}
        except Exception as e:
            return {"success": False, "error": str(e)}
    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Jina AI",
            "badge": "free",
            "tag": "10M tokens free for new users. NOT accessible from mainland China without proxy.",
            "env_vars": [
                {
                    "key": "JINA_API_KEY",
                    "prompt": "Jina AI API key",
                    "url": "https://jina.ai",
                },
            ],
        }
