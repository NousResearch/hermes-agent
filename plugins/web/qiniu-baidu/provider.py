"""Qiniu Cloud Baidu Search — OpenAI-compatible Baidu Search API. Requires QINIU_API_KEY from https://qiniu.com/ai/models."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class QiniuBaiduWebSearchProvider(WebSearchProvider):
    """Qiniu Baidu Search — free tier search provider."""

    @property
    def name(self) -> str:
        return "qiniu-baidu"

    @property
    def display_name(self) -> str:
        return "Qiniu Baidu Search"

    def is_available(self) -> bool:
        """Return True when QINIU_API_KEY is set."""
        return bool(os.getenv("QINIU_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx
        api_key = os.getenv("QINIU_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "QINIU_API_KEY not set. Get a key at https://qiniu.com/ai/models."}
        try:
            r = httpx.post("https://api.qnaigc.com/v1/search/web", json={"query": query, "max_results": min(limit, 50), "search_type": "web"}, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=10)
            r.raise_for_status()
            data = r.json()
            raw = list(((data or {}).get("data", []) or {}).get("results", []))[:limit]
            results = [
                {"title": str(i.get("title", "")), "url": str(i.get("url", "")),
                 "description": str(i.get("snippet", "")), "position": n + 1}
                for n, i in enumerate(raw)
            ]
            return {"success": True, "data": {"web": results}}
        except Exception as e:
            return {"success": False, "error": str(e)}
    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Qiniu Baidu Search",
            "badge": "free",
            "tag": "300万 new-user tokens, OpenAI-compatible interface wrapping Baidu.",
            "env_vars": [
                {
                    "key": "QINIU_API_KEY",
                    "prompt": "Qiniu Baidu Search API key",
                    "url": "https://qiniu.com/ai/models",
                },
            ],
        }
