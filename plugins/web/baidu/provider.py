"""Baidu AI Search API (Qianfan) — 100 queries/day free. Requires BAIDU_API_KEY from https://cloud.baidu.com (bce-v3 ALTAK format)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_QIANFAN_ENDPOINT = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"


class BaiduWebSearchProvider(WebSearchProvider):
    """Baidu Search (Qianfan AI Search) — free tier search provider."""

    @property
    def name(self) -> str:
        return "baidu"

    @property
    def display_name(self) -> str:
        return "Baidu Search"

    def is_available(self) -> bool:
        """Return True when BAIDU_API_KEY is set (bce-v3 ALTAK format)."""
        return bool(os.getenv("BAIDU_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.getenv("BAIDU_API_KEY", "").strip()
        if not api_key:
            return {
                "success": False,
                "error": "BAIDU_API_KEY not set. Get a key at https://cloud.baidu.com (bce-v3 ALTAK format).",
            }

        try:
            r = httpx.post(
                _QIANFAN_ENDPOINT,
                json={
                    "messages": [{"role": "user", "content": query}],
                    "model": "ernie-4.5-turbo-128k",
                    "stream": False,
                    "search_source": "baidu_search_v2",
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
                proxy=None,  # domestic endpoint — skip system proxy
            )
            r.raise_for_status()
            data = r.json()

            # Bail on zero results (web only — LLM answer is extra)
            refs = data.get("references", [])
            if not refs:
                return {"success": True, "data": {"web": []}}

            raw = refs[:limit]
            results = [
                {
                    "title": str(i.get("title", "")),
                    "url": str(i.get("url", "")),
                    "description": str(i.get("content", i.get("snippet", ""))),
                    "position": n + 1,
                }
                for n, i in enumerate(raw)
            ]
            return {"success": True, "data": {"web": results}}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Baidu Search",
            "badge": "free",
            "tag": "100 queries/day free, native Chinese content via Qianfan AI Search.",
            "env_vars": [
                {
                    "key": "BAIDU_API_KEY",
                    "prompt": "Baidu API Key (bce-v3 ALTAK format)",
                    "url": "https://cloud.baidu.com",
                },
            ],
        }
