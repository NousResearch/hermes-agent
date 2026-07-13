"""Kimi Router Web Search Provider — 中文走 Kimi，英文走 Firecrawl。"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

KIMI_KEY = os.environ.get("KIMI_CN_API_KEY", "")
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
KIMI_URL = "https://api.moonshot.cn/v1/chat/completions"
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v1/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"

_CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')


def _is_chinese(text: str) -> bool:
    return bool(_CHINESE_RE.search(text))


class KimiRouterProvider(WebSearchProvider):
    @property
    def name(self) -> str:
        return "kimi-router"

    @property
    def display_name(self) -> str:
        return "Kimi Router (中文→Kimi / English→Firecrawl)"

    def is_available(self) -> bool:
        return bool(KIMI_KEY)

    def supports_extract(self) -> bool:
        """Extract delegated to Firecrawl (only when English)."""
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        if _is_chinese(query):
            return self._kimi_search(query, limit)
        else:
            return self._firecrawl_search(query, limit)

    def extract(self, urls: List[str], **kwargs: Any) -> Any:
        """Extract always uses Firecrawl."""
        return self._firecrawl_extract(urls, **kwargs)

    # --- Kimi ---
    def _kimi_search(self, query: str, limit: int) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KIMI_KEY}",
        }
        body = {
            "model": "kimi-k2.6",
            "thinking": {"type": "disabled"},
            "messages": [{
                "role": "user",
                "content": f"搜索以下内容，返回{limit}条结果，每条包含标题、URL和50字摘要：\n\n{query}"
            }],
            "max_tokens": 2000,
            "tools": [{"type": "builtin_function", "function": {"name": "$web_search"}}],
            "tool_choice": "auto",
        }

        try:
            with httpx.Client(timeout=30) as client:
                r1 = client.post(KIMI_URL, headers=headers, json=body)
                r1.raise_for_status()
                data1 = r1.json()
        except Exception as e:
            return {"success": False, "error": f"Kimi step 1: {e}"}

        msg = data1.get("choices", [{}])[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            content = msg.get("content", "")
            return {
                "success": True,
                "data": {"web": [{"title": "Kimi", "url": "", "description": content[:500], "position": 1}]}
            }

        tc = tool_calls[0]
        search_args = json.loads(tc["function"]["arguments"])

        body2 = {
            "model": "kimi-k2.6",
            "thinking": {"type": "disabled"},
            "messages": [
                {"role": "user", "content": f"搜索：{query}"},
                {"role": "assistant", "content": None, "tool_calls": [tc]},
                {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(search_args)},
            ],
            "max_tokens": 1500,
            "tools": [{"type": "builtin_function", "function": {"name": "$web_search"}}],
        }

        try:
            with httpx.Client(timeout=60) as client:
                r2 = client.post(KIMI_URL, headers=headers, json=body2)
                r2.raise_for_status()
                data2 = r2.json()
        except Exception as e:
            return {"success": False, "error": f"Kimi step 2: {e}"}

        result = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "success": True,
            "data": {"web": [{"title": query, "url": "", "description": result[:1000], "position": 1}]}
        }

    # --- Firecrawl ---
    def _firecrawl_search(self, query: str, limit: int) -> Dict[str, Any]:
        if not FIRECRAWL_KEY:
            return {"success": False, "error": "FIRECRAWL_API_KEY not set"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FIRECRAWL_KEY}",
        }
        try:
            with httpx.Client(timeout=30) as client:
                r = client.post(FIRECRAWL_SEARCH_URL, headers=headers, json={
                    "query": query,
                    "limit": limit,
                    "scrapeOptions": {"formats": ["markdown"]}
                })
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            return {"success": False, "error": f"Firecrawl: {e}"}

        results = data.get("data", [])
        web = []
        for i, item in enumerate(results[:limit]):
            web.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", item.get("markdown", ""))[:500],
                "position": i + 1,
            })
        return {"success": True, "data": {"web": web}}

    def _firecrawl_extract(self, urls: List[str], **kwargs: Any) -> Any:
        if not FIRECRAWL_KEY:
            return [{"url": u, "error": "FIRECRAWL_API_KEY not set"} for u in urls]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FIRECRAWL_KEY}",
        }
        results = []
        for url in urls:
            try:
                with httpx.Client(timeout=60) as client:
                    r = client.post(FIRECRAWL_SCRAPE_URL, headers=headers, json={
                        "url": url,
                        "formats": ["markdown"]
                    })
                    r.raise_for_status()
                    d = r.json().get("data", {})
                    results.append({
                        "url": url,
                        "title": d.get("metadata", {}).get("title", ""),
                        "content": d.get("markdown", ""),
                        "raw_content": d.get("markdown", ""),
                    })
            except Exception as e:
                results.append({"url": url, "error": str(e)})
        return results
