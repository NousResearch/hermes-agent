"""Baidu AI Search (Qianfan) — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses the
Baidu AI Search Engine API from the Qianfan platform.

Search-only — content extraction is not supported.

Config keys this provider responds to::

    web:
      search_backend: "baidu"        # explicit per-capability
      backend: "baidu"               # shared fallback

Auth env var::

    BAIDU_API_KEY=...  # https://console.bce.baidu.com/ai-search/qianfan/ais/console/apiKey
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_BAIDU_SEARCH_ENDPOINT = "https://qianfan.baidubce.com/v2/ai_search/web_search"


class BaiduWebSearchProvider(WebSearchProvider):
    """Search-only Baidu AI Search provider via Qianfan."""

    @property
    def name(self) -> str:
        return "baidu"

    @property
    def display_name(self) -> str:
        return "Baidu Search"

    def is_available(self) -> bool:
        return bool(os.getenv("BAIDU_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.getenv("BAIDU_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "BAIDU_API_KEY is not set"}

        count = max(1, min(int(limit), 50))

        try:
            resp = httpx.post(
                _BAIDU_SEARCH_ENDPOINT,
                json={
                    "messages": [
                        {"content": query, "role": "user"},
                    ],
                    "search_source": "baidu_search_v2",
                    "resource_type_filter": [
                        {"type": "web", "top_k": count},
                    ],
                    "search_filter": {},
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Appbuilder-From": "hermes-agent",
                },
                timeout=20,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = ""
            try:
                body = exc.response.text[:300]
            except Exception:
                pass
            logger.warning("Baidu Search HTTP error: %d %s", status, body)
            return {
                "success": False,
                "error": f"Baidu Search returned HTTP {status}" + (f" — {body}" if body else ""),
            }
        except httpx.RequestError as exc:
            logger.warning("Baidu Search request error: %s", exc)
            return {"success": False, "error": f"Could not reach Baidu Search: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Baidu Search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Baidu Search response as JSON"}

        # Check for API-level error codes
        if "code" in data:
            msg = data.get("message", str(data.get("code", "unknown")))
            logger.warning("Baidu Search API error: %s", msg)
            return {"success": False, "error": f"Baidu Search API error: {msg}"}

        # Results come in the "references" array
        raw_results = data.get("references", []) or []
        truncated = raw_results[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("link", "") or r.get("url", "")),
                "description": str(r.get("content", "") or r.get("snippet", "")),
                "position": i + 1,
            }
            for i, r in enumerate(truncated)
        ]

        logger.info(
            "Baidu Search '%s': %d results (limit %d)",
            query, len(web_results), limit,
        )

        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Baidu Search",
            "badge": "free · paid",
            "tag": "Baidu AI Search via Qianfan — Chinese-optimized web search results.",
            "env_vars": [
                {
                    "key": "BAIDU_API_KEY",
                    "prompt": "Baidu Qianfan API key",
                    "url": "https://console.bce.baidu.com/ai-search/qianfan/ais/console/apiKey",
                },
            ],
        }
