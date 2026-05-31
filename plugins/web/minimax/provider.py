"""MiniMax Token Plan search provider."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class MiniMaxWebSearchProvider(WebSearchProvider):
    """Search provider backed by MiniMax Coding Plan search API."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        return bool(os.getenv("MINIMAX_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.getenv("MINIMAX_API_KEY", "").strip()
        api_host = os.getenv("MINIMAX_API_HOST", "https://api.minimax.io").strip().rstrip("/")
        if not api_key:
            return {"success": False, "error": "MINIMAX_API_KEY is not set"}
        if not query.strip():
            return {"success": False, "error": "query is empty"}

        try:
            resp = httpx.post(
                f"{api_host}/v1/coding_plan/search",
                json={"q": query},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "MM-API-Source": "Hermes",
                    "Content-Type": "application/json",
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("MiniMax search HTTP error: %s", exc)
            return {"success": False, "error": f"MiniMax returned HTTP {exc.response.status_code}"}
        except httpx.RequestError as exc:
            logger.warning("MiniMax search request error: %s", exc)
            return {"success": False, "error": f"Could not reach MiniMax: {exc}"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("MiniMax search parse error: %s", exc)
            return {"success": False, "error": "Could not parse MiniMax response as JSON"}

        base_resp = data.get("base_resp") or {}
        if base_resp and int(base_resp.get("status_code") or 0) != 0:
            return {"success": False, "error": f"MiniMax API error {base_resp.get('status_code')}: {base_resp.get('status_msg')}"}

        organic = data.get("organic") or []
        web_results = []
        for i, r in enumerate(organic[: max(1, min(int(limit), 20))]):
            web_results.append({
                "title": str(r.get("title", "")),
                "url": str(r.get("link", "")),
                "description": str(r.get("snippet", "")),
                "date": str(r.get("date", "")),
                "position": i + 1,
            })

        logger.info("MiniMax search '%s': %d results", query, len(web_results))
        return {"success": True, "data": {"web": web_results}, "provider": "minimax"}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "subscription · Token Plan",
            "tag": "MiniMax Coding Plan search API. Search only; use a reader/extractor for page content.",
            "env_vars": [
                {"key": "MINIMAX_API_KEY", "prompt": "MiniMax Token Plan API key", "url": "https://platform.minimax.io/"},
                {"key": "MINIMAX_API_HOST", "prompt": "MiniMax API host", "default": "https://api.minimax.io"},
            ],
        }
