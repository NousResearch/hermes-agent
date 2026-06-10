# plugins/web/my-backend/provider.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

class DiffbotWebSearchProvider(WebSearchProvider):
    """Minimal search-only provider against the Diffbot HTTP API."""

    @property
    def name(self) -> str:
        # Stable id used in web.search_backend / web.extract_backend / web.backend
        # config keys. Lowercase, no spaces; hyphens permitted.
        return "diffbot"

    @property
    def display_name(self) -> str:
        # Human label shown in `hermes tools`. Defaults to `name`.
        # Corrected typo in Diffbot
        return "Diffbot Web Search"

    def is_available(self) -> bool:
        # Cheap check — env var present, optional dep importable, etc.
        # MUST NOT make network calls (runs on every `hermes tools` paint).
        return bool(os.getenv("DIFFBOT_API_TOKEN", "").strip()) or bool(os.getenv("DIFFBOT_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.getenv("DIFFBOT_API_TOKEN") or os.getenv("DIFFBOT_API_KEY")
        if not api_key:
            raise KeyError("Neither DIFFBOT_API_TOKEN nor DIFFBOT_API_KEY is set in the environment.")
        try:
            resp = httpx.get(
                "https://llm.diffbot.com/api/v1/web_search",
                params={"text": query, "count": max(1, min(int(limit), 20))},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            return {"success": False, "error": str(exc)}

        # Response shape is fixed — see "Response shape" below.
        results_list = data.get("search_results") or data.get("results") or []
        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("pageUrl") or item.get("url") or "",
                        "description": item.get("content") or item.get("snippet") or item.get("description") or "",
                        "position": idx + 1,
                    }
                    for idx, item in enumerate(results_list)
                ],
            },
        }