"""Serper.dev web search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses the
Serper.dev Google Search API (``POST https://google.serper.dev/search``).

Search-only — content extraction is not supported.

Config keys this provider responds to::

    web:
      search_backend: "serper"       # explicit per-capability
      backend: "serper"              # shared fallback

Auth env var::

    SERPER_API_KEY=...  # https://serper.dev (free tier: 100 queries/mo)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_SERPER_ENDPOINT = "https://google.serper.dev/search"


class SerperWebSearchProvider(WebSearchProvider):
    """Search-only Serper.dev provider using the Google Search API."""

    @property
    def name(self) -> str:
        return "serper"

    @property
    def display_name(self) -> str:
        return "Serper.dev"

    def is_available(self) -> bool:
        return bool(os.getenv("SERPER_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "SERPER_API_KEY is not set"}

        count = max(1, min(int(limit), 20))

        try:
            resp = httpx.post(
                _SERPER_ENDPOINT,
                json={"q": query, "num": count},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = ""
            try:
                body = exc.response.text[:200]
            except Exception:
                pass
            logger.warning("Serper.dev HTTP error: %d %s", status, body)
            return {
                "success": False,
                "error": f"Serper.dev returned HTTP {status}" + (f" — {body}" if body else ""),
            }
        except httpx.RequestError as exc:
            logger.warning("Serper.dev request error: %s", exc)
            return {"success": False, "error": f"Could not reach Serper.dev: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Serper.dev response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Serper.dev response as JSON"}

        organic = data.get("organic", []) or []
        truncated = organic[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("link", "")),
                "description": str(r.get("snippet", "")),
                "position": i + 1,
            }
            for i, r in enumerate(truncated)
        ]

        logger.info(
            "Serper.dev search '%s': %d results (limit %d)",
            query, len(web_results), limit,
        )

        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Serper.dev",
            "badge": "free",
            "tag": "Google Search API via serper.dev — 100 queries/mo free tier.",
            "env_vars": [
                {
                    "key": "SERPER_API_KEY",
                    "prompt": "Serper.dev API key (free tier)",
                    "url": "https://serper.dev",
                },
            ],
        }
