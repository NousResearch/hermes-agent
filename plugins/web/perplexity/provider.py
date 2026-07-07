"""Perplexity Search API — plugin form.

Config keys this provider responds to::

    web:
      search_backend: "perplexity"
      backend: "perplexity"

Auth env var::

    PERPLEXITY_API_KEY=...

``PPLX_API_KEY`` is accepted as a legacy alias, but ``PERPLEXITY_API_KEY`` is
the documented name in the current Perplexity API docs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_PERPLEXITY_SEARCH_ENDPOINT = "https://api.perplexity.ai/search"


def _api_key() -> str:
    for name in ("PERPLEXITY_API_KEY", "PPLX_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value

    try:
        from hermes_cli.config import get_env_value
    except Exception:
        return ""

    for name in ("PERPLEXITY_API_KEY", "PPLX_API_KEY"):
        value = (get_env_value(name) or "").strip()
        if value:
            return value

    return ""


class PerplexityWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by the Perplexity Search API."""

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity Search"

    def is_available(self) -> bool:
        """Return True when a Perplexity API key is available."""
        return bool(_api_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Perplexity Search API query and normalize the results."""
        import httpx

        api_key = _api_key()
        if not api_key:
            return {
                "success": False,
                "error": "PERPLEXITY_API_KEY is not set",
            }

        max_results = max(1, min(int(limit), 20))

        try:
            response = httpx.post(
                _PERPLEXITY_SEARCH_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "max_results": max_results,
                    "search_context_size": "medium",
                },
                timeout=30,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning("Perplexity Search HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"Perplexity Search returned HTTP {status}",
            }
        except httpx.RequestError as exc:
            logger.warning("Perplexity Search request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach Perplexity Search: {exc}",
            }

        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Perplexity Search response parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse Perplexity Search response as JSON",
            }

        raw_results = payload.get("results") or []
        web_results = []
        for index, item in enumerate(raw_results[:max_results]):
            if not isinstance(item, dict):
                continue
            web_results.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "description": str(item.get("snippet") or ""),
                    "position": index + 1,
                }
            )

        logger.info(
            "Perplexity Search '%s': %d results (limit %d)",
            query,
            len(web_results),
            max_results,
        )

        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Perplexity Search",
            "badge": "API · search only",
            "tag": "Ranked results from Perplexity Search API; pair with Firecrawl/Exa/Tavily for extraction.",
            "env_vars": [
                {
                    "key": "PERPLEXITY_API_KEY",
                    "prompt": "Perplexity API key",
                    "url": "https://console.perplexity.ai/",
                },
            ],
        }
