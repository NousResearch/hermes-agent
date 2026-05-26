"""Perplexity Search API web search provider.

Config keys this provider responds to::

    web:
      search_backend: "perplexity"     # explicit per-capability
      backend: "perplexity"            # shared fallback (search-only)

Env vars::

    PERPLEXITY_API_KEY=...             # https://www.perplexity.ai/settings/api

This provider implements Perplexity's structured Search API (``POST /search``),
not Sonar chat completions. It returns ranked web result metadata only; use a
separate extract-capable backend for ``web_extract``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


_PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"


def _perplexity_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to Perplexity Search API and return parsed JSON."""
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable not set. "
            "Get your API key at https://www.perplexity.ai/settings/api"
        )

    response = httpx.post(
        _PERPLEXITY_SEARCH_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _normalize_perplexity_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map Perplexity ``/search`` response to Hermes' web_search shape."""
    web_results: List[Dict[str, Any]] = []
    for i, result in enumerate(response.get("results", []) or []):
        if not isinstance(result, dict):
            continue
        web_results.append(
            {
                "title": result.get("title", "") or "",
                "url": result.get("url", "") or "",
                "description": result.get("snippet", "") or "",
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


class PerplexityWebSearchProvider(WebSearchProvider):
    """Perplexity Search API provider for ``web_search``."""

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity Search"

    def is_available(self) -> bool:
        """Return True when ``PERPLEXITY_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("PERPLEXITY_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def supports_crawl(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Perplexity Search API query."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            max_results = min(max(int(limit), 1), 20)
            logger.info("Perplexity search: '%s' (limit=%d)", query, max_results)
            raw = _perplexity_request(
                {
                    "query": query,
                    "max_results": max_results,
                }
            )
            return _normalize_perplexity_search_results(raw)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Perplexity search error: %s", exc)
            return {"success": False, "error": f"Perplexity search failed: {exc}"}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Perplexity Search",
            "badge": "paid",
            "tag": "Structured Search API results from Perplexity; search-only.",
            "env_vars": [
                {
                    "key": "PERPLEXITY_API_KEY",
                    "prompt": "Perplexity API key",
                    "url": "https://www.perplexity.ai/settings/api",
                },
            ],
        }
