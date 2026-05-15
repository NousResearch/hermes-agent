"""Brave Search API web provider with Search plan support."""

from __future__ import annotations

import logging
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider
from plugins.web.brave_search.client import BraveSearchApiClient, is_brave_search_configured

logger = logging.getLogger(__name__)


class BraveSearchWebProvider(WebSearchProvider):
    """Search-only provider for Brave Search API paid or credit-enabled accounts."""

    @property
    def name(self) -> str:
        return "brave-search"

    @property
    def display_name(self) -> str:
        return "Brave Search API"

    def is_available(self) -> bool:
        return is_brave_search_configured()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute Brave web search and return the generic Hermes web shape."""
        response = BraveSearchApiClient().search_web(query, limit=limit)
        if not response.get("success"):
            return response

        web = response.get("data", {}).get("web", [])
        normalized = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "description": str(item.get("description", "")),
                "position": int(item.get("position", i + 1)),
            }
            for i, item in enumerate(web)
        ]

        logger.info("Brave Search API '%s': %d results", query, len(normalized))
        return {"success": True, "data": {"web": normalized}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Brave Search API",
            "badge": "paid · search · grounding",
            "tag": "Brave Search API Search plan with web, images, news, videos, suggestions, and grounding context.",
            "env_vars": [
                {
                    "key": "BRAVE_SEARCH_API_KEY",
                    "prompt": "Brave Search API key",
                    "url": "https://brave.com/search/api/",
                },
            ],
        }
