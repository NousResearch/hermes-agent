"""360 Search (Haosou) — Chinese search engine. NOTE: no public developer API available."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class Three60SearchWebSearchProvider(WebSearchProvider):
    """360 Search (好搜) — unavailable tier search provider."""

    @property
    def name(self) -> str:
        return "360-search"

    @property
    def display_name(self) -> str:
        return "360 Search (好搜)"

    def is_available(self) -> bool:
        """No public API available — always unavailable."""
        return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        return {"success": False, "error": "360 Search (好搜) does not offer a public search API. Use Baidu or Bocha for Chinese-language search."}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "360 Search (好搜)",
            "badge": "unavailable",
            "tag": "No public API available. Listed for registry completeness.",
            "env_vars": [
            ],
        }
