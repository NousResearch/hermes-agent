"""Kagi web search + extract provider via API v1.

Calls Kagi Search API (POST /api/v1/search) and Extract API
(POST /api/v1/extract) directly — no script dependency.

Configuration::

    # ~/.hermes/.env
    KAGI_API_KEY=<api_key>

    # ~/.hermes/config.yaml
    web:
      search_backend: kagi
      extract_backend: kagi
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import requests

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://kagi.com/api/v1/search"
_EXTRACT_URL = "https://kagi.com/api/v1/extract"


def _load_api_key() -> str | None:
    """Return KAGI_API_KEY from env or ~/.hermes/.env, or None."""
    env_val = os.environ.get("KAGI_API_KEY")
    if env_val:
        return env_val
    env_path = Path(os.path.expanduser("~/.hermes/.env"))
    try:
        for line in env_path.read_text().splitlines():
            if line.startswith("KAGI_API_KEY="):
                val = line.strip().split("=", 1)[1]
                if val:
                    return val
    except (OSError, FileNotFoundError):
        pass
    return None


def _has_kagi_api_key() -> bool:
    return _load_api_key() is not None


class KagiWebSearchProvider(WebSearchProvider):
    """Search + extract via Kagi API v1 (Bearer token auth)."""

    @property
    def name(self) -> str:
        return "kagi"

    @property
    def display_name(self) -> str:
        return "Kagi"

    def is_available(self) -> bool:
        return _has_kagi_api_key()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # -- search ---------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        api_key = _load_api_key()
        if not api_key:
            return {"success": False, "error": "KAGI_API_KEY not found in env or ~/.hermes/.env"}

        payload: Dict[str, Any] = {"query": query}
        if limit:
            payload["limit"] = limit

        try:
            resp = requests.post(
                _SEARCH_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Hermes-KagiBot/2.0",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            return {"success": False, "error": "Kagi search timed out (30s)"}
        except requests.RequestException as exc:
            return {"success": False, "error": f"Kagi search failed: {exc}"}

        items = data.get("data", {}).get("search", [])[:limit]
        web_results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": (item.get("snippet") or "").replace("\n", " "),
                "position": i + 1,
            }
            for i, item in enumerate(items)
        ]

        logger.info("Kagi search '%s': %d results (limit %d)", query, len(web_results), limit)
        return {"success": True, "data": {"web": web_results}}

    # -- extract --------------------------------------------------------------

    def extract(self, urls: List[str], **kwargs: Any) -> Any:
        api_key = _load_api_key()
        if not api_key:
            return {"success": False, "error": "KAGI_API_KEY not found in env or ~/.hermes/.env"}

        if len(urls) > 10:
            urls = urls[:10]
            logger.warning("Kagi Extract API supports max 10 URLs, truncating")

        timeout = kwargs.pop("timeout", None)
        payload: Dict[str, Any] = {
            "pages": [{"url": u} for u in urls],
            "format": "json",
        }
        if timeout:
            payload["timeout"] = timeout

        try:
            resp = requests.post(
                _EXTRACT_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Hermes-KagiBot/2.0",
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            return {"success": False, "error": "Kagi extract timed out (60s)"}
        except requests.RequestException as exc:
            return {"success": False, "error": f"Kagi extract failed: {exc}"}

        results = []
        for page in data.get("data", []):
            url = page.get("url", "")
            markdown = page.get("markdown")
            error = page.get("error")

            if error:
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": error,
                })
            elif markdown:
                # Use first heading as title, fallback to URL
                title = ""
                for line in markdown.splitlines():
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
                results.append({
                    "url": url,
                    "title": title or url,
                    "content": markdown,
                    "raw_content": markdown,
                })
            else:
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": "no content",
                })

        logger.info("Kagi extract: %d URLs, %d results", len(urls), len(results))
        return results

    # -- setup schema ---------------------------------------------------------

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Kagi",
            "badge": "paid · API v1 · search + extract",
            "tag": "Search and extract via Kagi API v1 — set KAGI_API_KEY in ~/.hermes/.env",
            "env_vars": [
                {
                    "key": "KAGI_API_KEY",
                    "prompt": "Kagi API key",
                    "url": "https://kagi.com/settings?p=api",
                },
            ],
        }
