"""MrScraper adapter for Hermes' standard web search/extract tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from agent.web_search_provider import WebSearchProvider
from plugins.browser.mrscraper.provider import fetch_rendered_html
from plugins.mrscraper_client import is_mrscraper_available
from plugins.web.mrscraper.tools import search_google_serp
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access


def _nested(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _serp_items(payload: Any) -> List[Mapping[str, Any]]:
    """Find the result list across currently observed SERP response envelopes."""
    candidates = (
        _nested(payload, "data", "web"),
        _nested(payload, "data", "organic"),
        _nested(payload, "data", "organic_results"),
        _nested(payload, "organic"),
        _nested(payload, "organic_results"),
        _nested(payload, "results"),
    )
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, Mapping)]
    return []


def _rendered_content(payload: Any) -> tuple[str, str, Dict[str, Any]]:
    """Extract content/title metadata while retaining the raw response."""
    if isinstance(payload, str):
        return payload, "", {}
    if not isinstance(payload, Mapping):
        return json.dumps(payload, ensure_ascii=False), "", {}

    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    assert isinstance(data, Mapping)
    content = data.get("markdown") or data.get("html") or data.get("content")
    if content is None:
        content = json.dumps(payload, ensure_ascii=False)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    title = str(data.get("title") or metadata.get("title") or "")
    return str(content), title, dict(metadata)


class MrScraperWebSearchProvider(WebSearchProvider):
    """Route Hermes web search and extraction through MrScraper."""

    @property
    def name(self) -> str:
        return "mrscraper"

    @property
    def display_name(self) -> str:
        return "MrScraper"

    def is_available(self) -> bool:
        return is_mrscraper_available()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            payload = search_google_serp({
                "query": query,
                "region": "us",
                "language": "en",
                "page": 1,
                "format": "json",
                "render_js": False,
            })
            items = _serp_items(payload)[: max(1, int(limit))]
            web = []
            for position, item in enumerate(items, start=1):
                web.append({
                    "title": str(item.get("title") or item.get("name") or ""),
                    "url": str(item.get("url") or item.get("link") or ""),
                    "description": str(
                        item.get("description") or item.get("snippet") or ""
                    ),
                    "position": int(item.get("position") or position),
                })
            return {"success": True, "data": {"web": web}}
        except Exception as exc:  # noqa: BLE001 — provider result contract
            return {"success": False, "error": f"MrScraper search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        requested_format = str(kwargs.get("format") or "markdown").lower()
        for url in urls:
            if not is_safe_url(url):
                results.append({"url": url, "title": "", "error": "Unsafe URL"})
                continue
            blocked = check_website_access(url)
            if blocked:
                results.append({
                    "url": url,
                    "title": "",
                    "error": blocked.get("message") or "Website access blocked",
                })
                continue
            try:
                payload = fetch_rendered_html({
                    "url": url,
                    "html": requested_format != "markdown",
                    "markdown": requested_format == "markdown",
                })
                content, title, metadata = _rendered_content(payload)
                results.append({
                    "url": url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "metadata": metadata,
                })
            except Exception as exc:  # noqa: BLE001 — per-URL error contract
                results.append({
                    "url": url,
                    "title": "",
                    "error": f"MrScraper extract failed: {exc}",
                })
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MrScraper",
            "badge": "paid",
            "tag": "Google SERP, rendered pages, and structured extraction",
            "env_vars": [
                {
                    "key": "MRSCRAPER_API_TOKEN",
                    "prompt": "MrScraper API token",
                    "url": "https://app.mrscraper.com",
                }
            ],
        }
