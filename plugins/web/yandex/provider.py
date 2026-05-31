"""Yandex Cloud Search API — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Search-only
backend using Yandex Cloud synchronous ``/web/search`` (single request, XML).

Config keys this provider responds to::

    web:
      search_backend: "yandex"
      backend: "yandex"

Env vars::

    YANDEX_CLOUD_API_KEY=...     # Yandex Cloud API key (Search API scope)
    YANDEX_CLOUD_FOLDER_ID=...   # Cloud folder id for billing / routing

Docs: https://yandex.cloud/en/docs/search-api/operations/web-search-sync
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_API_BASE_URL = "https://searchapi.api.cloud.yandex.net/v2"
_SEARCH_ENDPOINT = f"{_API_BASE_URL}/web/search"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} environment variable is not set")
    return value


def _build_group_spec(num_results: int) -> tuple[int, int]:
    """Return (groups_on_page, docs_in_group) for the search payload."""
    count = max(1, min(int(num_results), 20))
    if count <= 3:
        return 1, count
    docs_in_group = 3
    groups_on_page = min(100, (count + docs_in_group - 1) // docs_in_group)
    return groups_on_page, docs_in_group


def _decode_search_response(data: Dict[str, Any]) -> str:
    raw_data = data.get("rawData")
    if not isinstance(raw_data, str) or not raw_data:
        raise ValueError("Yandex Search response missing rawData")

    try:
        return base64.b64decode(raw_data).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to decode Yandex Search rawData: {exc}") from exc


def parse_yandex_xml_results(xml_response: str, *, limit: int) -> List[Dict[str, str]]:
    """Parse Yandex Search XML into flat result rows."""
    if not xml_response:
        raise ValueError("xml_response cannot be empty")

    root = ET.fromstring(xml_response)
    docs = root.findall(".//doc")
    results: List[Dict[str, str]] = []

    for doc in docs:
        if len(results) >= limit:
            break

        url_elem = doc.find("url")
        title_elem = doc.find("title")
        snippet_elem = doc.find("passages/passage")

        url = url_elem.text if url_elem is not None and url_elem.text else ""
        title = title_elem.text if title_elem is not None and title_elem.text else ""
        snippet = snippet_elem.text if snippet_elem is not None and snippet_elem.text else ""

        if not url:
            continue

        results.append(
            {
                "title": title,
                "url": url,
                "description": snippet,
                "domain": urlparse(url).netloc,
            }
        )

    return results


def yandex_cloud_search(query: str, *, limit: int = 5) -> str:
    """Run Yandex Cloud synchronous /web/search and return raw XML."""
    import httpx

    api_key = _require_env("YANDEX_CLOUD_API_KEY")
    folder_id = _require_env("YANDEX_CLOUD_FOLDER_ID")
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }

    groups_on_page, docs_in_group = _build_group_spec(limit)
    payload = {
        "query": {
            "searchType": "SEARCH_TYPE_RU",
            "queryText": query,
            "page": "0",
        },
        "groupSpec": {
            "groupMode": "GROUP_MODE_DEEP",
            "groupsOnPage": str(groups_on_page),
            "docsInGroup": str(docs_in_group),
        },
        "maxPassages": "3",
        "region": "ru",
        "l10N": "LOCALIZATION_RU",
        "folderId": folder_id,
        "responseFormat": "FORMAT_XML",
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(_SEARCH_ENDPOINT, json=payload, headers=headers)
        resp.raise_for_status()
        xml_response = _decode_search_response(resp.json())
        ET.fromstring(xml_response)
        return xml_response


class YandexWebSearchProvider(WebSearchProvider):
    """Yandex Cloud Search API — search-only provider."""

    @property
    def name(self) -> str:
        return "yandex"

    @property
    def display_name(self) -> str:
        return "Yandex Search"

    def is_available(self) -> bool:
        return bool(os.getenv("YANDEX_CLOUD_API_KEY", "").strip()) and bool(
            os.getenv("YANDEX_CLOUD_FOLDER_ID", "").strip()
        )

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        if not query.strip():
            return {"success": False, "error": "query cannot be empty"}

        if not self.is_available():
            return {
                "success": False,
                "error": (
                    "YANDEX_CLOUD_API_KEY and YANDEX_CLOUD_FOLDER_ID must be set. "
                    "See https://yandex.cloud/en/docs/search-api/"
                ),
            }

        bounded_limit = max(1, min(int(limit), 20))

        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Yandex search: '%s' (limit=%d)", query, bounded_limit)
            xml_response = yandex_cloud_search(query, limit=bounded_limit)
            parsed = parse_yandex_xml_results(xml_response, limit=bounded_limit)
            web_results = [
                {
                    "title": row["title"],
                    "url": row["url"],
                    "description": row["description"],
                    "position": idx + 1,
                }
                for idx, row in enumerate(parsed)
            ]
            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — httpx and XML errors
            logger.warning("Yandex search error: %s", exc)
            return {"success": False, "error": f"Yandex search failed: {exc}"}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Yandex Search",
            "badge": "paid",
            "tag": (
                "Yandex Cloud account + active billing required — pay-per-query Search API, "
                "no free tier. Uses YANDEX_CLOUD_* env vars (separate from YANDEX_API_KEY for LLM)."
            ),
            "env_vars": [
                {
                    "key": "YANDEX_CLOUD_API_KEY",
                    "prompt": "Yandex Cloud API key with search-api scope (not the LLM key)",
                    "url": "https://yandex.cloud/en/docs/search-api/quickstart",
                },
                {
                    "key": "YANDEX_CLOUD_FOLDER_ID",
                    "prompt": "Yandex Cloud folder ID for Search API billing",
                    "url": "https://console.cloud.yandex.ru/",
                },
            ],
        }
