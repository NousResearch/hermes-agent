"""Yandex Search API — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Search-only —
no dedicated extract endpoint exists in the Yandex Search API, so pair this
provider with Firecrawl/Tavily/Exa/Parallel for ``web_extract`` (same pattern
as SearXNG / Brave Search free tier).

Config keys this provider responds to::

    web:
      search_backend: "yandex"     # explicit per-capability
      backend: "yandex"            # shared fallback

Env vars::

    YANDEX_SEARCH_API_KEY=...   # https://yandex.cloud/en/services/search-api
    YANDEX_FOLDER_ID=...        # Yandex Cloud folder ID that owns the API key
    YANDEX_SEARCH_REGION=...    # optional — Yandex geo region ID (e.g. "225" = Russia)
    YANDEX_SEARCH_LANG=...      # optional — result localization, e.g. "LOCALIZATION_EN"
    YANDEX_SEARCH_TYPE=...      # optional — search domain, default "SEARCH_TYPE_RU"

Calls the ``WebSearch.Search`` v2 REST method::

    POST https://searchapi.api.cloud.yandex.net/v2/web/search
    Authorization: Api-Key <YANDEX_SEARCH_API_KEY>

The response wraps the results as base64-encoded XML in a ``rawData`` field
(protobuf JSON mapping for ``bytes``) — this module decodes and parses that
XML into the standard Hermes result shape.
"""

from __future__ import annotations

import base64
import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_YANDEX_ENDPOINT = "https://searchapi.api.cloud.yandex.net/v2/web/search"


def _text(node: Optional[ET.Element]) -> str:
    """Flatten an element's text (including nested emphasis tags) to a plain string."""
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _parse_yandex_xml(raw_xml: bytes, limit: int) -> Dict[str, Any]:
    """Parse a Yandex Search XML response body into the standard result shape.

    Expected structure (``FORMAT_XML``)::

        <yandexsearch>
          <response>
            <error code="...">message</error>          <!-- on failure -->
            <results>
              <grouping>
                <group>
                  <doc>
                    <url>...</url>
                    <title>...</title>
                    <headline>...</headline>
                    <passages><passage>...</passage></passages>
                  </doc>
                </group>
                ...
              </grouping>
            </results>
          </response>
        </yandexsearch>
    """
    root = ET.fromstring(raw_xml)
    response = root.find("response")
    if response is None:
        return {"success": False, "error": "Yandex Search response missing <response> element"}

    error = response.find("error")
    if error is not None:
        message = _text(error) or "unknown error"
        code = error.get("code", "")
        return {
            "success": False,
            "error": f"Yandex Search error{f' ({code})' if code else ''}: {message}",
        }

    web_results: List[Dict[str, Any]] = []
    for doc in response.iter("doc"):
        if len(web_results) >= limit:
            break
        description = _text(doc.find("headline"))
        if not description:
            passage = doc.find("./passages/passage")
            description = _text(passage)
        web_results.append(
            {
                "title": _text(doc.find("title")),
                "url": _text(doc.find("url")),
                "description": description,
                "position": len(web_results) + 1,
            }
        )

    return {"success": True, "data": {"web": web_results}}


class YandexWebSearchProvider(WebSearchProvider):
    """Search-only provider for the Yandex Cloud Search API (WebSearch.Search v2)."""

    @property
    def name(self) -> str:
        return "yandex"

    @property
    def display_name(self) -> str:
        return "Yandex Search"

    def is_available(self) -> bool:
        """Return True when both ``YANDEX_SEARCH_API_KEY`` and ``YANDEX_FOLDER_ID`` are set."""
        return bool(os.getenv("YANDEX_SEARCH_API_KEY", "").strip()) and bool(
            os.getenv("YANDEX_FOLDER_ID", "").strip()
        )

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the Yandex Search API (``WebSearch.Search``).

        Returns ``{"success": True, "data": {"web": [{"title", "url", "description", "position"}]}}``
        on success, or ``{"success": False, "error": str}`` on failure.
        """
        import httpx

        api_key = os.getenv("YANDEX_SEARCH_API_KEY", "").strip()
        folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
        if not api_key:
            return {"success": False, "error": "YANDEX_SEARCH_API_KEY is not set"}
        if not folder_id:
            return {"success": False, "error": "YANDEX_FOLDER_ID is not set"}

        groups_on_page = str(max(1, min(int(limit), 100)))
        query_spec: Dict[str, Any] = {
            "searchType": os.getenv("YANDEX_SEARCH_TYPE", "SEARCH_TYPE_RU").strip(),
            "queryText": query,
            "page": "0",
        }
        payload: Dict[str, Any] = {
            "query": query_spec,
            "groupSpec": {
                "groupMode": "GROUP_MODE_FLAT",
                "groupsOnPage": groups_on_page,
                "docsInGroup": "1",
            },
            "maxPassages": "4",
            "folderId": folder_id,
            "responseFormat": "FORMAT_XML",
        }
        region = os.getenv("YANDEX_SEARCH_REGION", "").strip()
        if region:
            payload["region"] = region
        lang = os.getenv("YANDEX_SEARCH_LANG", "").strip()
        if lang:
            payload["l10n"] = lang

        try:
            resp = httpx.post(
                _YANDEX_ENDPOINT,
                json=payload,
                headers={
                    "Authorization": f"Api-Key {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Yandex Search HTTP error: %s", exc)
            detail = ""
            try:
                detail = f": {exc.response.json().get('message', '')}"
            except Exception:  # noqa: BLE001 — best-effort error detail
                pass
            return {
                "success": False,
                "error": f"Yandex Search returned HTTP {exc.response.status_code}{detail}",
            }
        except httpx.RequestError as exc:
            logger.warning("Yandex Search request error: %s", exc)
            return {"success": False, "error": f"Could not reach Yandex Search: {exc}"}

        try:
            data = resp.json()
            raw_xml = base64.b64decode(data["rawData"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Yandex Search response parse error: %s", exc)
            return {"success": False, "error": "Could not parse Yandex Search response"}

        try:
            result = _parse_yandex_xml(raw_xml, limit)
        except ET.ParseError as exc:
            logger.warning("Yandex Search XML parse error: %s", exc)
            return {"success": False, "error": f"Could not parse Yandex Search XML payload: {exc}"}

        if result["success"]:
            logger.info(
                "Yandex Search '%s': %d results (limit %d)",
                query, len(result["data"]["web"]), limit,
            )
        return result

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Yandex Search",
            "badge": "paid",
            "tag": "Yandex Cloud web search — search only, requires API key + folder ID.",
            "env_vars": [
                {
                    "key": "YANDEX_SEARCH_API_KEY",
                    "prompt": "Yandex Search API key",
                    "url": "https://yandex.cloud/en/services/search-api",
                },
                {
                    "key": "YANDEX_FOLDER_ID",
                    "prompt": "Yandex Cloud folder ID",
                    "url": "https://yandex.cloud/en/docs/resource-manager/operations/folder/get-id",
                },
            ],
        }
