"""TinyFish web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True (TinyFish ``/search``)
- ``supports_extract()`` -> True (TinyFish ``/extract``)

TinyFish uses:
- GET  https://api.search.tinyfish.ai with ``X-API-Key`` header for search
- POST https://api.fetch.tinyfish.ai with ``X-API-Key`` header for extract

Config keys this provider responds to::

    web:
      search_backend: "tinyfish"     # explicit per-capability
      extract_backend: "tinyfish"   # explicit per-capability
      backend: "tinyfish"           # shared fallback for all

Env vars::

    TINYFISH_API_KEY=...     # required
    TINYFISH_LOCATION=...    # optional (e.g. "us", "eu")
    TINYFISH_LANGUAGE=...    # optional (e.g. "en", "de")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.search.tinyfish.ai"
EXTRACT_URL = "https://api.fetch.tinyfish.ai"


def _tinyfish_request(
    method: str, url: str, params: Dict[str, Any] | None = None, json: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Make an authenticated request to the TinyFish API.

    Raises ``ValueError`` when ``TINYFISH_API_KEY`` is unset; the caller
    catches and surfaces as a typed error response.
    """
    api_key = os.getenv("TINYFISH_API_KEY")
    if not api_key:
        raise ValueError(
            "TINYFISH_API_KEY environment variable not set. "
            "Set TINYFISH_API_KEY to use TinyFish."
        )

    headers = {"X-API-Key": api_key}
    logger.info("TinyFish %s request to %s", method, url)

    with httpx.Client(timeout=60) as client:
        if method.upper() == "GET":
            response = client.get(url, headers=headers, params=params)
        else:
            response = client.post(url, headers=headers, json=json)
    response.raise_for_status()
    return response.json()


def _normalize_tinyfish_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map TinyFish search response to ``{success, data: {web: [...]}}``."""
    web_results = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("snippet", "") or result.get("description", ""),
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def _normalize_tinyfish_documents(
    response: Dict[str, Any], fallback_url: str = ""
) -> List[Dict[str, Any]]:
    """Map TinyFish ``/extract`` response to standard documents.

    Documents follow the legacy LLM post-processing shape::

        {"url", "title", "content", "raw_content", "metadata"}

    Failures become result entries with an ``error`` field rather than raising.
    """
    documents: List[Dict[str, Any]] = []
    for result in response.get("results", []):
        url = result.get("url", fallback_url)
        raw = result.get("raw_content", "") or result.get("content", "")
        documents.append(
            {
                "url": url,
                "title": result.get("title", ""),
                "content": raw,
                "raw_content": raw,
                "metadata": {"sourceURL": url, "title": result.get("title", "")},
            }
        )
    for fail in response.get("failed_results", []):
        documents.append(
            {
                "url": fail.get("url", fallback_url),
                "title": "",
                "content": "",
                "raw_content": "",
                "error": fail.get("error", "extraction failed"),
                "metadata": {"sourceURL": fail.get("url", fallback_url)},
            }
        )
    return documents


class TinyfishWebSearchProvider(WebSearchProvider):
    """TinyFish search + extract provider."""

    @property
    def name(self) -> str:
        return "tinyfish"

    @property
    def display_name(self) -> str:
        return "TinyFish"

    def is_available(self) -> bool:
        """Return True when ``TINYFISH_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("TINYFISH_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def supports_crawl(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a TinyFish search."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("TinyFish search: '%s' (limit=%d)", query, limit)

            params: Dict[str, Any] = {"query": query}
            location = os.getenv("TINYFISH_LOCATION", "").strip()
            if location:
                params["location"] = location
            language = os.getenv("TINYFISH_LANGUAGE", "").strip()
            if language:
                params["language"] = language

            raw = _tinyfish_request("GET", SEARCH_URL, params=params)
            if limit:
                raw = dict(raw)
                raw["results"] = list(raw.get("results", []))[: min(limit, 100)]
            return _normalize_tinyfish_search_results(raw)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("TinyFish search error: %s", exc)
            return {"success": False, "error": f"TinyFish search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via TinyFish.

        Returns the legacy list-of-results shape; per-URL failures become
        items with ``error``.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("TinyFish extract: %d URL(s)", len(urls))

            payload: Dict[str, Any] = {
                "urls": urls,
            }

            raw = _tinyfish_request("POST", EXTRACT_URL, json=payload)
            return _normalize_tinyfish_documents(
                raw, fallback_url=urls[0] if urls else ""
            )
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except Exception as exc:  # noqa: BLE001
            logger.warning("TinyFish extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"TinyFish extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "TinyFish",
            "badge": "paid",
            "tag": "Web search + content extraction.",
            "env_vars": [
                {
                    "key": "TINYFISH_API_KEY",
                    "prompt": "TinyFish API key",
                    "url": "https://tinyfish.ai",
                },
            ],
        }