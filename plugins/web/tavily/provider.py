"""Tavily web search + content extraction + crawl — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Three
capabilities advertised:

- ``supports_search()``  -> True (Tavily ``/search``)
- ``supports_extract()`` -> True (Tavily ``/extract``)
- ``supports_crawl()``   -> True (Tavily ``/crawl``) — sync HTTP crawl;
  Firecrawl also advertises ``supports_crawl=True`` (async)

All three are sync — the underlying call is ``httpx.post(...)``. The
dispatcher in :func:`tools.web_tools.web_crawl_tool` (which is itself
async) will run sync providers in a thread when appropriate.

Config keys this provider responds to::

    web:
      search_backend: "tavily"     # explicit per-capability
      extract_backend: "tavily"    # explicit per-capability
      crawl_backend: "tavily"      # explicit per-capability
      backend: "tavily"            # shared fallback for all three

Env vars::

    TAVILY_API_KEY=...           # https://app.tavily.com/home (required)
    TAVILY_BASE_URL=...          # optional override of https://api.tavily.com

Auth note: Tavily uses ``api_key`` in the JSON body for /search and
/extract, but **also requires** ``Authorization: Bearer <key>`` for /crawl
(body-only auth returns 401 on /crawl). The plugin handles both.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _tavily_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the Tavily API and return the parsed JSON response.

    Mirrors :func:`tools.web_tools._tavily_request`. Raises ``ValueError``
    when ``TAVILY_API_KEY`` is unset; the caller catches and surfaces as
    a typed error response.
    """
    import httpx

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get your API key at https://app.tavily.com/home"
        )

    base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com")
    payload = dict(payload)  # don't mutate caller's dict
    payload["api_key"] = api_key
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)

    # Tavily /crawl requires Bearer header auth in addition to body auth;
    # /search and /extract are body-only.
    headers = {"Authorization": f"Bearer {api_key}"} if endpoint.strip("/") == "crawl" else {}

    response = httpx.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def _normalize_tavily_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map Tavily ``/search`` response to ``{success, data: {web: [...]}}``."""
    web_results = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("content", ""),
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def _normalize_tavily_documents(
    response: Dict[str, Any], fallback_url: str = ""
) -> List[Dict[str, Any]]:
    """Map Tavily ``/extract`` or ``/crawl`` response to standard documents.

    Documents follow the legacy LLM post-processing shape::

        {"url", "title", "content", "raw_content", "metadata"}

    Failures (``failed_results``, ``failed_urls``) become result entries
    with an ``error`` field rather than raising.
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
    for fail_url in response.get("failed_urls", []):
        url_str = fail_url if isinstance(fail_url, str) else str(fail_url)
        documents.append(
            {
                "url": url_str,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": "extraction failed",
                "metadata": {"sourceURL": url_str},
            }
        )
    return documents


class TavilyWebSearchProvider(WebSearchProvider):
    """Tavily search + extract + crawl provider."""

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def display_name(self) -> str:
        return "Tavily"

    def is_available(self) -> bool:
        """Return True when ``TAVILY_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("TAVILY_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def supports_crawl(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Tavily search."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Tavily search: '%s' (limit=%d)", query, limit)
            raw = _tavily_request(
                "search",
                {
                    "query": query,
                    "max_results": min(limit, 20),
                    "include_raw_content": False,
                    "include_images": False,
                },
            )
            return _normalize_tavily_search_results(raw)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Tavily search error: %s", exc)
            return {"success": False, "error": f"Tavily search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Tavily.

        Sync — the underlying call is httpx.post(...). Returns the legacy
        list-of-results shape; per-URL failures become items with ``error``.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("Tavily extract: %d URL(s)", len(urls))
            raw = _tavily_request(
                "extract",
                {
                    "urls": urls,
                    "include_images": False,
                },
            )
            return _normalize_tavily_documents(
                raw, fallback_url=urls[0] if urls else ""
            )
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Tavily extract failed: {exc}"}
                for u in urls
            ]

    def crawl(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Crawl a seed URL via Tavily's ``/crawl`` endpoint.

        Accepted kwargs (others ignored for forward compat):
          - ``instructions``: str — natural-language guidance for the crawl
          - ``depth``: str — ``"basic"`` (default) or ``"advanced"``
          - ``limit``: int — max pages to crawl (default 20)

        Returns ``{"results": [...]}`` shaped to match what
        :func:`tools.web_tools.web_crawl_tool` post-processes.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"results": [{"url": url, "title": "", "content": "", "error": "Interrupted"}]}

            instructions = kwargs.get("instructions")
            depth = kwargs.get("depth", "basic")
            limit = kwargs.get("limit", 20)

            logger.info("Tavily crawl: %s (depth=%s, limit=%d)", url, depth, limit)
            payload: Dict[str, Any] = {
                "url": url,
                "limit": limit,
                "extract_depth": depth,
            }
            if instructions:
                payload["instructions"] = instructions

            raw = _tavily_request("crawl", payload)
            return {
                "results": _normalize_tavily_documents(raw, fallback_url=url)
            }
        except ValueError as exc:
            return {"results": [{"url": url, "title": "", "content": "", "error": str(exc)}]}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily crawl error: %s", exc)
            return {
                "results": [
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"Tavily crawl failed: {exc}",
                    }
                ]
            }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Tavily",
            "badge": "paid",
            "tag": "Search + extract + crawl in one provider.",
            "env_vars": [
                {
                    "key": "TAVILY_API_KEY",
                    "prompt": "Tavily API key",
                    "url": "https://app.tavily.com/home",
                },
            ],
        }
