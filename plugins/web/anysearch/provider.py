"""AnySearch web search + content extraction — bundled, auto-loaded.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True (AnySearch ``/v1/search``)
- ``supports_extract()`` -> True (AnySearch MCP ``extract`` tool)

Config keys this provider responds to::

    web:
      search_backend: "anysearch"     # explicit per-capability
      extract_backend: "anysearch"    # explicit per-capability
      backend: "anysearch"            # shared fallback for both

Env vars::

    ANYSEARCH_API_KEY=...   # https://www.anysearch.com/console/api-keys (required)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANYSEARCH_API_BASE = "https://api.anysearch.com"
_SEARCH_TIMEOUT = 30
_EXTRACT_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_key() -> str:
    """Return the AnySearch API key or raise."""
    key = os.getenv("ANYSEARCH_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "ANYSEARCH_API_KEY environment variable not set. "
            "Get your API key at https://www.anysearch.com/console/api-keys"
        )
    return key


def _search_request(query: str, limit: int = 10) -> Dict[str, Any]:
    """POST to ``/v1/search`` and return the parsed response.

    AnySearch's search returns both ``snippet`` and full ``content``
    in every result — agent-friendly.
    """
    import httpx

    url = f"{ANYSEARCH_API_BASE}/v1/search"
    payload: Dict[str, Any] = {
        "query": query,
        "max_results": min(limit, 10),
    }

    response = httpx.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        timeout=_SEARCH_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _mcp_extract_request(url: str) -> str:
    """Call AnySearch's MCP ``extract`` tool via JSON-RPC over HTTP.

    Falls back to direct HTTP fetch if the MCP endpoint fails.
    Returns clean markdown content.
    """
    import httpx

    mcp_url = f"{ANYSEARCH_API_BASE}/mcp"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "extract",
            "arguments": {"url": url},
        },
    }

    try:
        response = httpx.post(
            mcp_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            timeout=_EXTRACT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        # Extract text content from MCP response
        if isinstance(data, dict) and "result" in data:
            content_items = data["result"].get("content", [])
            for item in content_items:
                if item.get("type") == "text":
                    return item.get("text", "")
    except Exception as exc:
        logger.warning("AnySearch MCP extract failed, falling back to direct fetch: %s", exc)

    # Fallback: direct HTTP fetch with httpx
    return _direct_fetch(url)


def _direct_fetch(url: str) -> str:
    """Fetch a URL directly and extract text content."""
    import httpx
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._text_parts: list[str] = []
            self._skip = False

        def handle_data(self, data: str) -> None:
            if not self._skip:
                stripped = data.strip()
                if stripped:
                    self._text_parts.append(stripped)

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in {"script", "style", "noscript", "svg"}:
                self._skip = True

        def handle_endtag(self, tag: str) -> None:
            if tag in {"script", "style", "noscript", "svg"}:
                self._skip = False

        def text(self) -> str:
            return "\n".join(self._text_parts)

    resp = httpx.get(url, timeout=_EXTRACT_TIMEOUT, follow_redirects=True)
    resp.raise_for_status()
    parser = _TextExtractor()
    parser.feed(resp.text)
    return parser.text()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnySearchWebSearchProvider(WebSearchProvider):
    """AnySearch search + extract provider."""

    @property
    def name(self) -> str:
        return "anysearch"

    @property
    def display_name(self) -> str:
        return "AnySearch"

    def is_available(self) -> bool:
        """Return True when ``ANYSEARCH_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("ANYSEARCH_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an AnySearch search.

        AnySearch returns both ``snippet`` and ``content`` per result.
        We use ``snippet`` for ``description`` (the standard Hermes
        search field), and ``content`` is available via extract if needed.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("AnySearch search: '%s' (limit=%d)", query, limit)
            raw = _search_request(query, limit=limit)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("AnySearch search error: %s", exc)
            return {"success": False, "error": f"AnySearch search failed: {exc}"}

        # Normalize response
        results = []
        for i, result in enumerate(raw.get("data", {}).get("results", [])):
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("snippet", ""),
                    "position": i + 1,
                }
            )

        return {"success": True, "data": {"web": results}}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via AnySearch.

        Uses AnySearch's MCP ``extract`` tool with fallback to direct
        HTTP fetch. Returns the legacy list-of-results shape; per-URL
        failures become items with ``error``.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("AnySearch extract: %d URL(s)", len(urls))
        except Exception:
            pass

        documents: List[Dict[str, Any]] = []

        for url in urls:
            try:
                content = _mcp_extract_request(url)
                documents.append(
                    {
                        "url": url,
                        "title": "",
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url},
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("AnySearch extract failed for %s: %s", url, exc)
                documents.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": str(exc),
                        "metadata": {"sourceURL": url},
                    }
                )

        return documents

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "AnySearch",
            "badge": "free · agent-native",
            "tag": (
                "Purpose-built for AI agents with 17 vertical domains "
                "(finance, academic, code, …). One-call search+content. "
                "1,000 free requests/day. Set ANYSEARCH_API_KEY."
            ),
            "env_vars": [
                {
                    "key": "ANYSEARCH_API_KEY",
                    "prompt": "AnySearch API key",
                    "url": "https://www.anysearch.com/console/api-keys",
                },
            ],
        }
