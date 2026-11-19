"""Helve web search and content extraction (``/v1/search``, ``/v1/extract``; sync httpx).

Env: ``HELVE_API_KEY`` (https://helve.dev),
optional ``HELVE_BASE_URL`` (default ``https://helve.dev``),
optional ``HELVE_SEARCH_PROVIDER`` (default ``auto``; supports ``fusion``, ``exa``, ``tavily``,
``brave``, ``serper``, ``firecrawl``, ``parallel``, ``perplexity``),
optional ``HELVE_EXTRACT_PROVIDER`` (default ``auto``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from plugins.web._common import (
    SEARCH_LIMIT_CAP,
    BaseWebSearchProvider,
    document,
    http_status_detail,
    page_error,
    provider_env,
    run_extract,
    run_search,
    search_ok,
    setup_schema,
    web_hit,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://helve.dev"
_KEY_URL = "https://helve.dev"


def _missing_key_error() -> str:
    return f"HELVE_API_KEY is not set. Get a key at {_KEY_URL}"


def _helve_request(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to Helve and return parsed JSON. Raises ValueError on missing key or HTTP >= 400."""
    api_key = provider_env("HELVE_API_KEY")
    if not api_key:
        raise ValueError(_missing_key_error())
    base_url = (provider_env("HELVE_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Helve %s request to %s", endpoint, url)
    response = httpx.post(
        url,
        json=payload,
        timeout=60,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    if response.status_code >= 400:
        raise ValueError(http_status_detail(response))
    return response.json()


def _normalize_search_results(response: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for i, r in enumerate(response.get("results") or []):
        if not isinstance(r, dict):
            continue
        snippets = r.get("snippets")
        snippet_text = "\n".join(s for s in snippets if isinstance(s, str) and s) if isinstance(snippets, list) else ""
        description = snippet_text or r.get("summary") or r.get("full_text") or ""
        rows.append(web_hit(str(r.get("url") or ""), str(r.get("title") or ""), str(description), i + 1))
    return search_ok(rows)


def _normalize_extract_documents(response: dict[str, Any], urls: list[str]) -> list[dict[str, Any]]:
    raw_results = [r for r in (response.get("results") or []) if isinstance(r, dict)]
    by_url = {r.get("url") or r.get("final_url") or "": r for r in raw_results}
    documents: list[dict[str, Any]] = []
    for url in urls:
        entry = by_url.get(url)
        if not entry:
            documents.append(page_error(url, "no content returned"))
            continue
        excerpts = entry.get("excerpts")
        excerpt_text = "\n\n".join(e for e in excerpts if isinstance(e, str) and e) if isinstance(excerpts, list) else ""
        content = entry.get("content") or excerpt_text or entry.get("summary") or ""
        if not content:
            documents.append(page_error(url, "no content returned"))
            continue
        documents.append(
            document(
                str(entry.get("url") or url),
                str(entry.get("title") or ""),
                str(content),
                source_url=str(entry.get("final_url") or url),
            )
        )
    return documents


class HelveWebSearchProvider(BaseWebSearchProvider):
    """Helve web search and content extraction provider."""

    NAME = "helve"
    DISPLAY_NAME = "Helve"
    KEY_ENV = "HELVE_API_KEY"
    EXTRACT = True
    KEYLESS = False

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            provider = (provider_env("HELVE_SEARCH_PROVIDER") or "auto").strip() or "auto"
            payload = {
                "query": query,
                "provider": provider,
                "max_results": max(1, min(limit, SEARCH_LIMIT_CAP)),
            }
            raw = _helve_request("v1/search", payload)
            return _normalize_search_results(raw)

        return run_search("Helve", logger, _body)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        def _body() -> list[dict[str, Any]]:
            provider = (provider_env("HELVE_EXTRACT_PROVIDER") or "auto").strip() or "auto"
            payload = {
                "urls": list(urls),
                "provider": provider,
            }
            raw = _helve_request("v1/extract", payload)
            return _normalize_extract_documents(raw, list(urls))

        return run_extract("Helve", logger, urls, _body)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "Helve",
            "paid",
            "Web search and page extraction via Helve (/v1/search and /v1/extract).",
            "HELVE_API_KEY",
            "Helve API key",
            _KEY_URL,
            web_tier="paid",
        )
