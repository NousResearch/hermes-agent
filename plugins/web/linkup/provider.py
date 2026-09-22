"""Linkup web search + page fetch (``/v1/search``, ``/v1/fetch``; sync httpx).

Env: ``LINKUP_API_KEY`` (https://app.linkup.so). Optional
``LINKUP_SEARCH_DEPTH`` = flash | fast (default) | standard | deep.
Optional ``LINKUP_BASE_URL``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from plugins.web._common import (
    SEARCH_LIMIT_CAP, BaseWebSearchProvider, document, extract_fail, http_status_detail, page_error, provider_env,
    run_extract, run_search, search_ok, setup_schema, title_hit,
)

logger = logging.getLogger(__name__)

_DEPTHS = {"flash", "fast", "standard", "deep"}
_MISSING_KEY = "LINKUP_API_KEY environment variable not set. Get your API key at https://app.linkup.so"


def _base_url() -> str:
    return (provider_env("LINKUP_BASE_URL") or "https://api.linkup.so").rstrip("/")


def _depth() -> str:
    depth = provider_env("LINKUP_SEARCH_DEPTH").lower()
    return depth if depth in _DEPTHS else "fast"


def _request(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = provider_env("LINKUP_API_KEY")
    if not api_key:
        raise ValueError(_MISSING_KEY)
    url = f"{_base_url()}/v1/{path.lstrip('/')}"
    logger.info("Linkup %s request to %s", path, url)
    response = httpx.post(
        url, json=payload, timeout=60,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    if response.status_code >= 400:
        raise ValueError(http_status_detail(response))
    data = response.json()
    return data if isinstance(data, dict) else {}


class LinkupWebSearchProvider(BaseWebSearchProvider):
    """Linkup search + fetch provider. Keyed only."""

    NAME = "linkup"
    DISPLAY_NAME = "Linkup"
    KEY_ENV = "LINKUP_API_KEY"
    EXTRACT = True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        def _body() -> Dict[str, Any]:
            depth = _depth()
            capped = max(1, min(int(limit), SEARCH_LIMIT_CAP))
            logger.info("Linkup search: '%s' (depth=%s, limit=%d)", query, depth, capped)
            response = _request("search", {
                "q": query, "depth": depth, "outputType": "searchResults", "maxResults": capped,
            })
            return search_ok([
                title_hit(r.get("name", "") or "", r.get("url", "") or "", r.get("content", "") or "", i + 1)
                for i, r in enumerate(response.get("results") or [])
                if isinstance(r, dict)
            ])

        return run_search("Linkup", logger, _body)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        def _body() -> List[Dict[str, Any]]:
            if not provider_env("LINKUP_API_KEY"):
                return extract_fail(urls, _MISSING_KEY)
            logger.info("Linkup extract: %d URL(s)", len(urls))
            documents: List[Dict[str, Any]] = []
            for url in urls:
                try:
                    raw = _request("fetch", {"url": url, "renderJs": True})
                    documents.append(document(url, "", raw.get("markdown") or ""))
                except Exception as exc:  # noqa: BLE001 — one bad URL must not drop the batch
                    documents.append(page_error(url, str(exc)))
            return documents

        return run_extract("Linkup", logger, urls, _body)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "Linkup", "paid",
            "Agent web search and page fetch as markdown. Requires LINKUP_API_KEY.",
            "LINKUP_API_KEY", "Linkup API key", "https://app.linkup.so",
        )
