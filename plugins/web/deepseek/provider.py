"""DeepSeek web search + content fetch (``/v1/tools/search``, ``/v1/tools/fetch``; sync httpx).

Env: ``DEEPSEEK_API_KEY`` (https://platform.deepseek.com/api_keys — shared with the
DeepSeek chat API), ``DEEPSEEK_SEARCH_BASE_URL``.

Both calls are raw JSON APIs billed **per successful request** (¥5 / 1,000 as of the
2026-10 launch), independent of model tokens. ``search`` returns DeepSeek's own
index (not an LLM's recollection): the row description prefers the API's long
``summary`` field over the display-oriented ``snippet``, since the former is written
to be read by a model.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from plugins.web._common import (
    BaseWebSearchProvider, document, http_status_detail, page_error, provider_env, run_extract,
    run_search, search_fail, search_ok, setup_schema, title_hit,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_MAX_COUNT = 50          # API cap (Hermes' own web_search limit is lower)
_MAX_QUERY_CHARS = 128   # API cap
_EXTRACT_WORKERS = 4


def _deepseek_request(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to a DeepSeek tools endpoint and return parsed JSON.

    Non-2xx raises ValueError carrying the response body, so a quota/auth failure
    reaches the model verbatim instead of as a generic transport error.
    """
    api_key = provider_env("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set — it is shared with the DeepSeek chat API "
            "(https://platform.deepseek.com/api_keys)."
        )
    base_url = (provider_env("DEEPSEEK_SEARCH_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    response = httpx.post(
        f"{base_url}{path}",
        json=payload,
        timeout=60,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    if response.status_code >= 400:
        raise ValueError(f"DeepSeek {path} returned HTTP {response.status_code}: {http_status_detail(response)}")
    return response.json()


def _normalize_deepseek_search_results(response: dict[str, Any]) -> dict[str, Any]:
    """Map ``results`` to the legacy row contract; ``summary`` beats ``snippet`` as description."""
    return search_ok([
        title_hit(
            str(r.get("title", "")),
            str(r.get("url", "")),
            str(r.get("summary") or r.get("snippet") or ""),
            i + 1,
        )
        for i, r in enumerate(response.get("results") or [])
    ])


class DeepSeekWebSearchProvider(BaseWebSearchProvider):
    """DeepSeek search + fetch provider (keyed; no keyless tier)."""

    NAME = "deepseek"
    DISPLAY_NAME = "DeepSeek Search"
    KEY_ENV = "DEEPSEEK_API_KEY"
    EXTRACT = True

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            q = (query or "").strip()[:_MAX_QUERY_CHARS]
            if not q:
                return search_fail("DeepSeek search: query must not be empty.")
            logger.info("DeepSeek search: '%s' (limit=%d)", q, limit)
            payload = {"query": q, "count": max(1, min(int(limit or 5), _MAX_COUNT)), "need_content": False}
            return _normalize_deepseek_search_results(_deepseek_request("/v1/tools/search", payload))

        return run_search("DeepSeek", logger, _body)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch is one URL per request; run a few concurrently and keep the caller's order."""

        def _one(url: str) -> dict[str, Any]:
            try:
                raw = _deepseek_request("/v1/tools/fetch", {"url": url})
            except ValueError as exc:
                # Fetch is one request per URL — a bad page/quota hit fails that URL only.
                return page_error(url, str(exc))
            content = raw.get("content") or ""
            if not content.strip():
                return page_error(url, "DeepSeek fetch returned empty content")
            return document(url, str(raw.get("title") or ""), content)

        def _body() -> list[dict[str, Any]]:
            logger.info("DeepSeek fetch: %d URL(s)", len(urls))
            if len(urls) == 1:
                return [_one(urls[0])]
            with ThreadPoolExecutor(max_workers=min(_EXTRACT_WORKERS, len(urls))) as pool:
                return list(pool.map(_one, urls))

        return run_extract("DeepSeek", logger, urls, _body)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "DeepSeek Search", "paid · ¥5 / 1k req",
            "Search + fetch via DeepSeek's in-house engine. Billed per successful request; shares DEEPSEEK_API_KEY with the chat API.",
            "DEEPSEEK_API_KEY", "DeepSeek API key (shared with the chat API)", "https://platform.deepseek.com/api_keys",
        )
