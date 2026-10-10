"""Tavily web search + content extraction (``/search``, ``/extract``; sync httpx).

Env: ``TAVILY_API_KEY`` (https://app.tavily.com/home, optional), ``TAVILY_BASE_URL``.
Keyed requests use ``Authorization: Bearer``; without a key the request is
keyless (``X-Tavily-Access-Mode: keyless``). Tavily is NOT in the zero-config
keyless ring — keyless access is opt-in by selecting Tavily in ``hermes tools``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from agent.web_acquisition_errors import WebCredentialsMissingError, WebInvalidResponseError, http_failure

from plugins.web._common import (
    SEARCH_LIMIT_CAP, BaseWebSearchProvider, document, extract_fail, provider_env, run_extract,
    run_search, search_fail, search_ok, setup_schema, title_hit, use_keyless,
)

logger = logging.getLogger(__name__)

_CLIENT_NAME = "hermes-agent"

_SEARCH_PAYLOAD = {"include_raw_content": False, "include_images": False}


class _TavilySearchHit(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    url: str = Field(min_length=1)
    title: str = ""
    content: str = ""


class _TavilySearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    results: list[_TavilySearchHit]


def _tavily_headers(api_key: str) -> dict[str, str]:
    headers = {"X-Client-Name": _CLIENT_NAME}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
    return headers


def _tavily_request(endpoint: str, payload: dict[str, Any], *, api_key: Optional[str] = None) -> dict[str, Any]:
    """POST to Tavily and return parsed JSON. ``api_key=None`` reads ``TAVILY_API_KEY``;
    pass ``""`` to force the keyless header even when a key exists
    (``web.provider_tier.tavily: free``). Non-2xx responses raise an acquisition
    error with status and retry-delay facts."""
    if api_key is None:
        api_key = provider_env("TAVILY_API_KEY")
    base_url = provider_env("TAVILY_BASE_URL") or "https://api.tavily.com"
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)
    response = httpx.post(url, json=payload, timeout=60, headers=_tavily_headers(api_key))
    if response.status_code >= 400:
        raise http_failure(response.status_code, response.headers)
    return response.json()


def _normalize_tavily_search_results(response: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = _TavilySearchResponse.model_validate(response)
    except ValidationError as exc:
        raise WebInvalidResponseError(cause=exc) from exc
    return search_ok([
        title_hit(r.title, r.url, r.content, i + 1)
        for i, r in enumerate(parsed.results)
    ])


class _TavilyDocument(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    url: str = Field(min_length=1)
    title: str | None = None
    raw_content: str | None = None
    content: str | None = None


class _TavilyFailedDocument(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    url: str = Field(min_length=1)
    error: str = "extraction failed"


class _TavilyExtractResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    results: list[_TavilyDocument]
    failed_results: list[_TavilyFailedDocument] = Field(default_factory=list)
    failed_urls: list[str] = Field(default_factory=list)


def _normalize_tavily_documents(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate ``/extract`` before converting content and per-page diagnostics."""
    try:
        parsed = _TavilyExtractResponse.model_validate(response)
    except ValidationError as exc:
        raise WebInvalidResponseError(cause=exc) from exc
    documents = [
        document(row.url, row.title or "", row.raw_content or row.content or "",
                 coverage="full" if row.raw_content is not None else "unknown")
        for row in parsed.results
    ]
    documents += [_failed_document(row.url, row.error) for row in parsed.failed_results]
    documents += [_failed_document(url, "extraction failed") for url in parsed.failed_urls]
    return documents


def _failed_document(url: str, error: str) -> dict[str, Any]:
    return {"url": url, "title": "", "content": "", "raw_content": "", "error": error, "metadata": {"sourceURL": url}}


def _missing_key_error(action: str) -> str:
    return f"TAVILY_API_KEY is not set. Get a key at https://app.tavily.com/home or select Tavily in `hermes tools` for opt-in keyless {action}."


def _auth(action: str) -> tuple[Optional[str], Optional[str], str]:
    """``(request_key, missing_key_error, log_prefix)``: request key is ``""`` when forcing
    keyless, ``None`` when neither key nor keyless applies (``missing_key_error`` set)."""
    api_key = provider_env("TAVILY_API_KEY")
    force_keyless = use_keyless("tavily", api_key)
    if not force_keyless and not api_key:
        return None, _missing_key_error(action), ""
    return "" if force_keyless else api_key, None, "keyless " if force_keyless else ""


class TavilyWebSearchProvider(BaseWebSearchProvider):
    """Tavily search + extract provider (keyed, or opt-in keyless)."""

    NAME = "tavily"
    DISPLAY_NAME = "Tavily"
    KEY_ENV = "TAVILY_API_KEY"
    EXTRACT = True
    KEYLESS = True

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            key, missing, prefix = _auth("search")
            if missing:
                return search_fail(missing, failure=WebCredentialsMissingError("TAVILY_API_KEY"))
            logger.info("Tavily %ssearch: '%s' (limit=%d)", prefix, query, limit)
            payload = {"query": query, "max_results": min(limit, SEARCH_LIMIT_CAP), **_SEARCH_PAYLOAD}
            return _normalize_tavily_search_results(_tavily_request("search", payload, api_key=key))

        return run_search("Tavily", logger, _body)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        def _body() -> list[dict[str, Any]]:
            key, missing, prefix = _auth("extract")
            if missing:
                return extract_fail(urls, missing, failure=WebCredentialsMissingError("TAVILY_API_KEY"))
            logger.info("Tavily %sextract: %d URL(s)", prefix, len(urls))
            raw = _tavily_request("extract", {"urls": urls, "include_images": False}, api_key=key)
            return _normalize_tavily_documents(raw)

        return run_extract("Tavily", logger, urls, _body)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "Tavily", "free · key optional", "Search + extract. Opt-in keyless; set TAVILY_API_KEY for higher limits.",
            "TAVILY_API_KEY", "Tavily API key (optional — keyless works when Tavily is selected)", "https://app.tavily.com/home",
        )
