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

from plugins.web._common import (
    SEARCH_LIMIT_CAP, BaseWebSearchProvider, document, extract_fail, http_status_detail, provider_env, run_extract,
    run_search, search_fail, search_ok, setup_schema, title_hit, use_keyless,
)

logger = logging.getLogger(__name__)

_CLIENT_NAME = "hermes-agent"

_SEARCH_PAYLOAD = {"include_raw_content": False, "include_images": False}


def _tavily_headers(api_key: str) -> Dict[str, str]:
    headers = {"X-Client-Name": _CLIENT_NAME}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
    return headers


def _tavily_request(endpoint: str, payload: Dict[str, Any], *, api_key: Optional[str] = None) -> Dict[str, Any]:
    """POST to Tavily and return parsed JSON. ``api_key=None`` reads ``TAVILY_API_KEY``;
    pass ``""`` to force the keyless header even when a key exists
    (``web.provider_tier.tavily: free``). Non-2xx raises ValueError with the body so
    Tavily's rate-limit/upgrade text reaches the model."""
    if api_key is None:
        api_key = provider_env("TAVILY_API_KEY")
    base_url = provider_env("TAVILY_BASE_URL") or "https://api.tavily.com"
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)
    response = httpx.post(url, json=payload, timeout=60, headers=_tavily_headers(api_key))
    if response.status_code >= 400:
        raise ValueError(http_status_detail(response))
    return response.json()


def _normalize_tavily_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    return search_ok([
        title_hit(r.get("title", ""), r.get("url", ""), r.get("content", ""), i + 1)
        for i, r in enumerate(response.get("results", []))
    ])


def _normalize_tavily_documents(response: Dict[str, Any], fallback_url: str = "") -> List[Dict[str, Any]]:
    """Map ``/extract`` to documents; ``failed_results`` / ``failed_urls`` become ``error`` entries."""
    documents = [
        document(r.get("url", fallback_url), r.get("title", ""), r.get("raw_content", "") or r.get("content", ""))
        for r in response.get("results", [])
    ]
    documents += [_failed_document(f.get("url", fallback_url), f.get("error", "extraction failed")) for f in response.get("failed_results", [])]
    documents += [_failed_document(str(u), "extraction failed") for u in response.get("failed_urls", [])]
    return documents


def _failed_document(url: str, error: str) -> Dict[str, Any]:
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

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        def _body() -> Dict[str, Any]:
            key, missing, prefix = _auth("search")
            if missing:
                return search_fail(missing)
            logger.info("Tavily %ssearch: '%s' (limit=%d)", prefix, query, limit)
            payload = {"query": query, "max_results": min(limit, SEARCH_LIMIT_CAP), **_SEARCH_PAYLOAD}
            return _normalize_tavily_search_results(_tavily_request("search", payload, api_key=key))

        return run_search("Tavily", logger, _body)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        def _body() -> List[Dict[str, Any]]:
            key, missing, prefix = _auth("extract")
            if missing:
                return extract_fail(urls, missing)
            logger.info("Tavily %sextract: %d URL(s)", prefix, len(urls))
            raw = _tavily_request("extract", {"urls": urls, "include_images": False}, api_key=key)
            return _normalize_tavily_documents(raw, fallback_url=urls[0] if urls else "")

        return run_extract("Tavily", logger, urls, _body)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "Tavily", "free · key optional", "Search + extract. Opt-in keyless; set TAVILY_API_KEY for higher limits.",
            "TAVILY_API_KEY", "Tavily API key (optional — keyless works when Tavily is selected)", "https://app.tavily.com/home",
        )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'WebSearchProvider': ('agent.web_search_provider', 'WebSearchProvider'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
