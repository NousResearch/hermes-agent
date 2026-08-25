"""Tavily web search + content extraction (``/search``, ``/extract``; sync httpx).

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True (Tavily ``/search``)
- ``supports_extract()`` -> True (Tavily ``/extract``)

Both are sync — the underlying call is ``httpx.post(...)``.

Config keys this provider responds to::

    web:
      search_backend: "tavily"     # explicit per-capability
      extract_backend: "tavily"    # explicit per-capability
      backend: "tavily"            # shared fallback for both

Env vars::

    TAVILY_API_KEY=...           # https://app.tavily.com/home (optional)
    TAVILY_BASE_URL=...          # optional override of https://api.tavily.com

Auth is header-based. A key uses ``Authorization: Bearer``; without a key
the request is keyless (``X-Tavily-Access-Mode: keyless``). Both paths
send ``X-Client-Name: hermes-agent``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_CLIENT_NAME = "hermes-agent"


def _tavily_headers(api_key: str) -> Dict[str, str]:
    """Build Tavily request headers for keyed or keyless access."""
    headers = {"X-Client-Name": _CLIENT_NAME}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
    return headers


_SEARCH_PAYLOAD = {"include_raw_content": False, "include_images": False}

    Keyed when ``TAVILY_API_KEY`` is set (Bearer auth); otherwise keyless.
    Non-2xx responses raise ``ValueError`` with the response body so Tavily's
    keyless rate-limit / upgrade text reaches the model.
    """
    from agent.web_search_provider import get_provider_env

    api_key = get_provider_env("TAVILY_API_KEY")
    base_url = get_provider_env("TAVILY_BASE_URL") or "https://api.tavily.com"
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)

    response = httpx.post(
        url,
        json=payload,
        timeout=60,
        headers=_tavily_headers(api_key),
    )
    if response.status_code >= 400:
        body = (response.text or "").strip()
        detail = body or f"HTTP {response.status_code}"
        raise ValueError(detail)
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

    def is_keyless_available(self) -> bool:
        """Tavily serves anonymous keyless requests (X-Tavily-Access-Mode).

        Default-on ring member of the keyless free tier: fresh installs
        rotate across Exa/Parallel/Tavily/Firecrawl/Keenable. False when
        the user pinned ``web.provider_tier.tavily: paid`` — an explicit
        paid selection opts the free endpoint out.
        """
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return keyless_enabled() and provider_tier("tavily") != "paid"

    def supports_search(self) -> bool:
        return True

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

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import search_with_failover, use_keyless

            if use_keyless("tavily", get_provider_env("TAVILY_API_KEY")):
                # Keyless free tier — ring dispatch with next-in-line
                # failover on rate limits.
                logger.info(
                    "Tavily keyless search: '%s' (limit=%d)", query, limit
                )
                return search_with_failover("tavily", query, limit)

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
        def _body() -> List[Dict[str, Any]]:
            key, missing, prefix = _auth("extract")
            if missing:
                return extract_fail(urls, missing)
            logger.info("Tavily %sextract: %d URL(s)", prefix, len(urls))
            raw = _tavily_request("extract", {"urls": urls, "include_images": False}, api_key=key)
            return _normalize_tavily_documents(raw, fallback_url=urls[0] if urls else "")

        Sync — the underlying call is httpx.post(...). Returns the legacy
        list-of-results shape; per-URL failures become items with ``error``.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import extract_with_failover, use_keyless

            if use_keyless("tavily", get_provider_env("TAVILY_API_KEY")):
                # Keyless free tier — ring dispatch with next-in-line
                # failover on rate limits.
                logger.info("Tavily keyless extract: %d URL(s)", len(urls))
                return extract_with_failover("tavily", list(urls))

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

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Tavily",
            "badge": "free · key optional",
            "tag": "Search + extract. Works keyless; set TAVILY_API_KEY for higher limits.",
            "env_vars": [
                {
                    "key": "TAVILY_API_KEY",
                    "prompt": "Tavily API key (optional — keyless works without it)",
                    "url": "https://app.tavily.com/home",
                },
            ],
        }
