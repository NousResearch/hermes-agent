"""Tavily web search + content extraction — plugin form.

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

Auth is header-based. A key uses ``Authorization: Bearer``; without a
key the request is keyless (``X-Tavily-Access-Mode: keyless``). Both
paths send ``X-Client-Name: hermes-agent``.

Tavily is **not** a member of the zero-config keyless ring
(``plugins.web.keyless_mcp._KEYLESS_RING``). Keyless access is opt-in:
select Tavily in ``hermes tools`` (or set ``web.backend: tavily``).
Fresh installs with no web credentials rotate across Exa / Parallel /
Firecrawl / Keenable instead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_CLIENT_NAME = "hermes-agent"

_SEARCH_PAYLOAD = {
    "include_raw_content": False,
    "include_images": False,
}


def _tavily_headers(api_key: str) -> Dict[str, str]:
    """Build Tavily request headers for keyed or keyless access."""
    headers = {"X-Client-Name": _CLIENT_NAME}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
    return headers


def _tavily_request(
    endpoint: str,
    payload: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """POST to the Tavily API and return the parsed JSON response.

    Keyed when *api_key* (or ``TAVILY_API_KEY``) is set (Bearer auth);
    otherwise keyless. Pass ``api_key=""`` to force the keyless header even
    when a key is present (``web.provider_tier.tavily: free``). Non-2xx
    responses raise ``ValueError`` with the response body so Tavily's
    keyless rate-limit / upgrade text reaches the model.
    """
    from agent.web_search_provider import get_provider_env

    if api_key is None:
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
    """Map Tavily ``/extract`` response to standard documents.

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


def _missing_key_error(action: str) -> str:
    return (
        f"TAVILY_API_KEY is not set. Get a key at https://app.tavily.com/home "
        f"or select Tavily in `hermes tools` for opt-in keyless {action}."
    )


class TavilyWebSearchProvider(WebSearchProvider):
    """Tavily search + extract provider (keyed, or opt-in keyless)."""

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def display_name(self) -> str:
        return "Tavily"

    def is_available(self) -> bool:
        """Return True when ``TAVILY_API_KEY`` is set to a non-empty value."""
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("TAVILY_API_KEY"))

    def is_keyless_available(self) -> bool:
        """Tavily serves anonymous keyless requests (X-Tavily-Access-Mode).

        Opt-in only — Tavily is not a member of the zero-config keyless
        ring. ``is_keyless_available`` is True so an explicit
        ``web.backend: tavily`` (or ``hermes tools`` pick) works without a
        key. False when the user pinned ``web.provider_tier.tavily: paid``.
        """
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return keyless_enabled() and provider_tier("tavily") != "paid"

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Tavily search (keyed path or opt-in keyless)."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import use_keyless

            api_key = get_provider_env("TAVILY_API_KEY")
            force_keyless = use_keyless("tavily", api_key)
            if not force_keyless and not api_key:
                return {"success": False, "error": _missing_key_error("search")}

            logger.info(
                "Tavily %ssearch: '%s' (limit=%d)",
                "keyless " if force_keyless else "",
                query,
                limit,
            )
            raw = _tavily_request(
                "search",
                {
                    "query": query,
                    "max_results": min(limit, 20),
                    **_SEARCH_PAYLOAD,
                },
                api_key="" if force_keyless else api_key,
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
        Keyless uses Tavily's own endpoint, not the keyless ring.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import use_keyless

            api_key = get_provider_env("TAVILY_API_KEY")
            force_keyless = use_keyless("tavily", api_key)
            if not force_keyless and not api_key:
                err = _missing_key_error("extract")
                return [
                    {"url": u, "title": "", "content": "", "error": err}
                    for u in urls
                ]

            logger.info(
                "Tavily %sextract: %d URL(s)",
                "keyless " if force_keyless else "",
                len(urls),
            )
            raw = _tavily_request(
                "extract",
                {
                    "urls": urls,
                    "include_images": False,
                },
                api_key="" if force_keyless else api_key,
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
            "tag": (
                "Search + extract. Opt-in keyless (not in the free-tier ring); "
                "set TAVILY_API_KEY for higher limits."
            ),
            "env_vars": [
                {
                    "key": "TAVILY_API_KEY",
                    "prompt": "Tavily API key (optional — keyless works when Tavily is selected)",
                    "url": "https://app.tavily.com/home",
                },
            ],
        }
