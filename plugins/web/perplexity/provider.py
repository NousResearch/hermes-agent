"""Perplexity web search + page snippets — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True (Perplexity Search API ``POST /search``)
- ``supports_extract()`` -> True (``POST /sdk/content/snippets`` — the
  query-relevant page-excerpt route behind ``pplx content snippets``)

Both are sync — the underlying call is ``httpx.post(...)``.

Config keys this provider responds to::

    web:
      search_backend: "perplexity"   # explicit per-capability
      extract_backend: "perplexity"  # explicit per-capability
      backend: "perplexity"          # shared fallback for both

Env vars::

    PERPLEXITY_API_KEY=...       # required for direct search and extract
    PERPLEXITY_BASE_URL=...      # optional override of https://api.perplexity.ai

No anonymous tier. The Nous Subscription selection serves search through
``perplexity-gateway.<TOOL_GATEWAY_DOMAIN>`` using the Nous token; a direct
key takes precedence. Managed extract stays on Firecrawl.

Extract caveat: Perplexity's only supported page-content route returns the
passages of a page relevant to a *query* (elisions marked ``…``), not the
whole page. ``web_extract`` has no query, so the URL's own path words are
used as the relevance query, which approximates "what is this page about".
Use Firecrawl / Exa / Parallel when a verbatim full-page dump is required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

from agent.web_search_provider import WebSearchProvider
from hermes_cli.version_info import get_version_info

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.perplexity.ai"
_KEY_URL = "https://www.perplexity.ai/account/api"

# Identify Hermes to Perplexity: the same static harness identity Hermes sends Kimi and
# OpenCode, plus Perplexity's integration header. No per-user identifier and no separate
# request; the call already carries the user's own API key.
_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{get_version_info().base_version}",
    "X-Pplx-Integration": "hermes-agent",
}


# Search API hard cap for search_type=web.
_MAX_SEARCH_RESULTS = 20
# Snippet budgets (backend limits: max_tokens 1-16384, per page 1-4096).
_MAX_TOKENS = 16384
_MAX_TOKENS_PER_PAGE = 4096


def _missing_key_error() -> str:
    return f"PERPLEXITY_API_KEY is not set. Get a key at {_KEY_URL}"


def _managed_gateway(token_reader=None):
    """Nous Tool Gateway config when web_search is on the managed route, else None."""
    from tools import managed_tool_gateway as gw
    from tools.web_tools import _managed_web_search

    if not _managed_web_search():
        return None
    return gw.resolve_managed_tool_gateway("perplexity", token_reader=token_reader)


def _perplexity_request(endpoint: str, payload: Dict[str, Any], gateway=None) -> Dict[str, Any]:
    """POST to Perplexity or the supplied gateway; return parsed JSON.

    Raises ``ValueError`` when the key is missing or on any non-2xx status,
    carrying the response body so Perplexity's own error text (invalid key,
    BAD_REQUEST, rate limit) reaches the model verbatim.
    """
    from agent.web_search_provider import get_provider_env

    api_key = get_provider_env("PERPLEXITY_API_KEY")
    headers = _HEADERS
    if gateway is not None:
        # Nous-owned key behind the gateway: identify the harness only, not a per-user integration.
        base_url, api_key, headers = gateway.gateway_origin.rstrip("/"), gateway.nous_user_token, {"User-Agent": _HEADERS["User-Agent"]}
    elif api_key:
        base_url = (get_provider_env("PERPLEXITY_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    else:
        raise ValueError(_missing_key_error())
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Perplexity %s request to %s", endpoint, url)

    response = httpx.post(
        url,
        json=payload,
        timeout=60,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **headers,
        },
    )
    if response.status_code >= 400:
        body = (response.text or "").strip()
        raise ValueError(body or f"HTTP {response.status_code}")
    return response.json()


def _normalize_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map Search API ``{results: [{title,url,snippet,...}]}`` to the tool shape."""
    web_results = []
    for i, result in enumerate(response.get("results") or []):
        web_results.append(
            {
                "title": result.get("title", "") or "",
                "url": result.get("url", "") or "",
                "description": result.get("snippet", "") or "",
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def _normalize_snippets(response: Dict[str, Any], urls: List[str]) -> List[Dict[str, Any]]:
    """Map ``{results: [{url,text?,tokens_count?,error?}]}`` to extract documents.

    One document per requested URL, in request order. A URL the backend
    omitted or flagged with ``error`` becomes a document carrying ``error``
    rather than raising — a 200 does not mean every page succeeded.
    """
    by_url = {r.get("url", ""): r for r in (response.get("results") or []) if isinstance(r, dict)}
    documents: List[Dict[str, Any]] = []
    for url in urls:
        result = by_url.get(url, {})
        text = result.get("text") or ""
        doc: Dict[str, Any] = {
            "url": url,
            "title": "",
            "content": text,
            "raw_content": text,
            "metadata": {"sourceURL": url},
        }
        error = result.get("error")
        if error or not text:
            doc["error"] = str(error) if error else "no content returned"
        documents.append(doc)
    return documents


def _query_for_urls(urls: List[str]) -> str:
    """Derive a relevance query from URL path words (``/bloom-filter`` -> ``bloom filter``)."""
    words: List[str] = []
    for url in urls:
        parsed = urlparse(url)
        for token in parsed.path.replace("-", " ").replace("_", " ").replace("/", " ").split():
            if token.lower() not in words and not token.isdigit():
                words.append(token.lower())
        if not parsed.path.strip("/"):
            words.append(parsed.netloc)
    return " ".join(words)[:500] or " ".join(urls)[:500]


class PerplexityWebSearchProvider(WebSearchProvider):
    """Direct or managed search; direct-key content snippets for extract."""

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity"

    def is_available(self) -> bool:
        """True with a ``PERPLEXITY_API_KEY``, or on the managed route with a likely-usable Nous token."""
        from agent.web_search_provider import get_provider_env
        from tools.managed_tool_gateway import peek_nous_access_token

        return bool(get_provider_env("PERPLEXITY_API_KEY")) or _managed_gateway(token_reader=peek_nous_access_token) is not None

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Perplexity Search API query.

        ``search_context_size: low`` keeps ``snippet`` at description length;
        the default (``high``) returns multi-KB page excerpts per hit, which
        belongs in ``web_extract`` rather than a results list.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            from agent.web_search_provider import get_provider_env
            from tools.web_tools import _managed_web_search

            direct = bool(get_provider_env("PERPLEXITY_API_KEY"))
            gateway = None if direct else _managed_gateway()
            if gateway is None and not direct and _managed_web_search():
                from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, selection_error
                raise ValueError(selection_error(
                    "web", NOUS_MANAGED_PROVIDER, "the Nous Tool Gateway is not available (not entitled or unreachable)"))
            logger.info("Perplexity search: '%s' (limit=%d%s)", query, limit, ", managed" if gateway else "")
            payload = {
                "query": query,
                "max_results": max(1, min(limit, _MAX_SEARCH_RESULTS)),
                "search_context_size": "low",
            }
            if gateway is not None:
                payload["search_type"] = "fast"
            raw = _perplexity_request("search", payload, gateway)
            return _normalize_search_results(raw)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Perplexity search error: %s", exc)
            return {"success": False, "error": f"Perplexity search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Return query-relevant snippets for one or more URLs.

        Sync — the underlying call is httpx.post(...). Per-URL failures
        become items with ``error``; a missing key errors every URL.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

            logger.info("Perplexity snippets: %d URL(s)", len(urls))
            raw = _perplexity_request(
                "sdk/content/snippets",
                {
                    "query": _query_for_urls(urls),
                    "urls": list(urls),
                    "max_tokens": _MAX_TOKENS,
                    "max_tokens_per_page": _MAX_TOKENS_PER_PAGE,
                },
            )
            return _normalize_snippets(raw, list(urls))
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Perplexity extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Perplexity extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Perplexity",
            "badge": "paid",
            "tag": (
                "Perplexity Search API — ranked, date-stamped web results plus "
                "query-relevant page snippets for extract."
            ),
            "env_vars": [
                {
                    "key": "PERPLEXITY_API_KEY",
                    "prompt": "Perplexity API key",
                    "url": _KEY_URL,
                },
            ],
            "web_tier": "paid",
        }
