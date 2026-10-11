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

No anonymous tier at Perplexity itself. The managed route serves search through
``perplexity-gateway.<TOOL_GATEWAY_DOMAIN>`` as ``search_type: "fast"``, which the
gateway serves to any Nous identity (paid, unpaid or anonymous guest); a direct key
takes precedence.
Managed extract stays on Firecrawl.

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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.web_search_provider import WebSearchProvider
from agent.web_acquisition import WebExtractCapabilities
from agent.web_acquisition_errors import WebAttributionInvalidError, WebCancelledError, WebCredentialsMissingError, WebInvalidResponseError, WebManagedGatewayUnavailableError, acquisition_error, http_failure
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
    return gw.resolve_free_search_gateway(token_reader=token_reader)


def _perplexity_request(endpoint: str, payload: dict[str, Any], gateway=None) -> dict[str, Any]:
    """POST to Perplexity or the supplied gateway; return parsed JSON.

    Missing credentials and non-2xx responses raise acquisition errors with
    safe facts for programmatic callers.
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
        raise WebCredentialsMissingError("PERPLEXITY_API_KEY")
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
        raise http_failure(response.status_code, response.headers)
    return response.json()


class _PerplexitySearchHit(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    url: str = Field(min_length=1)
    title: str | None = None
    snippet: str | None = None


class _PerplexitySearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    results: list[_PerplexitySearchHit]


def _normalize_search_results(response: dict[str, Any]) -> dict[str, Any]:
    """Map Search API ``{results: [{title,url,snippet,...}]}`` to the tool shape."""
    try:
        parsed = _PerplexitySearchResponse.model_validate(response)
    except ValidationError as exc:
        raise WebInvalidResponseError(cause=exc) from exc
    web_results = []
    for i, result in enumerate(parsed.results):
        web_results.append(
            {
                "title": result.title or "",
                "url": result.url,
                "description": result.snippet or "",
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


class _PerplexitySnippet(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    url: str = Field(min_length=1)
    text: str | None = None
    error: str | None = None


class _PerplexitySnippetsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    results: list[_PerplexitySnippet]


def _normalize_snippets(response: dict[str, Any], urls: list[str]) -> list[dict[str, Any]]:
    """Validate snippets without dropping malformed, duplicate or unrelated results."""
    try:
        parsed = _PerplexitySnippetsResponse.model_validate(response)
    except ValidationError as exc:
        raise WebInvalidResponseError(cause=exc) from exc
    by_url: dict[str, _PerplexitySnippet] = {}
    for row in parsed.results:
        if row.url not in urls or row.url in by_url:
            raise WebAttributionInvalidError()
        by_url[row.url] = row
    documents: list[dict[str, Any]] = []
    for url in urls:
        row = by_url.get(url)
        text = row.text or "" if row is not None else ""
        doc: dict[str, Any] = {"url": url, "title": "", "content": text, "raw_content": text,
                               "metadata": {"sourceURL": url}, "requested_url": url, "coverage": "partial"}
        if row is None or row.error or not text:
            doc["error"] = row.error if row is not None and row.error else "no content returned"
        documents.append(doc)
    return documents


def _query_for_urls(urls: list[str]) -> str:
    """Derive a relevance query from URL path words (``/bloom-filter`` -> ``bloom filter``)."""
    words: list[str] = []
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

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Execute a Perplexity Search API query.

        ``search_context_size: low`` keeps ``snippet`` at description length;
        the default (``high``) returns multi-KB page excerpts per hit, which
        belongs in ``web_extract`` rather than a results list.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted", "failure": WebCancelledError().to_failure()}

            from agent.web_search_provider import get_provider_env
            from tools.managed_tool_gateway import resolve_free_search_gateway
            from tools.web_tools import _managed_web_search

            direct = bool(get_provider_env("PERPLEXITY_API_KEY"))
            managed = not direct and _managed_web_search()
            gateway = resolve_free_search_gateway() if managed else None
            if gateway is None and managed:
                from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, selection_error

                return {
                    "success": False,
                    "error": selection_error(
                        "web", NOUS_MANAGED_PROVIDER,
                        "there is no usable Nous identity (sign in with `/login`)",
                    ),
                    "failure": WebManagedGatewayUnavailableError().to_failure(),
                }
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
            return {"success": False, "error": str(exc), "failure": acquisition_error(exc).to_failure()}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Perplexity search error: %s", exc)
            return {"success": False, "error": f"Perplexity search failed: {exc}", "failure": acquisition_error(exc).to_failure()}

    def extract_capabilities(self) -> WebExtractCapabilities:
        return WebExtractCapabilities(formats=("markdown",))

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Return query-relevant snippets for one or more URLs.

        Sync — the underlying call is httpx.post(...). Per-URL failures
        become items with ``error``; a missing key errors every URL.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": "", "failure": WebCancelledError().to_failure()} for u in urls]

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
            return [{"url": u, "title": "", "content": "", "error": str(exc), "failure": acquisition_error(exc).to_failure()} for u in urls]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Perplexity extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Perplexity extract failed: {exc}", "failure": acquisition_error(exc).to_failure()}
                for u in urls
            ]

    def get_setup_schema(self) -> dict[str, Any]:
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
