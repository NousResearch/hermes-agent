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

    PERPLEXITY_API_KEY=...       # https://www.perplexity.ai/account/api (required)
    PERPLEXITY_BASE_URL=...      # optional override of https://api.perplexity.ai

Keyed only — Perplexity has no anonymous tier, so this provider is not a
member of the zero-config keyless ring and never resolves without a key.

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

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.perplexity.ai"
_KEY_URL = "https://www.perplexity.ai/account/api"

# Search API hard cap for search_type=web.
_MAX_SEARCH_RESULTS = 20
# Snippet budgets (backend limits: max_tokens 1-16384, per page 1-4096).
_MAX_TOKENS = 16384
_MAX_TOKENS_PER_PAGE = 4096


def _missing_key_error() -> str:
    return f"PERPLEXITY_API_KEY is not set. Get a key at {_KEY_URL}"


def _perplexity_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the Perplexity API and return the parsed JSON response.

    Raises ``ValueError`` when the key is missing or on any non-2xx status,
    carrying the response body so Perplexity's own error text (invalid key,
    BAD_REQUEST, rate limit) reaches the model verbatim.
    """
    from agent.web_search_provider import get_provider_env

    api_key = get_provider_env("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError(_missing_key_error())
    base_url = (get_provider_env("PERPLEXITY_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Perplexity %s request to %s", endpoint, url)

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
    """Perplexity Search API (search) + content snippets (extract), keyed only."""

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity"

    def is_available(self) -> bool:
        """Return True when ``PERPLEXITY_API_KEY`` is set to a non-empty value."""
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("PERPLEXITY_API_KEY"))

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

            logger.info("Perplexity search: '%s' (limit=%d)", query, limit)
            raw = _perplexity_request(
                "search",
                {
                    "query": query,
                    "max_results": max(1, min(limit, _MAX_SEARCH_RESULTS)),
                    "search_context_size": "low",
                },
            )
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
