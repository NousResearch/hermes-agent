"""Ollama Web Search — search + page fetch via ollama.com's REST APIs.

Implements Ollama's web search API (https://docs.ollama.com/capabilities/web-search):

- ``supports_search()``  -> True  (``POST https://ollama.com/api/web_search`` —
  ``{"query", "max_results"}`` → ``{"results": [{"title", "url", "content"}]}``;
  ``max_results`` caps at 10 server-side)
- ``supports_extract()`` -> True  (``POST https://ollama.com/api/web_fetch`` —
  ``{"url"}`` → ``{"title", "content", "links"}``; one URL per request, fanned
  out on a small thread pool — the extract dispatcher runs sync providers in a
  worker thread and bounds the whole batch with ``web.extract_timeout``)

Config: ``web.search_backend`` / ``web.extract_backend`` / ``web.backend: "ollama"``.
Env: ``OLLAMA_API_KEY`` (https://ollama.com/settings/keys — free Ollama account).
The key is shared with Ollama Cloud inference, so like xai this backend is not a
dedicated web credential; it activates only via an explicit ``web.*_backend``
selection and is deliberately absent from the autodetect credential cascade.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import httpx

from plugins.web._common import (
    BaseWebSearchProvider,
    document,
    page_error,
    provider_env,
    run_extract,
    run_search,
    search_fail,
    search_ok,
    setup_schema,
    titled_rows,
)
from hermes_cli import __version__ as _HERMES_VERSION

logger = logging.getLogger(__name__)

_SEARCH_ENDPOINT = "https://ollama.com/api/web_search"
_FETCH_ENDPOINT = "https://ollama.com/api/web_fetch"

# Server-side cap on web_search's max_results.
_MAX_SEARCH_RESULTS = 10
# Per-request HTTP timeout; the extract dispatcher applies its own batch wall-clock cap.
_REQUEST_TIMEOUT_S = 60
# Concurrent web_fetch POSTs within one extract() batch (the tool caps URLs at 5).
_MAX_CONCURRENT_FETCHES = 4
# Links harvested from web_fetch, appended to the page content for navigation.
_MAX_FETCH_LINKS = 25


def _missing_key_error() -> str:
    return "OLLAMA_API_KEY is not set. Get a key at https://ollama.com/settings/keys (free Ollama account)"


def _ollama_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to an ollama.com web API and return the parsed JSON response.

    Raises ``ValueError`` when the key is missing or on any non-2xx status,
    carrying Ollama's own error text (e.g. ``{"error": "Unauthorized"}``)
    so it reaches the model verbatim.
    """
    api_key = provider_env("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError(_missing_key_error())
    response = httpx.post(
        endpoint,
        json=payload,
        timeout=_REQUEST_TIMEOUT_S,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
        },
    )
    if response.status_code >= 400:
        body = (response.text or "").strip()
        try:
            detail = json.loads(body).get("error") if body else None
        except Exception:  # noqa: BLE001 — any body shape is fine to surface raw
            detail = None
        raise ValueError(detail or body or f"HTTP {response.status_code}")
    return response.json()


class OllamaWebSearchProvider(BaseWebSearchProvider):
    """Ollama web search (search) + web fetch (extract), keyed only."""

    NAME = "ollama"
    DISPLAY_NAME = "Ollama Web Search"
    KEY_ENV = "OLLAMA_API_KEY"
    EXTRACT = True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search via ollama.com's web_search API; ``content`` becomes the row description."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return search_fail("Interrupted")
            logger.info("Ollama web search: '%s' (limit=%d)", query, limit)
            raw = _ollama_request(
                _SEARCH_ENDPOINT,
                {
                    "query": query,
                    "max_results": max(1, min(int(limit), _MAX_SEARCH_RESULTS)),
                },
            )
            return search_ok(titled_rows(raw.get("results") or [], "content"))
        except ValueError as exc:
            return search_fail(str(exc))
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Ollama web search error: %s", exc)
            return search_fail(f"Ollama web search failed: {exc}")

    def _fetch_one(self, url: str) -> Dict[str, Any]:
        """One web_fetch POST → an extract document; failures become per-URL errors."""
        try:
            raw = _ollama_request(_FETCH_ENDPOINT, {"url": url})
        except ValueError as exc:
            return page_error(url, str(exc))
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Ollama web fetch error for %s: %s", url, exc)
            return page_error(url, f"Ollama web fetch failed: {exc}")
        content = str(raw.get("content") or "")
        if not content:
            return page_error(url, "no content returned")
        links = [str(link) for link in (raw.get("links") or []) if link][: _MAX_FETCH_LINKS]
        if links:
            content = content + "\n\nLinks:\n" + "\n".join(f"- {link}" for link in links)
        return document(url, str(raw.get("title") or ""), content)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Fetch page content for one or more URLs via ollama.com's web_fetch API.

        Sync — one POST per URL, fanned out concurrently. Per-URL failures
        become entries with ``error``; a missing key errors every URL.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [page_error(u, "Interrupted") for u in urls]
        except Exception:  # noqa: BLE001 — interrupt check is best-effort in scripts
            pass
        logger.info("Ollama web fetch: %d URL(s)", len(urls))
        return run_extract("Ollama", logger, list(urls), lambda: self._fetch_all(list(urls)))

    def _fetch_all(self, urls: List[str]) -> List[Dict[str, Any]]:
        if len(urls) == 1:
            return [self._fetch_one(urls[0])]
        with ThreadPoolExecutor(max_workers=min(len(urls), _MAX_CONCURRENT_FETCHES)) as pool:
            return list(pool.map(self._fetch_one, urls))

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "Ollama Web Search", "free",
            "Ollama's web search + page fetch APIs — requires a free Ollama account.",
            "OLLAMA_API_KEY", "Ollama API key", "https://ollama.com/settings/keys",
        )