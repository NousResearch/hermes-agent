"""Oxylabs AI Studio web search + extract — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True  (AI Studio ``/search`` endpoints)
- ``supports_extract()`` -> True  (AI Studio ``/scrape`` endpoint)

Talks to the AI Studio REST API directly over ``httpx`` (a Hermes core
dependency) — no vendor SDK, no lazy ``pip install``. This matches the
dependency-free pattern of the other bundled providers that wrap a simple
REST surface (Tavily, xAI, Brave, SearXNG).

API surface (base ``https://api-aistudio.oxylabs.io``, override via
``OXYLABS_AI_STUDIO_API_URL``; auth header ``x-api-key``)::

    POST /search/instant     # limit<=10, content-free — returns data inline
    POST /search/run         # otherwise — returns {run_id}
    GET  /search/run/data     # poll: 202 = pending, 200 + status=completed
    POST /scrape             # returns {run_id}
    GET  /scrape/run/data     # poll: 202 = pending, 200 + status=completed

Config keys this provider responds to::

    web:
      search_backend: "oxylabs"     # explicit per-capability
      extract_backend: "oxylabs"    # explicit per-capability
      backend: "oxylabs"            # shared fallback for both

Env vars::

    OXYLABS_AI_STUDIO_API_KEY=...   # https://aistudio.oxylabs.io/api-key
    OXYLABS_AI_STUDIO_API_URL=...   # optional base-URL override (testing)

Forward-compat kwargs honored on ``extract``:

- ``render_javascript`` (bool)  — JS-render the page before extraction.
- ``geo_location`` (str)        — geo-target the request (ISO country code).
- ``format`` (str)              — ``"markdown"`` (default), ``"json"``,
                                  ``"html"`` (mapped to ``markdown`` — AI
                                  Studio does not return raw HTML).

Post-redirect URL re-check: the API does not expose the final URL after
redirects, so :func:`tools.website_policy.check_website_access` only runs
pre-flight against the URL the caller passed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from agent.web_search_provider import WebSearchProvider
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


_API_KEY_ENV = "OXYLABS_AI_STUDIO_API_KEY"
_API_URL_ENV = "OXYLABS_AI_STUDIO_API_URL"
_API_KEY_URL = "https://aistudio.oxylabs.io/api-key"
_DEFAULT_BASE_URL = "https://api-aistudio.oxylabs.io"

# Sent as User-Agent on every request so Oxylabs can attribute API traffic
# back to the Hermes integration.
_INTEGRATION_UA = "hermes-agent"

# Limit at/below which the faster content-free ``/search/instant`` endpoint
# is used (returns results inline, no polling). Mirrors the SDK's routing.
_INSTANT_SEARCH_MAX_LIMIT = 10

_POLL_INTERVAL_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 35.0
_SEARCH_DEADLINE_SECONDS = 90.0   # total budget for the polled search path
_EXTRACT_TIMEOUT_SECONDS = 60.0   # per-URL guard around submit + poll


def _read_api_key() -> str:
    """Return the configured API key, or raise ``ValueError`` if unset."""
    api_key = os.getenv(_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(
            f"{_API_KEY_ENV} environment variable not set. "
            f"Get your API key at {_API_KEY_URL}"
        )
    return api_key


def _base_url() -> str:
    """Return the API base URL, honoring the optional env override."""
    return (os.getenv(_API_URL_ENV, "").strip() or _DEFAULT_BASE_URL).rstrip("/")


def _request_headers(api_key: str) -> Dict[str, str]:
    return {"x-api-key": api_key, "User-Agent": _INTEGRATION_UA}


# ---------------------------------------------------------------------------
# Response shape normalization
# ---------------------------------------------------------------------------


def _scrape_data_to_content(data: Any) -> str:
    """Reduce a scrape job's ``data`` field to a string for ``content``."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        # Structured output_format — keep the payload as a stringified body
        # so downstream LLM post-processing has something to chew on. The
        # dict is also surfaced separately under ``metadata``.
        import json

        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(data)
    return str(data)


def _normalize_format_kwarg(format_value: Optional[str], default: str = "markdown") -> str:
    """Map dispatcher ``format`` values to AI Studio ``output_format`` literals.

    Hermes' dispatcher passes ``"markdown"`` / ``"html"``; AI Studio's
    scraper supports ``"markdown"``, ``"json"``, ``"csv"``, ``"screenshot"``,
    and ``"toon"``. Map ``"html"`` to ``"markdown"`` and pass through
    native values as-is.
    """
    if not format_value:
        return default
    if format_value == "html":
        return "markdown"
    if format_value in ("markdown", "json", "csv", "screenshot", "toon"):
        return format_value
    return default


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class OxylabsWebSearchProvider(WebSearchProvider):
    """Oxylabs AI Studio search + extract provider (REST, no SDK)."""

    # ------------------------------------------------------------------ ABC

    @property
    def name(self) -> str:
        return "oxylabs"

    @property
    def display_name(self) -> str:
        return "Oxylabs AI Studio"

    def is_available(self) -> bool:
        """Return True when ``OXYLABS_AI_STUDIO_API_KEY`` is set."""
        return bool(os.getenv(_API_KEY_ENV, "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # ------------------------------------------------------------------ search

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an Oxylabs AI search.

        Sync. Returns the legacy ``{"success": True, "data": {"web": [...]}}``
        envelope on success, ``{"success": False, "error": str}`` on in-flight
        failure.

        Pre-flight errors (``ValueError`` from a missing
        ``OXYLABS_AI_STUDIO_API_KEY``) propagate to the dispatcher's top-level
        handler, which wraps them as ``tool_error(...)`` — matching the legacy
        ``{"error": "Error searching web: ..."}`` envelope. Only in-flight
        errors are caught and surfaced as ``{"success": False, "error": ...}``.

        Returns listings only (``return_content=False``) — per-URL content
        belongs in :meth:`extract`.
        """
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return {"success": False, "error": "Interrupted"}

        logger.info("Oxylabs search: query=%r limit=%d return_content=False", query, limit)

        # _read_api_key() raises ValueError on unconfigured systems — let it
        # propagate so the dispatcher emits the legacy envelope shape
        # ({"error": "Error searching web: ..."}).
        api_key = _read_api_key()

        try:
            with httpx.Client(
                base_url=_base_url(),
                headers=_request_headers(api_key),
                timeout=_REQUEST_TIMEOUT_SECONDS,
            ) as client:
                if limit <= _INSTANT_SEARCH_MAX_LIMIT:
                    resp = client.post(
                        "/search/instant", json={"query": query, "limit": limit}
                    )
                    resp.raise_for_status()
                    raw_results = resp.json().get("data") or []
                else:
                    resp = client.post(
                        "/search/run",
                        json={"query": query, "limit": limit, "return_content": False},
                    )
                    resp.raise_for_status()
                    run_id = resp.json()["run_id"]
                    raw_results = self._poll_sync(client, "/search/run/data", run_id) or []

            web_results: List[Dict[str, Any]] = []
            for i, result in enumerate(raw_results):
                result_map = result if isinstance(result, dict) else {}
                web_results.append(
                    {
                        "title": result_map.get("title", "") or "",
                        "url": result_map.get("url", "") or "",
                        "description": result_map.get("description", "") or "",
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Oxylabs search error: %s", exc)
            return {"success": False, "error": f"Oxylabs search failed: {exc}"}

    def _poll_sync(self, client: httpx.Client, path: str, run_id: str) -> Any:
        """Poll a job until it completes; return its ``data``.

        202 means still running. A 200 with ``status == "completed"`` returns
        the data; ``status == "failed"`` raises. Bounded by
        ``_SEARCH_DEADLINE_SECONDS``.
        """
        deadline = time.monotonic() + _SEARCH_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            resp = client.get(path, params={"run_id": run_id})
            if resp.status_code != 202:
                resp.raise_for_status()
                body = resp.json()
                status = body.get("status")
                if status == "completed":
                    return body.get("data")
                if status == "failed":
                    raise RuntimeError(f"job failed: {body.get('error_code')}")
            time.sleep(_POLL_INTERVAL_SECONDS)
        raise TimeoutError("timed out waiting for Oxylabs search job")

    # ------------------------------------------------------------------ extract

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via the AI Studio scraper.

        Async. Each URL is scraped with a 60s ``asyncio.wait_for`` guard
        around the submit + poll round-trip. Per-URL failures (timeout, SSRF
        / policy block, API error) become items with an ``error`` field
        rather than raising.

        Recognized kwargs (others ignored for forward compat):
        - ``format``: see :func:`_normalize_format_kwarg`. Default markdown.
        - ``render_javascript``: bool (default False).
        - ``geo_location``: str (ISO country code).
        """
        from tools.interrupt import is_interrupted as _is_interrupted

        if _is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        output_format = _normalize_format_kwarg(kwargs.get("format"))
        render_javascript = bool(kwargs.get("render_javascript", False))
        geo_location = kwargs.get("geo_location")

        # Pre-flight API key check — fail the whole batch fast rather than
        # per-URL.
        try:
            api_key = _read_api_key()
        except ValueError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": str(exc)} for u in urls
            ]

        results: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(
            base_url=_base_url(),
            headers=_request_headers(api_key),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        ) as client:
            for url in urls:
                if _is_interrupted():
                    results.append({"url": url, "error": "Interrupted", "title": ""})
                    continue

                # Website-access policy gate. The API doesn't expose the
                # post-redirect URL, so we only gate on the input — see the
                # module docstring.
                blocked = check_website_access(url)
                if blocked:
                    logger.info(
                        "Blocked Oxylabs extract for %s by rule %s",
                        blocked["host"],
                        blocked["rule"],
                    )
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": blocked["message"],
                            "blocked_by_policy": {
                                "host": blocked["host"],
                                "rule": blocked["rule"],
                                "source": blocked["source"],
                            },
                        }
                    )
                    continue

                logger.info(
                    "Oxylabs scrape: url=%r output_format=%r "
                    "render_javascript=%s geo_location=%r",
                    url,
                    output_format,
                    render_javascript,
                    geo_location,
                )
                try:
                    data, run_id = await asyncio.wait_for(
                        self._scrape_one(
                            client, url, output_format, render_javascript, geo_location
                        ),
                        timeout=_EXTRACT_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Oxylabs scrape timed out for %s", url)
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": (
                                "Scrape timed out after 60s — page may be too "
                                "large or unresponsive. Try browser_navigate "
                                "instead."
                            ),
                        }
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Oxylabs scrape failed for %s: %s", url, exc)
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "error": str(exc),
                        }
                    )
                    continue

                content = _scrape_data_to_content(data)
                metadata: Dict[str, Any] = {"sourceURL": url}
                if isinstance(data, dict):
                    metadata["extracted"] = data
                if run_id:
                    metadata["oxylabs_run_id"] = run_id

                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": content,
                        "raw_content": content,
                        "metadata": metadata,
                    }
                )

        return results

    async def _scrape_one(
        self,
        client: httpx.AsyncClient,
        url: str,
        output_format: str,
        render_javascript: bool,
        geo_location: Optional[str],
    ) -> tuple[Any, Optional[str]]:
        """Submit one scrape job and poll it to completion.

        Returns ``(data, run_id)``. Raises on API / poll failure (caught by
        the per-URL handler in :meth:`extract`).
        """
        body: Dict[str, Any] = {
            "url": url,
            "output_format": output_format,
            "render_javascript": render_javascript,
        }
        if geo_location:
            body["geo_location"] = geo_location

        resp = await client.post("/scrape", json=body)
        resp.raise_for_status()
        run_id = resp.json().get("run_id")
        data = await self._poll_async(client, "/scrape/run/data", run_id)
        return data, run_id

    async def _poll_async(self, client: httpx.AsyncClient, path: str, run_id: str) -> Any:
        """Poll a scrape job until it completes; return its ``data``.

        202 means still running. A 200 with ``status == "completed"`` returns
        the data; ``status == "failed"`` raises. The overall wait is bounded
        by the caller's ``asyncio.wait_for`` guard.
        """
        while True:
            resp = await client.get(path, params={"run_id": run_id})
            if resp.status_code != 202:
                resp.raise_for_status()
                body = resp.json()
                status = body.get("status")
                if status == "completed":
                    return body.get("data")
                if status == "failed":
                    raise RuntimeError(f"Oxylabs scrape failed: {body.get('error_code')}")
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    # ------------------------------------------------------------ setup hint

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Oxylabs AI Studio",
            "badge": "paid",
            "tag": (
                "Search + extract backed by Oxylabs' AI Studio. "
                "Per-call render_javascript and geo_location supported."
            ),
            "env_vars": [
                {
                    "key": _API_KEY_ENV,
                    "prompt": "Oxylabs AI Studio API key",
                    "url": _API_KEY_URL,
                },
            ],
        }
