"""fastCRW web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Wraps the
``crw`` Python SDK (https://fastcrw.com), the single-binary web scraper for
AI agents (Firecrawl-compatible API).

Three operating modes, selected automatically from the environment:

- **Cloud** — ``CRW_API_KEY`` set (no URL): HTTP mode against the hosted
  fastCRW API. Supports both search and extract.
- **Self-hosted** — ``CRW_API_URL`` set: HTTP mode against your own CRW
  instance. Supports both search and extract.
- **Local / subprocess** — neither set: spawns the ``crw-mcp`` binary
  (auto-downloaded on first use). Extract works with zero config; search is
  cloud-only (the SDK raises in subprocess mode), so :meth:`supports_search`
  reports ``False`` and the dispatcher falls back to another search backend.

Config keys this provider responds to::

    web:
      search_backend: "crw"     # explicit per-capability (cloud/self-hosted)
      extract_backend: "crw"    # explicit per-capability (any mode)
      backend: "crw"            # shared fallback for both

Env vars::

    CRW_API_KEY=...   # https://fastcrw.com — enables cloud search + extract
    CRW_API_URL=...   # self-hosted instance, e.g. http://localhost:3000

The SDK is lazy-loaded via :func:`tools.lazy_deps.ensure` ("search.crw") so
cold-start CLI users who haven't configured CRW never pay the import cost.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

# Hosted fastCRW API base — used when only CRW_API_KEY is set (cloud mode).
_CRW_DEFAULT_URL = "https://fastcrw.com/api"


def _crw_cloud_configured() -> bool:
    """Return True when cloud or self-hosted mode is configured.

    Either credential gives the SDK an ``api_url`` and therefore enables
    ``search()`` (which is cloud-only). Cheap env probe — no network, no
    package import — safe to call on every ``hermes tools`` paint.
    """
    return bool(
        os.getenv("CRW_API_KEY", "").strip() or os.getenv("CRW_API_URL", "").strip()
    )


def _crw_binary_present() -> bool:
    """Return True when the ``crw-mcp`` binary is already available locally.

    Network-free: checks the ``CRW_BINARY`` override, the system ``PATH``,
    and the SDK's download cache, but never triggers a download (that would
    make ``is_available()`` — called at registration time and on every
    picker paint — block on a network fetch).
    """
    env_path = os.environ.get("CRW_BINARY", "").strip()
    if env_path:
        return Path(env_path).exists()
    if shutil.which("crw-mcp"):
        return True
    try:
        from crw._binary import _find_cached_latest

        return _find_cached_latest() is not None
    except Exception:  # noqa: BLE001 — crw not installed / cache unreadable
        return False


def _get_crw_client() -> Any:
    """Lazy-import and cache a ``CrwClient`` configured for the active mode.

    Cache lives on :mod:`tools.web_tools` (as ``_crw_client``) so unit tests
    that reset that name between cases keep working — mirrors the exa/parallel
    plugins.
    """
    import tools.web_tools as _wt

    cached = getattr(_wt, "_crw_client", None)
    if cached is not None:
        return cached

    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.crw", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — lazy_deps surfaces install hints
        raise ImportError(str(exc))

    from crw import CrwClient  # noqa: WPS433 — deliberately lazy

    api_url = os.getenv("CRW_API_URL", "").strip().rstrip("/")
    api_key = os.getenv("CRW_API_KEY", "").strip()

    if api_url:
        client = CrwClient(api_url=api_url, api_key=api_key or None)
    elif api_key:
        client = CrwClient(api_url=_CRW_DEFAULT_URL, api_key=api_key)
    else:
        # Subprocess mode — no server needed, binary auto-downloaded on use.
        client = CrwClient()

    _wt._crw_client = client
    return client


def _reset_client_for_tests() -> None:
    """Drop the cached CRW client so tests can re-instantiate cleanly."""
    import tools.web_tools as _wt

    _wt._crw_client = None


class CrwWebSearchProvider(WebSearchProvider):
    """fastCRW search + extract provider.

    ``extract()`` is async (mirrors firecrawl/parallel) so the per-URL scrape
    runs in a worker thread with a timeout without blocking the event loop.
    ``search()`` is sync — the SDK call is a blocking HTTP request.
    """

    @property
    def name(self) -> str:
        return "crw"

    @property
    def display_name(self) -> str:
        return "fastCRW"

    def is_available(self) -> bool:
        """Return True when CRW can service calls in the current environment.

        Cloud / self-hosted creds → available (SDK is lazy-installed on first
        use, like exa). Otherwise available only when the subprocess binary is
        already present locally (network-free check — never downloads here).
        """
        if _crw_cloud_configured():
            return True
        return _crw_binary_present()

    def supports_search(self) -> bool:
        """Search is cloud-only. Advertise it only in cloud/self-hosted mode.

        In subprocess mode the SDK's ``search()`` raises, so reporting False
        lets the dispatcher fall back to another available search backend
        instead of surfacing an error.
        """
        return _crw_cloud_configured()

    def supports_extract(self) -> bool:
        """Extract (scrape) works in every mode, including keyless subprocess."""
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a fastCRW search (cloud / self-hosted only).

        Returns ``{"success": True, "data": {"web": [...]}}`` on success, or
        ``{"success": False, "error": str}`` on failure (incl. subprocess-mode
        ``CrwError`` and SDK install errors).
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("fastCRW search: '%s' (limit=%d)", query, limit)
            raw = _get_crw_client().search(query, limit=min(limit, 20))

            # The SDK already unwraps the API envelope's ``data``, leaving one of
            # several shapes (verified against a live crw instance + the
            # crw-server response contract):
            #   • a flat list                  — some instances return data=[...]
            #   • {"results": [...]}           — crw-server canonical (no sources)
            #   • {"results": {"web": [...]}}  — canonical, grouped by source
            #   • {"web": [...]}               — grouped, no results wrapper
            #   • None                         — data: null
            # Walk these without assuming a single one so a successful search
            # never silently yields zero results, and ``data: null`` can't raise.
            # (Genuine upstream failures — ``success: false`` / non-2xx — are
            # raised by the SDK and caught below, never reaching this point.)
            payload = raw.get("results", raw) if isinstance(raw, dict) else raw
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                items = payload.get("web")
                if not isinstance(items, list):
                    items = next(
                        (v for v in payload.values() if isinstance(v, list)), []
                    )
            else:
                items = []

            web_results = []
            for i, result in enumerate(items):
                if not isinstance(result, dict):
                    continue
                web_results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", "")
                        or result.get("snippet", ""),
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except ImportError as exc:
            return {"success": False, "error": f"crw SDK not installed: {exc}"}
        except Exception as exc:  # noqa: BLE001 — surface as failure
            logger.warning("fastCRW search error: %s", exc)
            return {"success": False, "error": f"fastCRW search failed: {exc}"}

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via fastCRW ``scrape``.

        Async — each URL is scraped in a worker thread with a 60s timeout so a
        slow page can't stall the event loop or the whole batch. Per-URL
        failures (policy block, redirect re-block, timeout, scrape error)
        become result entries with an ``error`` field rather than raising.
        SSRF filtering already happened in the web_extract dispatcher; here we
        enforce the website-access policy before and after redirects, matching
        the firecrawl provider.
        """
        from tools.interrupt import is_interrupted

        format = kwargs.get("format")
        formats: List[str]
        if format == "markdown":
            formats = ["markdown"]
        elif format == "html":
            formats = ["html"]
        else:
            formats = ["markdown", "html"]

        try:
            client = _get_crw_client()
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": f"crw SDK not installed: {exc}"}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            return [
                {"url": u, "title": "", "content": "", "error": f"fastCRW unavailable: {exc}"}
                for u in urls
            ]

        results: List[Dict[str, Any]] = []
        for url in urls:
            if is_interrupted():
                results.append({"url": url, "error": "Interrupted", "title": ""})
                continue

            # Pre-scrape website policy gate
            blocked = check_website_access(url)
            if blocked:
                logger.info(
                    "Blocked web_extract for %s by rule %s",
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

            try:
                logger.info("fastCRW scraping: %s", url)
                try:
                    scrape_data = await asyncio.wait_for(
                        asyncio.to_thread(client.scrape, url, formats=formats),
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    logger.warning("fastCRW scrape timed out for %s", url)
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": (
                                "Scrape timed out after 60s — page may be too large "
                                "or unresponsive. Try browser_navigate instead."
                            ),
                        }
                    )
                    continue

                if not isinstance(scrape_data, dict):
                    scrape_data = {}
                metadata = scrape_data.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                title = metadata.get("title", "")
                final_url = metadata.get("sourceURL", url)

                # Re-check website-access policy after any redirect
                final_blocked = check_website_access(final_url)
                if final_blocked:
                    logger.info(
                        "Blocked redirected web_extract for %s by rule %s",
                        final_blocked["host"],
                        final_blocked["rule"],
                    )
                    results.append(
                        {
                            "url": final_url,
                            "title": title,
                            "content": "",
                            "raw_content": "",
                            "error": final_blocked["message"],
                            "blocked_by_policy": {
                                "host": final_blocked["host"],
                                "rule": final_blocked["rule"],
                                "source": final_blocked["source"],
                            },
                        }
                    )
                    continue

                content_markdown = scrape_data.get("markdown")
                content_html = scrape_data.get("html")
                if format == "markdown" or (format is None and content_markdown):
                    chosen_content = content_markdown or ""
                else:
                    chosen_content = content_html or content_markdown or ""

                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": chosen_content,
                        "raw_content": chosen_content,
                        "metadata": metadata,
                    }
                )
            except Exception as scrape_err:  # noqa: BLE001
                logger.debug("fastCRW scrape failed for %s: %s", url, scrape_err)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": str(scrape_err),
                    }
                )

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "fastCRW",
            "badge": "free",
            "tag": (
                "Single-binary web scraper. Extract runs locally with no key "
                "(auto-installs the crw-mcp binary); add a cloud key or "
                "self-hosted URL to enable search."
            ),
            "env_vars": [
                {
                    "key": "CRW_API_KEY",
                    "prompt": "fastCRW cloud API key (optional — enables search)",
                    "url": "https://fastcrw.com",
                },
                {
                    "key": "CRW_API_URL",
                    "prompt": "Self-hosted CRW URL (optional, e.g. http://localhost:3000)",
                },
            ],
        }
