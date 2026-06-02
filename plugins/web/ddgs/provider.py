"""DuckDuckGo search — plugin form (via the ``ddgs`` package).

Subclasses the plugin-facing :class:`agent.web_search_provider.WebSearchProvider`.
The legacy in-tree module ``tools.web_providers.ddgs`` was removed in the
same commit that moved this code under ``plugins/``; this file is now the
canonical implementation.

The ``ddgs`` package is an optional dependency. ``is_available()`` reflects
whether the package is importable; the plugin still registers either way so
``hermes tools`` can prompt the user to install it.
"""

from __future__ import annotations

import concurrent.futures as _cf
import logging
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class DDGSWebSearchProvider(WebSearchProvider):
    """DuckDuckGo HTML-scrape search provider.

    No API key needed. Rate limits are enforced server-side by DuckDuckGo;
    the provider surfaces ``DuckDuckGoSearchException`` and other ddgs errors
    as ``{"success": False, "error": ...}`` rather than raising.
    """

    @property
    def name(self) -> str:
        return "ddgs"

    @property
    def display_name(self) -> str:
        return "DuckDuckGo (ddgs)"

    def is_available(self) -> bool:
        """Return True when the ``ddgs`` package is importable.

        Probes the import once; cheap because Python caches the import. Must
        NOT perform network I/O — runs at tool-registration time and on every
        ``hermes tools`` paint.
        """
        try:
            import ddgs  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a DuckDuckGo search and return normalized results."""
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            return {
                "success": False,
                "error": "ddgs package is not installed — run `pip install ddgs`",
            }

        # DDGS().text yields at most `max_results` items; we cap defensively
        # in case the package ignores the hint.
        safe_limit = max(1, int(limit))

        # Overall search timeout so a slow/rate-limited DuckDuckGo response
        # cannot hang the agent loop indefinitely.  The DDGS constructor's
        # ``timeout`` parameter only covers individual HTTP requests; the
        # multi-engine retry loop inside ddgs has no wall-clock cap of its
        # own.  We enforce one here via concurrent.futures so the provider
        # always returns within a bounded time regardless of ddgs internals.
        _SEARCH_TIMEOUT = 30  # seconds; covers the full text() call

        def _search_sync() -> list:
            results = []
            with DDGS(timeout=10) as client:
                for i, hit in enumerate(client.text(query, max_results=safe_limit)):
                    if i >= safe_limit:
                        break
                    url = str(hit.get("href") or hit.get("url") or "")
                    results.append(
                        {
                            "title": str(hit.get("title", "")),
                            "url": url,
                            "description": str(hit.get("body", "")),
                            "position": i + 1,
                        }
                    )
            return results

        _pool = _cf.ThreadPoolExecutor(max_workers=1)
        try:
            _future = _pool.submit(_search_sync)
            try:
                web_results = _future.result(timeout=_SEARCH_TIMEOUT)
            except _cf.TimeoutError:
                logger.warning(
                    "DDGS search timed out after %ds for query: %r",
                    _SEARCH_TIMEOUT, query,
                )
                return {
                    "success": False,
                    "error": (
                        f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT}s — "
                        "DuckDuckGo may be rate-limiting or slow. "
                        "Try again later or switch to a different search provider."
                    ),
                }
        except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
            logger.warning("DDGS search error: %s", exc)
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}
        finally:
            _pool.shutdown(wait=False, cancel_futures=True)

        logger.info("DDGS search '%s': %d results (limit %d)", query, len(web_results), limit)
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "DuckDuckGo (ddgs)",
            "badge": "free · no key · search only",
            "tag": "Search via the ddgs Python package — no API key (pair with any extract provider)",
            "env_vars": [],
            # Trigger `_run_post_setup("ddgs")` after the user picks this row
            # so the ddgs Python package gets pip-installed on first selection.
            "post_setup": "ddgs",
        }
