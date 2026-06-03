"""DuckDuckGo search — plugin form (via the ``ddgs`` package).

Subclasses the plugin-facing
:class:`agent.web_search_provider.WebSearchProvider`.
The legacy in-tree module ``tools.web_providers.ddgs`` was removed in
the same commit that moved this code under ``plugins/``; this file is
now the canonical implementation.

The ``ddgs`` package is an optional dependency. ``is_available()``
reflects whether the package is importable; the plugin still registers
either way so ``hermes tools`` can prompt the user to install it.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Hard wall-clock cap on any single DDGS search call.  DDGS can hang for
# 20+ minutes when DuckDuckGo rate-limits it because individual HTTP
# request timeouts do not bound the *total* retry loop time.
_SEARCH_TIMEOUT_SECONDS = 30


def _run_ddgs_search(query: str, safe_limit: int) -> List[Dict[str, Any]]:
    """Blocking helper executed inside a worker thread.

    Isolated so that :func:`concurrent.futures.Future.result` can kill
    the wait after ``_SEARCH_TIMEOUT_SECONDS`` even though the thread
    itself cannot be forcibly interrupted.
    """
    from ddgs import DDGS  # type: ignore

    web_results: List[Dict[str, Any]] = []
    with DDGS() as client:
        for i, hit in enumerate(
            client.text(query, max_results=safe_limit)
        ):
            if i >= safe_limit:
                break
            url = str(hit.get("href") or hit.get("url") or "")
            web_results.append(
                {
                    "title": str(hit.get("title", "")),
                    "url": url,
                    "description": str(hit.get("body", "")),
                    "position": i + 1,
                }
            )
    return web_results


class DDGSWebSearchProvider(WebSearchProvider):
    """DuckDuckGo HTML-scrape search provider.

    No API key needed. Rate limits are enforced server-side by
    DuckDuckGo; the provider surfaces ``DuckDuckGoSearchException`` and
    other ddgs errors as ``{"success": False, "error": ...}`` rather
    than raising.
    """

    @property
    def name(self) -> str:
        return "ddgs"

    @property
    def display_name(self) -> str:
        return "DuckDuckGo (ddgs)"

    def is_available(self) -> bool:
        """Return True when the ``ddgs`` package is importable.

        Probes the import once; cheap because Python caches the import.
        Must NOT perform network I/O — runs at tool-registration time
        and on every ``hermes tools`` paint.
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
        """Execute a DuckDuckGo search and return normalized results.

        The call is run in a thread and bounded by
        ``_SEARCH_TIMEOUT_SECONDS`` (default 30 s).  When DuckDuckGo
        rate-limits the ddgs client it can retry internally for 20+
        minutes; this cap prevents the agent from hanging indefinitely.
        """
        try:
            from ddgs import DDGS  # noqa: F401 — confirm package present
        except ImportError:
            return {
                "success": False,
                "error": (
                    "ddgs package is not installed"
                    " — run `pip install ddgs`"
                ),
            }

        # DDGS().text yields at most `max_results` items; cap defensively
        # in case the package ignores the hint.
        safe_limit = max(1, int(limit))

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                future = executor.submit(
                    _run_ddgs_search, query, safe_limit
                )
                web_results = future.result(
                    timeout=_SEARCH_TIMEOUT_SECONDS
                )
        except concurrent.futures.TimeoutError:
            logger.warning(
                "DDGS search timed out after %ds for query %r",
                _SEARCH_TIMEOUT_SECONDS,
                query,
            )
            return {
                "success": False,
                "error": (
                    f"DuckDuckGo search timed out after"
                    f" {_SEARCH_TIMEOUT_SECONDS} seconds"
                    " (likely rate-limited). Try again later or"
                    " switch to a different search provider."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("DDGS search error: %s", exc)
            return {
                "success": False,
                "error": f"DuckDuckGo search failed: {exc}",
            }

        logger.info(
            "DDGS search %r: %d results (limit %d)",
            query,
            len(web_results),
            limit,
        )
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "DuckDuckGo (ddgs)",
            "badge": "free · no key · search only",
            "tag": (
                "Search via the ddgs Python package"
                " — no API key (pair with any extract provider)"
            ),
            "env_vars": [],
            # Trigger `_run_post_setup("ddgs")` after the user picks
            # this row so the ddgs Python package gets pip-installed on
            # first selection.
            "post_setup": "ddgs",
        }
