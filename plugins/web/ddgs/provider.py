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

import logging
import threading
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Maximum wall-clock seconds for the entire ddgs search operation.
# Individual HTTP requests already have their own timeout (DDGS default 5 s),
# but when multiple engines chain slow requests the overall call can hang
# indefinitely.  This cap prevents a single web search from blocking the
# entire agent loop (issue #36776).
_SEARCH_TIMEOUT_SECS = 30


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
        """Execute a DuckDuckGo search and return normalized results.

        The synchronous ``ddgs`` library call is wrapped in a thread with a
        hard wall-clock timeout so that a hung search cannot block the entire
        agent loop indefinitely.
        """
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

        def _do_search() -> None:
            """Run the actual ddgs query inside a thread, storing the result."""
            try:
                results: list[dict[str, Any]] = []
                with DDGS() as client:
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
                _box["results"] = results
            except Exception as exc:  # noqa: BLE001
                _box["error"] = exc

        # Run in a daemon thread so the caller can enforce a wall-clock
        # timeout.  Daemon threads are abandoned on timeout — the ddgs
        # library's own HTTP timeouts will eventually release resources.
        _box: dict[str, Any] = {}
        worker = threading.Thread(target=_do_search, daemon=True)

        try:
            worker.start()
            worker.join(timeout=_SEARCH_TIMEOUT_SECS)
            if worker.is_alive():
                logger.warning(
                    "DDGS search timed out after %ds for query: %s",
                    _SEARCH_TIMEOUT_SECS, query,
                )
                return {
                    "success": False,
                    "error": (
                        f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECS}s. "
                        "Try a shorter query or switch to a faster search backend."
                    ),
                }
            if "error" in _box:
                raise _box["error"]
            web_results = _box.get("results", [])
        except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
            logger.warning("DDGS search error: %s", exc)
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}

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
