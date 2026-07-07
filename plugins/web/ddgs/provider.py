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

# Overall wall-clock cap for a single ddgs search. The DDGS constructor's
# ``timeout`` only bounds individual HTTP requests; ddgs's multi-engine retry
# loop has no overall cap, so a slow/rate-limited DuckDuckGo response can hang
# the (single, shared) agent loop indefinitely and block every platform
# (#36776). Enforce a hard cap here via a worker thread.
_SEARCH_TIMEOUT_SECS = 30


def _ensure_ddgs_package_importable() -> bool:
    """Return True once ``ddgs`` is importable, lazy-installing if allowed."""
    try:
        import ddgs  # noqa: F401

        return True
    except ImportError:
        pass

    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.ddgs", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — lazy_deps carries the install hint
        logger.debug("DDGS lazy install unavailable: %s", exc)
        return False

    try:
        import ddgs  # noqa: F401

        return True
    except ImportError:
        return False


def _run_ddgs_search(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """Run the blocking ddgs query and return normalized hits.

    Module-level (not a closure) so tests can patch it directly without
    spawning a real multi-second worker thread. ``DDGS(timeout=...)`` bounds
    each individual HTTP request; the overall wall-clock cap is enforced by
    the caller via a future timeout.
    """
    from ddgs import DDGS  # type: ignore

    results: list[dict[str, Any]] = []
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
        return _ensure_ddgs_package_importable()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a DuckDuckGo search and return normalized results.

        The synchronous ``ddgs`` call is run in a worker thread with a hard
        wall-clock timeout (``_SEARCH_TIMEOUT_SECS``) so a hung search cannot
        block the shared agent loop indefinitely (#36776).
        """
        if not _ensure_ddgs_package_importable():
            return {
                "success": False,
                "error": (
                    "ddgs package is not installed and lazy installation could not make "
                    "it available — run `pip install ddgs==9.14.4`"
                ),
            }

        # DDGS().text yields at most `max_results` items; we cap defensively
        # in case the package ignores the hint.
        safe_limit = max(1, int(limit))

        # A fresh single-worker pool per call (rather than a module-level one)
        # is intentional: on timeout the blocking ddgs call cannot be cancelled
        # and keeps running, so a shared pool would serialise every later search
        # behind that hung worker. A per-call pool isolates each search from a
        # previously-hung one.
        pool = _cf.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_run_ddgs_search, query, safe_limit)
            try:
                web_results = future.result(timeout=_SEARCH_TIMEOUT_SECS)
            except _cf.TimeoutError:
                logger.warning(
                    "DDGS search timed out after %ds for query: %r",
                    _SEARCH_TIMEOUT_SECS, query,
                )
                return {
                    "success": False,
                    "error": (
                        f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECS}s — "
                        "DuckDuckGo may be rate-limiting or slow. Try again later "
                        "or switch to a different search provider."
                    ),
                }
        except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
            logger.warning("DDGS search error: %s", exc)
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}
        finally:
            # Return immediately without joining the worker. On timeout the
            # already-running ddgs call can't be cancelled (cancel_futures only
            # affects not-yet-started work), so the worker runs to completion
            # on its own; it writes nothing shared, so leaking it is safe.
            pool.shutdown(wait=False, cancel_futures=True)

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
