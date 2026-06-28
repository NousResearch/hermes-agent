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
import html as _html
import logging
import re
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlparse

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Overall wall-clock cap for a single ddgs search. The DDGS constructor's
# ``timeout`` only bounds individual HTTP requests; ddgs's multi-engine retry
# loop has no overall cap, so a slow/rate-limited DuckDuckGo response can hang
# the (single, shared) agent loop indefinitely and block every platform
# (#36776). Enforce a hard cap here via a worker thread.
_SEARCH_TIMEOUT_SECS = 30
_DEFAULT_DDGS_BACKEND = "duckduckgo"


def _ddgs_backend_from_config() -> str:
    try:
        from hermes_cli.config import load_config

        web_cfg = load_config().get("web") or {}
        configured = web_cfg.get("ddgs_backend") if isinstance(web_cfg, dict) else ""
    except Exception:
        configured = ""
    return str(configured or _DEFAULT_DDGS_BACKEND).strip() or _DEFAULT_DDGS_BACKEND


def _run_ddgs_search(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """Run the blocking ddgs query and return normalized hits.

    Module-level (not a closure) so tests can patch it directly without
    spawning a real multi-second worker thread. ``DDGS(timeout=...)`` bounds
    each individual HTTP request; the overall wall-clock cap is enforced by
    the caller via a future timeout.
    """
    from ddgs import DDGS  # type: ignore

    backend = _ddgs_backend_from_config()
    results: list[dict[str, Any]] = []
    with DDGS(timeout=10) as client:
        try:
            hits = client.text(query, max_results=safe_limit, backend=backend)
        except TypeError as exc:
            if "backend" not in str(exc):
                raise
            hits = client.text(query, max_results=safe_limit)
        except Exception as exc:
            if not _should_try_html_fallback(exc):
                raise
            logger.info("DDGS package search failed; trying DuckDuckGo HTML fallback: %s", exc)
            return _run_duckduckgo_html_search(query, safe_limit)

        for i, hit in enumerate(hits):
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


def _should_try_html_fallback(exc: Exception) -> bool:
    text = str(exc)
    return (
        "No results found" in text
        or "ConnectError" in text
        or "tail" in text and ".ts.net" in text
    )


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", _html.unescape(text)).strip()


def _decode_duckduckgo_url(href: str) -> str:
    href = _html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    target = parse_qs(parsed.query).get("uddg", [""])[0]
    return unquote(target) if target else href


def _run_duckduckgo_html_search(query: str, safe_limit: int) -> list[dict[str, Any]]:
    import httpx

    response = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
        timeout=10,
    )
    response.raise_for_status()
    chunks = re.findall(
        r'<div class="result results_links results_links_deep web-result.*?</div>\s*</div>',
        response.text,
        re.S,
    )
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        title_match = re.search(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            chunk,
            re.S,
        )
        if not title_match:
            continue
        snippet_match = re.search(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            chunk,
            re.S,
        )
        url = _decode_duckduckgo_url(title_match.group(1))
        results.append(
            {
                "title": _strip_html(title_match.group(2)),
                "url": url,
                "description": _strip_html(snippet_match.group(1)) if snippet_match else "",
                "position": len(results) + 1,
            }
        )
        if len(results) >= safe_limit:
            break
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

        The synchronous ``ddgs`` call is run in a worker thread with a hard
        wall-clock timeout (``_SEARCH_TIMEOUT_SECS``) so a hung search cannot
        block the shared agent loop indefinitely (#36776).
        """
        try:
            import ddgs  # type: ignore  # noqa: F401 — availability probe
        except ImportError:
            return {
                "success": False,
                "error": "ddgs package is not installed — run `pip install ddgs`",
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
