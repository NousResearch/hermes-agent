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
from typing import Any, Dict, List

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
        # The ``ddgs`` Python package gained an ``extract()`` method in 8.0
        # (it used to be search-only via HTML scraping). The hermes
        # ``ddgs`` provider was registered as search-only when the upstream
        # package was 5.x, and the capability flag was never updated when
        # the package added multi-engine search and per-URL extraction.
        # We override the base default (False) to True so hermes routes
        # ``web_extract`` through this provider when ``web.backend=ddgs``
        # and no extract-capable keyed backend (tavily/parallel/exa/
        # firecrawl) is configured. See extract() implementation below
        # for the actual call.
        return True

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

        try:
            web_results = []
            with DDGS() as client:
                for i, hit in enumerate(client.text(query, max_results=safe_limit)):
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
        except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
            logger.warning("DDGS search error: %s", exc)
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}

        logger.info("DDGS search '%s': %d results (limit %d)", query, len(web_results), limit)
        return {"success": True, "data": {"web": web_results}}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via the ``ddgs`` package.

        Returns the hermes web_extract shape: a list of dicts with ``url``,
        ``title``, ``content``, ``raw_content`` (best-effort duplicates of
        ``content`` — ddgs doesn't separate the two), and ``metadata``.

        The upstream ``ddgs`` ``DDGS().extract()`` returns ``{url, content}``
        per URL. We normalize to the registry contract; per-URL failures
        surface as ``{"error": ...}`` entries so the caller can decide
        whether to retry or skip.
        """
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            return [
                {
                    "url": u,
                    "error": "ddgs package is not installed — run `pip install ddgs`",
                }
                for u in urls
            ]

        results: List[Dict[str, Any]] = []
        fmt = (kwargs.get("format") or "text_markdown").strip()
        with DDGS() as client:
            for url in urls:
                try:
                    raw = client.extract(url, fmt=fmt)
                except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
                    logger.warning("DDGS extract error for %s: %s", url, exc)
                    results.append({"url": url, "error": f"DuckDuckGo extract failed: {exc}"})
                    continue
                if not isinstance(raw, dict):
                    results.append(
                        {"url": url, "error": f"DuckDuckGo extract returned non-dict: {type(raw).__name__}"}
                    )
                    continue
                content = str(raw.get("content", "") or "")
                title = str(raw.get("title", "") or "")
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": content,
                        "raw_content": content,  # ddgs doesn't separate
                        "metadata": {
                            "source": "ddgs",
                            "format": fmt,
                            "length": len(content),
                        },
                    }
                )
        logger.info("DDGS extract: %d/%d URLs succeeded", sum(1 for r in results if "error" not in r), len(urls))
        return results

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
