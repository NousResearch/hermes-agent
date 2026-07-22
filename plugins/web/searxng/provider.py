"""SearXNG search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Same JSON
API call (``/search?format=json``), same result normalization. The legacy
in-tree module ``tools.web_providers.searxng`` was removed in the same
commit that moved this code under ``plugins/``; this file is now the
canonical implementation.

Search-only — SearXNG aggregates results from upstream engines but does not
fetch/extract arbitrary URLs. ``supports_extract()`` returns False.

Config keys this provider responds to::

    web:
      search_backend: "searxng"     # explicit per-capability
      backend: "searxng"            # shared fallback

Env var::

    SEARXNG_URL=http://localhost:8080
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _searxng_url() -> str:
    """Return SEARXNG_URL from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value("SEARXNG_URL")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("SEARXNG_URL", "")
    return (val or "").strip()


class SearXNGWebSearchProvider(WebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def display_name(self) -> str:
        return "SearXNG"

    def is_available(self) -> bool:
        """Return True when ``SEARXNG_URL`` is set."""
        return bool(_searxng_url())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the configured SearXNG instance.

        Uses ``format=html`` + regex parsing of ``<article class="result">``
        containers rather than ``format=json`` because SearXNG returns
        HTTP 403 on ``format=json`` whenever ``server.public_instance``
        is ``false`` (intentional design — JSON output is restricted on
        private instances to limit bot access). ``format=html`` is the
        same rendered page users see in the browser, always available
        on every SearXNG instance (no flag required), and the
        ``<article class="result">`` markup is stable across SearXNG
        themes and result-type variants (default, images, videos, …).
        This works on both private and public instances with no
        ``link_token`` handshake required.

        See https://docs.searxng.org/admin/searx.limiter.html for the
        rationale (SearXNG intentionally restricts JSON output on
        private instances to limit bot access).
        """
        import re

        import httpx

        base_url = _searxng_url().rstrip("/")
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        params: Dict[str, Any] = {
            "q": query,
            "format": "html",
            "language": "auto",
            "safesearch": 0,
        }

        try:
            resp = httpx.get(
                f"{base_url}/search",
                params=params,
                timeout=15,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Hermes SearXNG Provider) "
                        "AppleWebKit/537.36 Chrome/126.0"
                    ),
                    "Accept": "text/html, application/xhtml+xml, */*",
                    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"SearXNG returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach SearXNG at {base_url}: {exc}",
            }

        body = resp.text

        # Extract <article class="result...">...</article> containers. The
        # `result` prefix matches all result-type variants (`result-default`,
        # `result-images`, `result-videos`, `result-map`, ...).
        articles = re.findall(
            r'<article[^>]*class="result[^"]*"[^>]*>(.*?)</article>',
            body,
            re.DOTALL,
        )

        results: list = []
        for art in articles:
            title_url_m = re.search(
                r'<h3[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.+?)</a>',
                art,
                re.DOTALL,
            )
            if not title_url_m:
                # Container without a parsable title link — skip.
                continue
            title = re.sub(r"<[^>]+>", "", title_url_m.group(2)).strip()
            url = title_url_m.group(1).strip()

            snippet = ""
            snippet_m = re.search(
                r'<p[^>]+class="content[^"]*"[^>]*>(.+?)</p>',
                art,
                re.DOTALL,
            )
            if snippet_m:
                snippet = re.sub(r"<[^>]+>", "", snippet_m.group(1))
                snippet = re.sub(r"\s+", " ", snippet).strip()

            results.append(
                {"title": title, "url": url, "description": snippet}
            )

        # SearXNG's HTML returns results in relevance order; cap to ``limit``
        # and add 1-based position so downstream consumers don't have to.
        web_results = results[:limit]
        for i, r in enumerate(web_results):
            r["position"] = i + 1

        logger.info(
            "SearXNG (html) search '%s': %d results (from %d articles, limit %d)",
            query,
            len(web_results),
            len(articles),
            limit,
        )

        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SearXNG",
            "badge": "free · self-hosted",
            "tag": "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "env_vars": [
                {
                    "key": "SEARXNG_URL",
                    "prompt": "SearXNG instance URL (e.g. http://localhost:8080)",
                    "url": "https://searx.space/",
                },
            ],
        }
