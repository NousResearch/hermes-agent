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

Env vars::

    SEARXNG_URL=http://localhost:8080
    SEARXNG_GENERAL_ENGINES=bing,mojeek,presearch
"""

from __future__ import annotations

import logging
import os
from html.parser import HTMLParser
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_NEWS_HINTS = (
    "news",
    "headlines",
    "breaking",
    "latest",
    "today",
    "recent",
    "current",
    "updates",
)
_DEFAULT_GENERAL_ENGINES = "bing,mojeek,presearch"


class _SearXNGHTMLParser(HTMLParser):
    """Small dependency-free parser for SearXNG HTML result pages."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.results: List[Dict[str, Any]] = []
        self._in_article = False
        self._in_title_link = False
        self._in_content = False
        self._current: Dict[str, str] = {}
        self._title_chunks: List[str] = []
        self._content_chunks: List[str] = []

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        for key, value in attrs:
            if key == "class" and value:
                return set(value.split())
        return set()

    @staticmethod
    def _attr(attrs: list[tuple[str, str | None]], name: str) -> str:
        for key, value in attrs:
            if key == name and value:
                return value
        return ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.results) >= self.limit:
            return
        classes = self._classes(attrs)
        if tag == "article" and "result" in classes:
            self._in_article = True
            self._current = {}
            self._title_chunks = []
            self._content_chunks = []
            return
        if not self._in_article:
            return
        if tag == "a" and not self._current.get("url"):
            href = self._attr(attrs, "href")
            if href:
                self._current["url"] = href
                self._in_title_link = True
        elif tag == "p" and "content" in classes:
            self._in_content = True

    def handle_data(self, data: str) -> None:
        if self._in_title_link:
            self._title_chunks.append(data)
        elif self._in_content:
            self._content_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title_link:
            self._in_title_link = False
        elif tag == "p" and self._in_content:
            self._in_content = False
        elif tag == "article" and self._in_article:
            self._in_article = False
            title = " ".join("".join(self._title_chunks).split())
            url = self._current.get("url", "")
            if title and url:
                self.results.append(
                    {
                        "title": title,
                        "url": url,
                        "content": " ".join("".join(self._content_chunks).split()),
                        "score": 0,
                    }
                )


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
        return bool(os.getenv("SEARXNG_URL", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def _base_url(self) -> str:
        return os.getenv("SEARXNG_URL", "").strip().rstrip("/")

    def _general_engines(self) -> str:
        """Return comma-separated engines for general queries.

        Many self-hosted SearXNG instances have default general engines that are
        CAPTCHA/rate-limit prone. A known-good default improves fresh installs,
        while ``SEARXNG_GENERAL_ENGINES=`` lets operators opt out.
        """
        return os.getenv("SEARXNG_GENERAL_ENGINES", _DEFAULT_GENERAL_ENGINES).strip()

    def _looks_like_news_query(self, query: str) -> bool:
        q = query.lower()
        return any(hint in q for hint in _NEWS_HINTS)

    def _json_params(self, query: str, *, category: str = "general", language: str = "en", engines: str = "") -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if category:
            params["categories"] = category
        if language:
            params["language"] = language
        if engines and category == "general":
            params["engines"] = engines
        return params

    def _request_json(self, base_url: str, params: Dict[str, Any]) -> tuple[list[dict], dict]:
        import httpx

        resp = httpx.get(
            f"{base_url}/search",
            params=params,
            timeout=15,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []
        return raw_results, data

    def _request_html(self, base_url: str, query: str, limit: int) -> list[dict]:
        import httpx

        resp = httpx.get(
            f"{base_url}/search",
            params={"q": query},
            timeout=15,
            headers={"Accept": "text/html"},
        )
        resp.raise_for_status()
        parser = _SearXNGHTMLParser(limit=limit)
        parser.feed(resp.text)
        return parser.results

    def _normalize_results(self, raw_results: list[dict], limit: int) -> list[dict]:
        def _score(result: dict) -> float:
            try:
                return float(result.get("score", 0))
            except (TypeError, ValueError):
                return 0.0

        valid_results = [r for r in raw_results if r.get("url")]
        sorted_results = sorted(valid_results, key=_score, reverse=True)[:limit]
        return [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "") or r.get("snippet", "")),
                "position": i + 1,
            }
            for i, r in enumerate(sorted_results)
        ]

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the configured SearXNG instance.

        The provider tries the JSON API first, then retries common self-hosted
        SearXNG failure modes before falling back to parsing the HTML result
        page. The returned shape stays compatible with other Hermes web
        providers: ``{"success": True, "data": {"web": [...]}}``.
        """
        import httpx

        base_url = self._base_url()
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        general_engines = self._general_engines()
        attempts: list[tuple[str, Dict[str, Any]]] = []
        if self._looks_like_news_query(query):
            attempts.append(("news", self._json_params(query, category="news", language="en")))
            attempts.append(("news-fallback-general", self._json_params(query, category="general", language="en", engines=general_engines)))
        else:
            attempts.append(("general", self._json_params(query, category="general", language="en", engines=general_engines)))

        attempts.append(("without-language", self._json_params(query, category="general", language="", engines=general_engines)))
        if general_engines:
            attempts.append(("default-engines", self._json_params(query, category="general", language="en", engines="")))

        last_error = ""
        last_data: dict = {}
        raw_results: list[dict] = []
        for label, params in attempts:
            try:
                raw_results, last_data = self._request_json(base_url, params)
                if any(r.get("url") for r in raw_results):
                    logger.info("SearXNG %s search returned %d raw results for %r", label, len(raw_results), query)
                    break
                if raw_results:
                    logger.info(
                        "SearXNG %s search returned %d URL-less raw results for %r; continuing fallbacks",
                        label,
                        len(raw_results),
                        query,
                    )
                unresponsive = last_data.get("unresponsive_engines") if isinstance(last_data, dict) else None
                if unresponsive:
                    logger.info("SearXNG %s search had unresponsive engines for %r: %s", label, query, unresponsive)
            except httpx.HTTPStatusError as exc:
                logger.warning("SearXNG HTTP error during %s attempt: %s", label, exc)
                last_error = f"SearXNG returned HTTP {exc.response.status_code}"
                continue
            except httpx.RequestError as exc:
                logger.warning("SearXNG request error during %s attempt: %s", label, exc)
                last_error = f"Could not reach SearXNG at {base_url}: {exc}"
                break
            except Exception as exc:  # noqa: BLE001 - JSON parse and malformed response fallback to HTML
                logger.warning("SearXNG JSON attempt %s failed: %s", label, exc)
                last_error = "Could not parse SearXNG response as JSON"
                break

        has_usable_raw_results = any(r.get("url") for r in raw_results)
        should_try_html = not has_usable_raw_results and (
            not last_error or last_error.startswith("Could not parse SearXNG response")
        )
        if should_try_html:
            logger.info("SearXNG JSON attempts produced no usable results for %r; trying HTML fallback", query)
            try:
                raw_results = self._request_html(base_url, query, limit)
                if raw_results:
                    last_error = ""
            except httpx.HTTPStatusError as exc:
                logger.warning("SearXNG HTML fallback HTTP error: %s", exc)
                last_error = f"SearXNG returned HTTP {exc.response.status_code}"
            except httpx.RequestError as exc:
                logger.warning("SearXNG HTML fallback request error: %s", exc)
                last_error = f"Could not reach SearXNG at {base_url}: {exc}"
            except Exception as exc:  # noqa: BLE001
                logger.warning("SearXNG HTML fallback parse error: %s", exc)
                last_error = "Could not parse SearXNG response as JSON or HTML"

        if last_error and not raw_results:
            return {"success": False, "error": last_error}

        web_results = self._normalize_results(raw_results, limit)
        logger.info("SearXNG search '%s': %d results (from %d raw, limit %d)", query, len(web_results), len(raw_results), limit)
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
                {
                    "key": "SEARXNG_GENERAL_ENGINES",
                    "prompt": "Optional comma-separated general engines (default: bing,mojeek,presearch; blank disables pinning)",
                    "url": "https://docs.searxng.org/user/search-syntax.html",
                },
            ],
        }
