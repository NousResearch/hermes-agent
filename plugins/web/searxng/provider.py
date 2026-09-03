"""SearXNG search provider.

Searches a self-hosted SearXNG instance through its plain-HTML results
page (``GET /search?q=...``). This instance family (v2026+) returns 403
for ``format=json`` and for POST with browser UAs, so the HTML results
page is parsed directly. Search-only: ``web_extract`` is not supported
(pair with firecrawl/tavily/exa for extraction).

Instance base URL comes from ``SEARXNG_URL`` (config-aware env lookup).
"""

from __future__ import annotations

import html as _htmlmod
import logging
import re
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider, get_provider_env

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 "
    "Safari/537.36"
)
_TIMEOUT = 15.0

# One search-result card in the v2026+ SPA results page.
_ARTICLE_RE = re.compile(r'<article class="result[^"]*".*?</article>', re.S)
# Title + URL live in <h3><a href="...">TITLE</a></h3> (title may nest
# <span class="highlight">…</span>).
_H3_LINK_RE = re.compile(r"<h3>.*?<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.S)
# Description lives in <p class="content">…</p> when present.
_CONTENT_RE = re.compile(r'<p class="content">(.*?)</p>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")
# Markers on the no-hits page (instance answered but found nothing).
_NO_RESULTS_MARKERS = ("No results were found", "response-error")


def _clean(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


class SearXNGWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by a self-hosted SearXNG instance."""

    @property
    def name(self) -> str:
        return "searxng"

    def is_available(self) -> bool:
        return bool(get_provider_env("SEARXNG_URL"))

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        base = get_provider_env("SEARXNG_URL").rstrip("/")
        if not base:
            return {"success": False, "error": "SEARXNG_URL is not set."}
        try:
            resp = httpx.get(
                f"{base}/search",
                params={"q": query, "safesearch": "0"},
                headers={
                    "User-Agent": _UA,
                    "Accept": (
                        "text/html,application/xhtml+xml,application/xml;"
                        "q=0.9,image/webp,*/*;q=0.8"
                    ),
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            return {"success": False, "error": f"SearXNG request failed: {exc}"}
        if resp.status_code != 200:
            return {"success": False, "error": f"SearXNG returned HTTP {resp.status_code}"}

        results: List[Dict[str, Any]] = []
        for article in _ARTICLE_RE.findall(resp.text):
            m = _H3_LINK_RE.search(article)
            if not m:
                continue
            url_, title_html = m.group(1), m.group(2)
            title = _htmlmod.unescape(_clean(title_html))
            if not title:
                continue
            desc = ""
            cm = _CONTENT_RE.search(article)
            if cm:
                desc = _htmlmod.unescape(_clean(cm.group(1)))
            results.append(
                {
                    "title": title,
                    "url": url_,
                    "description": desc,
                    "position": len(results) + 1,
                }
            )
            if len(results) >= limit:
                break

        if not results and any(marker in resp.text for marker in _NO_RESULTS_MARKERS):
            # The instance answered but had no hits — a valid empty result.
            return {"success": True, "data": {"web": []}}

        return {"success": True, "data": {"web": results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SearXNG (self-hosted)",
            "badge": "free",
            "tag": "Search your own SearXNG instance — no API key needed.",
            "env_vars": [
                {
                    "key": "SEARXNG_URL",
                    "prompt": "SearXNG instance base URL",
                    "url": "https://docs.searxng.org/",
                },
            ],
        }
