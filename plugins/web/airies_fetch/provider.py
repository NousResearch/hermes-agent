"""AIRIES Fetch — non-automated web data retrieval.

Uses plain HTTP requests and HTML parsing only. No headless browsers,
no Firecrawl agents, no JavaScript rendering. Search hits DuckDuckGo's
static HTML endpoint; extract fetches page HTML and strips tags locally.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.parse import quote_plus, urljoin, urlparse

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_USER_AGENT = "AIRIES-Agent/1.0 (+https://github.com/FounderOfFluxLM/AIRIES-AGENT)"


class _TextExtractor(HTMLParser):
    """Collect visible text, skipping script/style blocks."""

    def __init__(self):
        super().__init__()
        self._skip = 0
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self._chunks))


class _DDGResultParser(HTMLParser):
    """Parse DuckDuckGo HTML lite search results."""

    def __init__(self):
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._in_result = False
        self._in_title = False
        self._in_snippet = False
        self._title = ""
        self._url = ""
        self._snippet = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")
        if tag == "div" and "result" in cls:
            self._in_result = True
            self._title = ""
            self._url = ""
            self._snippet = ""
        if self._in_result and tag == "a" and "result__a" in cls:
            self._in_title = True
            self._url = attrs_d.get("href", "")
        if self._in_result and tag == "a" and "result__snippet" in cls:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
        if tag == "a" and self._in_snippet:
            self._in_snippet = False
        if tag == "div" and self._in_result:
            if self._url and self._title:
                self.results.append(
                    {
                        "title": self._title.strip(),
                        "url": self._url.strip(),
                        "description": self._snippet.strip(),
                    }
                )
            self._in_result = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title += data
        if self._in_snippet:
            self._snippet += data


class AriesFetchProvider(WebSearchProvider):
    """Manual HTTP fetch + HTML parse — no browser automation."""

    @property
    def name(self) -> str:
        return "airies_fetch"

    @property
    def display_name(self) -> str:
        return "AIRIES Fetch (non-auto)"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        q = (query or "").strip()
        if not q:
            return {"success": False, "error": "Query is required"}

        try:
            from hermes_cli.airies_subscription import get_subscription_manager

            ok, msg = get_subscription_manager().check_web_fetch_allowed()
            if not ok:
                return {"success": False, "error": msg}
        except Exception:
            pass

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=20,
                follow_redirects=True,
            )
            resp.raise_for_status()
        except Exception as exc:
            return {"success": False, "error": f"Search fetch failed: {exc}"}

        parser = _DDGResultParser()
        try:
            parser.feed(resp.text)
        except Exception as exc:
            return {"success": False, "error": f"Could not parse search HTML: {exc}"}

        web = []
        for i, hit in enumerate(parser.results[: max(1, int(limit))]):
            web.append(
                {
                    "title": hit["title"],
                    "url": hit["url"],
                    "description": hit["description"],
                    "position": i + 1,
                }
            )

        try:
            from hermes_cli.airies_subscription import get_subscription_manager

            get_subscription_manager().record_event("web_fetch")
        except Exception:
            pass

        return {"success": True, "data": {"web": web}}

    def extract(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        import httpx

        if not urls:
            return {"success": False, "error": "At least one URL is required"}

        try:
            from hermes_cli.airies_subscription import get_subscription_manager

            ok, msg = get_subscription_manager().check_web_fetch_allowed()
            if not ok:
                return {"success": False, "error": msg}
        except Exception:
            pass

        data = []
        for raw_url in urls[:5]:
            url = (raw_url or "").strip()
            if not url:
                continue
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                data.append({"url": url, "title": "", "content": "", "error": "Only http(s) URLs supported"})
                continue
            try:
                resp = httpx.get(
                    url,
                    headers={"User-Agent": _USER_AGENT},
                    timeout=25,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                extractor = _TextExtractor()
                extractor.feed(resp.text)
                text = extractor.text()
                title_match = re.search(r"<title[^>]*>(.*?)</title>", resp.text, re.I | re.S)
                title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else url
                data.append(
                    {
                        "url": str(resp.url),
                        "title": title,
                        "content": text[:15000],
                        "raw_content": text,
                        "metadata": {"content_type": resp.headers.get("content-type", "")},
                    }
                )
            except Exception as exc:
                data.append({"url": url, "title": "", "content": "", "error": str(exc)})

        try:
            from hermes_cli.airies_subscription import get_subscription_manager

            get_subscription_manager().record_event("web_fetch")
        except Exception:
            pass

        if not data:
            return {"success": False, "error": "No URLs could be fetched"}
        return {"success": True, "data": data}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "AIRIES Fetch",
            "badge": "free · non-auto · no key",
            "tag": "Plain HTTP + HTML parsing. No browser automation or JS rendering.",
            "env_vars": [],
        }
