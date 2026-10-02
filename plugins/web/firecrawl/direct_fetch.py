"""Bounded public HTML recovery when a self-hosted scraper cannot connect.

This is a single ordinary HTTP fetch, without browser execution, authentication,
retries, or bypassing access controls. It uses Hermes' existing URL/connect guard.
"""
from __future__ import annotations

import asyncio
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit

from tools.url_safety import async_is_safe_url, create_ssrf_safe_async_client, normalize_url_for_request
from tools.website_policy import check_website_access

_MAX_BYTES = 2 * 1024 * 1024
_TIMEOUT = 20.0
_MAX_HTML_DEPTH = 256
_SKIP = {"script", "style", "noscript", "nav", "header", "footer", "svg", "template"}
_BLOCKS = {"p", "div", "section", "article", "main", "li", "br", "h1", "h2", "h3", "h4", "pre", "tr"}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class _PageText(HTMLParser):
    """Prefer article/main text; omit scripts and site chrome from body fallback."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.body, self.main, self.title = [], [], []
        self.password = False

    def handle_starttag(self, tag, attrs):
        if tag == "input" and dict(attrs).get("type", "").lower() == "password":
            self.password = True
        if tag in _BLOCKS:
            self.handle_data("\n")
        if tag not in _VOID:
            if len(self.stack) >= _MAX_HTML_DEPTH:
                raise ValueError("HTML nesting exceeds direct extraction limit")
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.stack:
            self.stack = self.stack[:len(self.stack) - 1 - self.stack[::-1].index(tag)]
        if tag in _BLOCKS:
            self.handle_data("\n")

    def handle_data(self, data):
        if "title" in self.stack:
            self.title.append(data)
        if any(tag in _SKIP for tag in self.stack) or "head" in self.stack:
            return
        self.body.append(data)
        if "main" in self.stack or "article" in self.stack:
            self.main.append(data)

    def text(self):
        chunks = self.main if self.main else self.body
        return "\n".join(line for raw in "".join(chunks).splitlines() if (line := " ".join(raw.split())))


async def recover_html(url: str, format: str | None, backend_error: str) -> dict | None:
    """Return content with provenance, or None so the original backend error survives."""
    from tools.interrupt import is_interrupted

    async def fetch():
        from agent.redact import _PREFIX_RE
        current, seen = normalize_url_for_request(url), set()
        async with create_ssrf_safe_async_client(timeout=_TIMEOUT, follow_redirects=False) as client:
            for _ in range(6):
                parsed = urlsplit(current)
                if (parsed.scheme not in {"http", "https"} or parsed.username is not None
                        or _PREFIX_RE.search(unquote(current)) or is_interrupted() or current in seen
                        or not await async_is_safe_url(current) or check_website_access(current)):
                    return None
                seen.add(current)
                client.cookies.clear()
                async with client.stream("GET", current, headers={"Accept": "text/html"}) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            return None
                        target = normalize_url_for_request(urljoin(current, location))
                        if current.startswith("https:") and not target.startswith("https:"):
                            return None
                        current = target
                        continue
                    if response.status_code != 200 or response.headers.get("content-type", "").split(";", 1)[0].strip().lower() not in {"text/html", "application/xhtml+xml"}:
                        return None
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > _MAX_BYTES or is_interrupted():
                            return None
                    html = bytes(body).decode(response.encoding or "utf-8", errors="replace")
                page = _PageText()
                page.feed(html)
                text, title = page.text(), " ".join("".join(page.title).split())
                # A 200 challenge/login/shell is not extracted article content.
                challenge = (title + " " + text[:1000]).lower()
                if page.password or len(text) < 100 or any(marker in challenge for marker in (
                    "just a moment", "verify you are human", "enable javascript and cookies", "checking your browser", "access denied",
                )):
                    return None
                content = html if format == "html" else text
                return {"url": current, "title": title, "content": content, "raw_content": content,
                        "metadata": {"sourceURL": url, "rescued_from": "firecrawl", "extraction_method": "direct_html",
                                     "backend_error": backend_error[:300]}}
        return None

    try:
        return await asyncio.wait_for(fetch(), timeout=_TIMEOUT)
    except Exception:  # Best effort: retain the scraper's original diagnostic.
        return None
