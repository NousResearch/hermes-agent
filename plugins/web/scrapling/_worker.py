"""Scrapling sidecar worker — runs under the sidecar venv's own interpreter.

Reads one JSON request from stdin, writes one JSON envelope to stdout:

    request  {"op": "search", "query": str, "limit": int}
             {"op": "extract", "urls": [str, ...]}
    response {"ok": true, "results": [...]} | {"ok": false, "error": str}

The worker is deliberately self-contained: it imports nothing from the Hermes
runtime, because the two live in different Python environments.
"""

from __future__ import annotations

import json
import os
import re
import sys

# The DDG Lite parser is shared with the Hermes-side provider, but the two run
# in different Python environments: import it by path, not as a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ddg import parse_ddg_lite_results

# DDG Lite serves anonymous clients; its robots.txt is "Allow: /".
SEARCH_URL = "https://lite.duckduckgo.com/lite/"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT_S = 25

# A body shorter than this is a JS shell rather than content (measured: the
# DDG-style JS shell returns ~90 chars over HTTP and ~1500 once rendered).
THIN_BODY_CHARS = 300
JS_TIMEOUT_S = 45
JS_SETTLE_MS = 1200

_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _clean_text(page) -> str:
    try:
        text = page.get_all_text()
    except Exception:
        text = ""
    lines = [line.strip() for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _page_title(page) -> str:
    try:
        titles = page.css("title::text")
        if titles:
            first = titles[0]
            if isinstance(first, (list, tuple)):
                first = first[0] if first else ""
            if first:
                return str(first).strip()
    except Exception:
        pass
    return ""


def _render_with_js(url: str) -> str:
    """Headless-Chromium render, used only when the HTTP body is a JS shell."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=USER_AGENT)
                page.goto(url, wait_until="domcontentloaded", timeout=int(JS_TIMEOUT_S * 1000))
                if JS_SETTLE_MS > 0:
                    page.wait_for_timeout(JS_SETTLE_MS)
                text = page.inner_text("body")
            finally:
                browser.close()
    except Exception:
        return ""
    lines = [line.strip() for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _do_search(query: str, limit: int) -> list[dict]:
    from scrapling.fetchers import Fetcher

    page = Fetcher().get(
        SEARCH_URL, params={"q": query}, timeout=REQUEST_TIMEOUT_S,
        stealthy_headers=True, headers={"User-Agent": USER_AGENT},
    )
    status = int(getattr(page, "status", 0) or 0)
    html = getattr(page, "html_content", "") or ""
    if status and status >= 400:
        raise RuntimeError(f"HTTP {status} from search endpoint")
    rows = parse_ddg_lite_results(html)
    return rows[: max(1, int(limit))] if limit else rows


def _do_extract(urls: list[str]) -> list[dict]:
    from scrapling.fetchers import Fetcher

    results = []
    for url in urls:
        try:
            page = Fetcher().get(
                url, timeout=REQUEST_TIMEOUT_S, stealthy_headers=True,
                headers={"User-Agent": USER_AGENT},
            )
        except Exception as exc:
            results.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue
        status = int(getattr(page, "status", 0) or 0)
        final_url = str(getattr(page, "url", url) or url)
        title = _page_title(page)
        text = _clean_text(page)
        if status and status >= 400:
            results.append({"url": final_url, "requested_url": url, "error": f"HTTP {status}"})
            continue
        if len(text) < THIN_BODY_CHARS:
            rendered = _render_with_js(url)
            if rendered:
                text = rendered
        results.append({"url": final_url, "requested_url": url, "title": title, "text": text})
    return results


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        json.dump({"ok": False, "error": f"invalid request: {exc}"}, sys.stdout)
        return 2
    try:
        op = str(request.get("op") or "")
        if op == "search":
            results = _do_search(str(request.get("query") or ""), int(request.get("limit") or 5))
        elif op == "extract":
            results = _do_extract([str(u) for u in (request.get("urls") or [])])
        else:
            raise ValueError(f"unknown op {op!r}")
    except Exception as exc:
        json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 1
    json.dump({"ok": True, "results": results}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
