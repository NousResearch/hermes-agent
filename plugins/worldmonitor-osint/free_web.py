"""Free-tier World Monitor collection via public web API (browser-like HTTP).

World Monitor's web app (https://worldmonitor.app) exposes several JSON
endpoints without Pro OAuth or wm_ keys. Intelligence briefs remain Pro-only;
this module collects what the Free web tier can access.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

FREE_WEB_ORIGIN = "https://worldmonitor.app"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Paths reachable without Pro auth (verified against worldmonitor.app).
FREE_JSON_ROUTES: dict[str, str] = {
    "news_digest": "/api/news/v1/list-feed-digest",
    "gpsjam": "/api/gpsjam",
    "oref_alerts": "/api/oref-alerts",
    "version": "/api/version",
}


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta: dict[str, str] = {}
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = attr.get("name") or attr.get("property") or ""
            content = attr.get("content") or ""
            if name and content:
                self.meta[name] = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def fetch_json(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 45.0,
) -> dict[str, Any]:
    """GET JSON from worldmonitor.app with browser-like headers."""
    query = ""
    if params:
        filtered = {k: v for k, v in params.items() if v is not None and v != ""}
        if filtered:
            query = "?" + urllib.parse.urlencode(filtered, doseq=True)
    url = f"{FREE_WEB_ORIGIN}{path}{query}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
            "User-Agent": BROWSER_UA,
            "Referer": f"{FREE_WEB_ORIGIN}/",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1500]
        raise RuntimeError(
            json.dumps(
                {
                    "success": False,
                    "http_status": exc.code,
                    "url": url,
                    "error": detail or exc.reason,
                    "tier": "free_web",
                },
                ensure_ascii=False,
            )
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            json.dumps(
                {"success": False, "url": url, "error": str(exc.reason or exc), "tier": "free_web"},
                ensure_ascii=False,
            )
        ) from exc

    if not body.strip():
        return {}
    parsed = json.loads(body)
    if isinstance(parsed, dict):
        return parsed
    return {"data": parsed}


def crawl_app_shell(*, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch the public app HTML and extract title / OpenGraph metadata."""
    url = f"{FREE_WEB_ORIGIN}/"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    parser = _MetaParser()
    parser.feed(html)
    api_hints = sorted(set(re.findall(r"/api/[a-zA-Z0-9_\-/]+", html)))

    return {
        "url": url,
        "title": parser.title.strip(),
        "meta": parser.meta,
        "html_bytes": len(html),
        "api_paths_in_html": api_hints[:50],
    }


def probe_free_tier(*, timeout: float = 15.0) -> dict[str, Any]:
    """Check whether the Free web JSON tier responds."""
    try:
        version = fetch_json(FREE_JSON_ROUTES["version"], timeout=timeout)
        return {
            "available": True,
            "origin": FREE_WEB_ORIGIN,
            "version": version,
            "routes": list(FREE_JSON_ROUTES),
        }
    except Exception as exc:
        return {
            "available": False,
            "origin": FREE_WEB_ORIGIN,
            "error": str(exc),
            "routes": list(FREE_JSON_ROUTES),
        }


def _news_headlines(digest: dict[str, Any], *, limit: int, focus: str) -> list[dict[str, Any]]:
    """Flatten category digest into headline rows; bias toward Japan/security when focus set."""
    focus_keys = ("politics", "middleeast", "gov", "tech", "ai", "finance", "europe", "us")
    if focus == "japan_security":
        focus_keys = ("politics", "middleeast", "gov", "tech", "ai", "us", "europe", "finance")

    categories = digest.get("categories") if isinstance(digest.get("categories"), dict) else {}
    rows: list[dict[str, Any]] = []
    for cat in focus_keys:
        bucket = categories.get(cat)
        if not isinstance(bucket, dict):
            continue
        items = bucket.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["category"] = cat
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                return rows
    return rows[:limit] if limit > 0 else rows


def free_snapshot(
    *,
    focus: str = "japan_security",
    news_lang: str = "en",
    news_limit: int = 20,
    include_shell: bool = True,
) -> dict[str, Any]:
    """Aggregate Free-tier World Monitor feeds (no Pro key / OAuth)."""
    out: dict[str, Any] = {
        "success": False,
        "tier": "free_web",
        "focus": focus,
        "origin": FREE_WEB_ORIGIN,
        "sections": {},
        "errors": [],
        "pro_only_skipped": [
            "country_risk",
            "country_intel_brief",
            "regional_brief",
            "risk_scores",
        ],
    }

    fetches: list[tuple[str, str, dict[str, Any] | None]] = [
        ("news_digest", FREE_JSON_ROUTES["news_digest"], {"variant": "full", "lang": news_lang}),
        ("gpsjam", FREE_JSON_ROUTES["gpsjam"], None),
        ("oref_alerts", FREE_JSON_ROUTES["oref_alerts"], None),
        ("version", FREE_JSON_ROUTES["version"], None),
    ]
    for key, path, params in fetches:
        try:
            out["sections"][key] = fetch_json(path, params)
        except Exception as exc:
            out["errors"].append({"section": key, "error": str(exc)})

    digest = out["sections"].get("news_digest") or {}
    if isinstance(digest, dict):
        out["news_headlines"] = _news_headlines(digest, limit=news_limit, focus=focus)

    if include_shell:
        try:
            out["sections"]["app_shell"] = crawl_app_shell()
        except Exception as exc:
            out["errors"].append({"section": "app_shell", "error": str(exc)})

    out["success"] = bool(out["sections"])
    out["free_tier_probe"] = probe_free_tier()
    return out
