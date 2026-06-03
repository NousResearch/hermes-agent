"""Browser-tools sidecar service.

A small FastAPI service that gives Hermes agents one shared place for heavy
browser/scraping dependencies. It combines:

- CloakBrowser CLI/runtime for stealth rendering, humanized browsing, and
  screenshots when available.
- Scrapling for fast HTTP fetching, CSS/XPath extraction, and future crawling
  when available.
- urllib fallback for basic public pages when neither optional runtime exists.

The API is intentionally simple JSON so any Hermes profile, worker, or external
agent can call it over an internal Docker network.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Hermes browser-tools sidecar", version="0.1.0")

MAX_TIMEOUT_MS = 120_000
MAX_TEXT_LIMIT = 50_000
CLOAK_FETCH = os.getenv("CLOAK_FETCH", "/opt/data/bin/cloak-fetch")


def _clamp(value: int | None, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _capabilities() -> dict[str, bool]:
    def has_module(name: str) -> bool:
        try:
            __import__(name)
            return True
        except Exception:
            return False

    return {
        "cloak_fetch_cli": Path(CLOAK_FETCH).exists() or bool(shutil.which("cloak-fetch")),
        "scrapling": has_module("scrapling"),
        "playwright": has_module("playwright"),
        "patchright": has_module("patchright"),
    }


def _plain_http(url: str, timeout_ms: int, text_limit: int) -> dict[str, Any]:
    started = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "HermesBrowserTools/0.1"})
    with urllib.request.urlopen(req, timeout=(timeout_ms / 1000.0)) as resp:
        raw = resp.read(2_000_000)
        final_url = resp.geturl()
        status = getattr(resp, "status", None)
        content_type = resp.headers.get_content_charset() or "utf-8"
    body = raw.decode(content_type, errors="replace")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    text = re.sub(r"\s+", " ", text).strip()
    return {
        "success": True,
        "mode_used": "plain_http",
        "url": url,
        "final_url": final_url,
        "title": title,
        "response_status": status,
        "html_length": len(body),
        "text_length": len(text),
        "text": text[:text_limit],
        "text_truncated": len(text) > text_limit,
        "duration_s": round(time.time() - started, 3),
    }


def _scrapling_http(url: str, timeout_ms: int, text_limit: int) -> dict[str, Any]:
    started = time.time()
    from scrapling.fetchers import Fetcher

    page = Fetcher.get(url, timeout=max(1, int(timeout_ms / 1000)))
    text = page.css("body::text").get(default="") or page.text
    text = re.sub(r"\s+", " ", text or "").strip()
    title = page.css("title::text").get(default="") or ""
    status = getattr(page, "status", None) or getattr(page, "status_code", None)
    return {
        "success": True,
        "mode_used": "scrapling_http",
        "url": url,
        "final_url": getattr(page, "url", url),
        "title": title,
        "response_status": status,
        "html_length": len(str(page.body)) if getattr(page, "body", None) is not None else None,
        "text_length": len(text),
        "text": text[:text_limit],
        "text_truncated": len(text) > text_limit,
        "duration_s": round(time.time() - started, 3),
    }


def _cloak_fetch(url: str, timeout_ms: int, text_limit: int, screenshot: bool, humanize: bool, headless: bool = True) -> dict[str, Any]:
    started = time.time()
    exe = CLOAK_FETCH if Path(CLOAK_FETCH).exists() else shutil.which("cloak-fetch")
    if not exe:
        raise RuntimeError("cloak-fetch executable not found")
    cmd = [exe, url, "--timeout-ms", str(timeout_ms), "--text-limit", str(text_limit), "--headless", "true" if headless else "false"]
    if humanize:
        cmd.append("--humanize")
    if screenshot:
        out_dir = Path(os.getenv("BROWSER_TOOLS_ARTIFACT_DIR", "/tmp/browser-tools-artifacts"))
        out_dir.mkdir(parents=True, exist_ok=True)
        shot = out_dir / f"screenshot-{int(time.time() * 1000)}.png"
        cmd.extend(["--screenshot", str(shot)])
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=(timeout_ms / 1000.0) + 30)
    if not proc.stdout.strip():
        return {"success": False, "mode_used": "cloak", "error": proc.stderr.strip() or "no output", "exit_code": proc.returncode}
    data = json.loads(proc.stdout)
    data["mode_used"] = "cloak"
    data["duration_s"] = round(time.time() - started, 3)
    data.setdefault("exit_code", proc.returncode)
    return data


def _select_fetch_mode(requested: str, wants_screenshot: bool) -> list[str]:
    if requested and requested != "auto":
        return [requested]
    if wants_screenshot:
        return ["cloak", "scrapling_dynamic", "scrapling_http", "plain_http"]
    return ["scrapling_http", "cloak", "plain_http"]


class FetchRequest(BaseModel):
    url: str
    mode: Literal["auto", "cloak", "scrapling_http", "scrapling_dynamic", "scrapling_stealth", "plain_http"] = "auto"
    timeout_ms: int = Field(30_000, ge=1_000, le=MAX_TIMEOUT_MS)
    text_limit: int = Field(12_000, ge=0, le=MAX_TEXT_LIMIT)
    screenshot: bool = False
    humanize: bool = False
    headless: bool = True
    wait_until: str = "domcontentloaded"


class ExtractRequest(BaseModel):
    url: str
    mode: Literal["auto", "scrapling_http", "scrapling_dynamic", "scrapling_stealth"] = "scrapling_http"
    selectors: dict[str, str] = Field(default_factory=dict)
    timeout_ms: int = Field(30_000, ge=1_000, le=MAX_TIMEOUT_MS)
    text_limit: int = Field(12_000, ge=0, le=MAX_TEXT_LIMIT)
    include_links: bool = True


# Pydantic v2 can need an explicit rebuild when this file is imported through
# importlib in tests instead of normal module loading.
FetchRequest.model_rebuild()
ExtractRequest.model_rebuild()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "capabilities": _capabilities()}


@app.post("/fetch")
def fetch(req: FetchRequest) -> dict[str, Any]:
    timeout_ms = _clamp(req.timeout_ms, 30_000, 1_000, MAX_TIMEOUT_MS)
    text_limit = _clamp(req.text_limit, 12_000, 0, MAX_TEXT_LIMIT)
    errors: list[dict[str, str]] = []
    for mode in _select_fetch_mode(req.mode, req.screenshot):
        try:
            if mode == "cloak":
                return _cloak_fetch(req.url, timeout_ms, text_limit, req.screenshot, req.humanize, req.headless)
            if mode == "scrapling_http":
                return _scrapling_http(req.url, timeout_ms, text_limit)
            if mode in {"scrapling_dynamic", "scrapling_stealth"}:
                # Kept explicit for routing clarity. Implementations require browser
                # installs and can be enabled later without changing the client API.
                raise RuntimeError(f"{mode} requires browser runtime setup; use cloak or scrapling_http for now")
            if mode == "plain_http":
                return _plain_http(req.url, timeout_ms, text_limit)
        except Exception as e:
            errors.append({"mode": mode, "error": str(e), "error_type": type(e).__name__})
    return {"success": False, "url": req.url, "errors": errors, "capabilities": _capabilities()}


def _selector_values(page: Any, selector: str) -> list[str]:
    if selector.startswith("xpath:"):
        values = page.xpath(selector.removeprefix("xpath:")).getall()
    else:
        values = page.css(selector).getall()
    return [str(v).strip() for v in values if str(v).strip()]


@app.post("/extract")
def extract(req: ExtractRequest) -> dict[str, Any]:
    timeout_ms = _clamp(req.timeout_ms, 30_000, 1_000, MAX_TIMEOUT_MS)
    text_limit = _clamp(req.text_limit, 12_000, 0, MAX_TEXT_LIMIT)
    try:
        from scrapling.fetchers import Fetcher

        page = Fetcher.get(req.url, timeout=max(1, int(timeout_ms / 1000)))
        fields = {name: _selector_values(page, selector) for name, selector in req.selectors.items()}
        text = re.sub(r"\s+", " ", (page.css("body::text").get(default="") or page.text or "")).strip()
        links: list[str] = []
        if req.include_links:
            base = getattr(page, "url", req.url)
            links = [urllib.parse.urljoin(base, href) for href in page.css("a::attr(href)").getall() if href]
        return {
            "success": True,
            "mode_used": "scrapling_http",
            "url": req.url,
            "final_url": getattr(page, "url", req.url),
            "title": page.css("title::text").get(default="") or "",
            "response_status": getattr(page, "status", None) or getattr(page, "status_code", None),
            "fields": fields,
            "links": links,
            "text": text[:text_limit],
            "text_length": len(text),
            "text_truncated": len(text) > text_limit,
        }
    except Exception as e:
        return {"success": False, "url": req.url, "error": str(e), "error_type": type(e).__name__, "capabilities": _capabilities()}
