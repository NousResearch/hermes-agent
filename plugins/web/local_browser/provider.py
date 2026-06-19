"""Local browser-backed web extraction provider.

This provider intentionally avoids paid scraper/search APIs. It renders pages
with the local ``agent-browser`` CLI, extracts title/body text/HTML from the DOM,
and falls back to a simple urllib + BeautifulSoup/markdownify static extraction
when the browser path fails.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import uuid
from typing import Any, Dict, List
from urllib.request import Request, urlopen

from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_BROWSER_TIMEOUT_SECONDS = 60
_STATIC_TIMEOUT_SECONDS = 25
_MAX_RAW_CHARS = 500_000
_USER_AGENT = "HermesLocalExtract/1.0 (+https://github.com/NousResearch/hermes-agent)"


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def _html_to_markdown(html: str, fallback_text: str = "") -> str:
    if not html:
        return fallback_text
    try:
        from markdownify import markdownify as md

        content = md(html, heading_style="ATX", strip=["script", "style", "noscript"])
        # Keep line structure but strip excessive blank lines/space.
        lines = [line.rstrip() for line in content.splitlines()]
        compact: list[str] = []
        blank = False
        for line in lines:
            if line.strip():
                compact.append(line)
                blank = False
            elif not blank:
                compact.append("")
                blank = True
        return "\n".join(compact).strip() or fallback_text
    except Exception as exc:  # noqa: BLE001
        logger.debug("markdownify failed: %s", exc)
        return fallback_text


def _static_extract(url: str, fmt: str | None = None) -> Dict[str, Any]:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=_STATIC_TIMEOUT_SECONDS) as response:  # noqa: S310 - SSRF checked before call
        content_type = response.headers.get("content-type", "")
        raw = response.read(_MAX_RAW_CHARS + 1)
        if len(raw) > _MAX_RAW_CHARS:
            raw = raw[:_MAX_RAW_CHARS]
        charset = response.headers.get_content_charset() or "utf-8"
    html = raw.decode(charset, "replace")

    title = ""
    text = ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
            tag.decompose()
        title = _normalize_text(soup.title.get_text(" ")) if soup.title else ""
        text = soup.get_text("\n", strip=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("BeautifulSoup extraction failed: %s", exc)
        text = _normalize_text(html)

    content = html if fmt == "html" else _html_to_markdown(html, text)
    return {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": content,
        "metadata": {
            "source": "local-static",
            "content_type": content_type,
            "html_length": len(html),
        },
    }


def _run_agent_browser(args: list[str], *, input_text: str | None = None, timeout: int = _BROWSER_TIMEOUT_SECONDS) -> str:
    proc = subprocess.run(
        ["agent-browser", *args],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
        raise RuntimeError(detail[-1000:])
    return proc.stdout


def _decode_agent_browser_eval(stdout: str) -> Dict[str, Any]:
    last = stdout.strip().splitlines()[-1] if stdout.strip() else "{}"
    data: Any = json.loads(last)
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError(f"unexpected eval result type: {type(data).__name__}")
    return data


def _browser_extract(url: str, fmt: str | None = None) -> Dict[str, Any]:
    session = f"web-extract-{uuid.uuid4().hex[:10]}"
    try:
        _run_agent_browser(["--session", session, "open", url], timeout=_BROWSER_TIMEOUT_SECONDS)
        try:
            _run_agent_browser(["--session", session, "wait", "--load", "networkidle"], timeout=35)
        except Exception:
            # Some sites never settle; a short deterministic wait is better
            # than burning the whole tool timeout.
            _run_agent_browser(["--session", session, "wait", "1500"], timeout=10)

        js = """
(() => JSON.stringify({
  title: document.title || '',
  url: location.href,
  text: document.body ? document.body.innerText : '',
  html: document.documentElement ? document.documentElement.outerHTML : ''
}))()
""".strip()
        data = _decode_agent_browser_eval(
            _run_agent_browser(["--session", session, "eval", "--stdin"], input_text=js)
        )
        text = data.get("text") or ""
        html = data.get("html") or ""
        content = html if fmt == "html" else _html_to_markdown(html, text)
        return {
            "url": data.get("url") or url,
            "title": _normalize_text(data.get("title") or ""),
            "content": content,
            "raw_content": content,
            "metadata": {
                "source": "local-agent-browser",
                "rendered": True,
                "text_length": len(text),
                "html_length": len(html),
            },
        }
    finally:
        try:
            _run_agent_browser(["--session", session, "close"], timeout=15)
        except Exception:
            pass


class LocalBrowserExtractProvider(WebSearchProvider):
    """Local web_extract provider using agent-browser with static fallback."""

    @property
    def name(self) -> str:
        return "local-browser"

    @property
    def display_name(self) -> str:
        return "Local Browser"

    def is_available(self) -> bool:
        return shutil.which("agent-browser") is not None

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        fmt = (kwargs.get("format") or "markdown").lower().strip()
        if fmt not in {"markdown", "html"}:
            fmt = "markdown"

        results: list[dict[str, Any]] = []
        for url in urls:
            started = time.monotonic()
            if not is_safe_url(url):
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": "Blocked: URL targets a private or internal network address",
                })
                continue
            blocked = check_website_access(url)
            if blocked:
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": blocked["message"],
                    "blocked_by_policy": {
                        "host": blocked["host"],
                        "rule": blocked["rule"],
                        "source": blocked["source"],
                    },
                })
                continue

            try:
                result = _browser_extract(url, fmt=fmt)
                result.setdefault("metadata", {})["elapsed_seconds"] = round(time.monotonic() - started, 2)
                results.append(result)
            except Exception as browser_exc:  # noqa: BLE001
                logger.warning("local-browser extract failed for %s: %s; trying static fallback", url, browser_exc)
                try:
                    result = _static_extract(url, fmt=fmt)
                    result.setdefault("metadata", {})["browser_error"] = str(browser_exc)[:300]
                    result["metadata"]["elapsed_seconds"] = round(time.monotonic() - started, 2)
                    results.append(result)
                except Exception as static_exc:  # noqa: BLE001
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": (
                            "Local browser extraction failed: "
                            f"{str(browser_exc)[:300]}; static fallback failed: {str(static_exc)[:300]}"
                        ),
                    })
        return results
