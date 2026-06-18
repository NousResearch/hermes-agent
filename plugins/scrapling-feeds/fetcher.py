"""Polite HTTP fetch — Scrapling Fetcher when installed, urllib fallback.

Does NOT use StealthyFetcher or bot-evasion paths (defensive OSINT only).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any

USER_AGENT = (
    "hermes-gov-feeds/1.0 (+https://github.com/NousResearch/hermes-agent; "
    "government RSS reader)"
)
DEFAULT_TIMEOUT = 25


def scrapling_available() -> bool:
    try:
        from scrapling.fetchers import Fetcher  # noqa: F401

        return True
    except ImportError:
        return False


def fetch_url(url: str, *, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Fetch feed or page body. Returns success, body, status, backend."""
    if scrapling_available():
        result = _fetch_scrapling(url, timeout=timeout)
        if result.get("success"):
            return result
    return _fetch_urllib(url, timeout=timeout)


def _fetch_scrapling(url: str, *, timeout: int) -> dict[str, Any]:
    try:
        from scrapling.fetchers import Fetcher

        response = Fetcher.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        status = int(getattr(response, "status", 0) or 0)
        raw = getattr(response, "body", None)
        if raw:
            text = raw.decode("utf-8", errors="replace")
        else:
            text = getattr(response, "text", None) or ""
        if not text.strip():
            text = str(response)
        if status and status >= 400:
            return {
                "success": False,
                "error": f"HTTP {status}",
                "status": status,
                "backend": "scrapling",
            }
        return {
            "success": True,
            "body": text,
            "status": status or 200,
            "backend": "scrapling",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc)[:300],
            "backend": "scrapling",
        }


def _fetch_urllib(url: str, *, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            text = body.decode(charset, errors="replace")
            return {
                "success": True,
                "body": text,
                "status": getattr(resp, "status", 200),
                "backend": "urllib",
            }
    except urllib.error.HTTPError as exc:
        return {
            "success": False,
            "error": f"HTTP {exc.code}: {exc.reason}",
            "status": exc.code,
            "backend": "urllib",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc)[:300],
            "backend": "urllib",
        }
