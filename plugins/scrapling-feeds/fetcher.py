"""Polite HTTP fetch with guarded Scrapling and HTTPX backends.

Does NOT use StealthyFetcher or bot-evasion paths (defensive OSINT only).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from tools.url_safety import SSRFProtectedTransport, resolve_safe_url_addresses

USER_AGENT = (
    "hermes-gov-feeds/1.0 (+https://github.com/NousResearch/hermes-agent; "
    "government RSS reader)"
)
DEFAULT_TIMEOUT = 25
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_SCRAPLING_HOST_SUFFIXES = (
    "cisa.gov",
    "digital.go.jp",
    "mhlw.go.jp",
    "mod.go.jp",
    "mofa.go.jp",
    "nisc.go.jp",
)


def scrapling_available() -> bool:
    try:
        from scrapling.fetchers import Fetcher  # noqa: F401

        return True
    except ImportError:
        return False


def fetch_url(url: str, *, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Fetch feed or page body. Returns success, body, status, backend."""
    if not resolve_safe_url_addresses(url, allow_private_urls=False):
        return {
            "success": False,
            "error": "Blocked unsafe feed URL",
            "backend": "safety",
        }
    if scrapling_available() and _scrapling_allowed(url):
        result = _fetch_scrapling(url, timeout=timeout)
        if result.get("success"):
            return result
    return _fetch_httpx(url, timeout=timeout)


def _scrapling_allowed(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme.lower() == "https" and any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in _SCRAPLING_HOST_SUFFIXES
    )


def _fetch_scrapling(url: str, *, timeout: int) -> dict[str, Any]:
    try:
        from scrapling.fetchers import Fetcher

        response = Fetcher.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects="safe",
            max_redirects=10,
        )
        status = int(getattr(response, "status", 0) or 0)
        final_url = str(getattr(response, "url", "") or url)
        if not resolve_safe_url_addresses(final_url, allow_private_urls=False):
            return {
                "success": False,
                "error": "Blocked unsafe feed redirect",
                "backend": "scrapling",
            }
        raw = getattr(response, "body", None)
        if raw and len(raw) > MAX_RESPONSE_BYTES:
            return {
                "success": False,
                "error": "Feed response exceeded size limit",
                "status": status or 200,
                "backend": "scrapling",
            }
        if raw:
            text = raw.decode("utf-8", errors="replace")
        else:
            text = getattr(response, "text", None) or ""
        if not text.strip():
            text = str(response)
        if len(text.encode("utf-8", errors="replace")) > MAX_RESPONSE_BYTES:
            return {
                "success": False,
                "error": "Feed response exceeded size limit",
                "status": status or 200,
                "backend": "scrapling",
            }
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


def _fetch_httpx(url: str, *, timeout: int) -> dict[str, Any]:
    try:
        with httpx.Client(
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
            },
            follow_redirects=True,
            timeout=httpx.Timeout(timeout),
            transport=SSRFProtectedTransport(allow_private_urls=False),
        ) as client:
            with client.stream("GET", url) as resp:
                status = resp.status_code
                if status >= 400:
                    return {
                        "success": False,
                        "error": f"HTTP {status}",
                        "status": status,
                        "backend": "httpx",
                    }
                body = bytearray()
                for chunk in resp.iter_bytes():
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        return {
                            "success": False,
                            "error": "Feed response exceeded size limit",
                            "status": status,
                            "backend": "httpx",
                        }
                    body.extend(chunk)
                charset = resp.encoding or "utf-8"
            text = bytes(body).decode(charset, errors="replace")
            return {
                "success": True,
                "body": text,
                "status": status,
                "backend": "httpx",
            }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc)[:300],
            "backend": "httpx",
        }
