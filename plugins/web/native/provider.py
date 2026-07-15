"""Local HTTP fetch + readability extract provider — plugin form.

Subclasses the plugin-facing :class:`agent.web_search_provider.WebSearchProvider`.
No API key required — uses httpx for HTTP GET and readability-lxml for
main-content extraction. Extract-only (``supports_search() -> False``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time as time_module
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────
# All behavioral knobs live under ``web.native`` in config.yaml. Defaults
# below mirror ``DEFAULT_CONFIG["web"]["native"]`` and are used verbatim when
# config is unavailable (e.g. a standalone unit-test import).

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)

_NATIVE_DEFAULTS: Dict[str, Any] = {
    "timeout": 30,
    "max_redirects": 5,
    "max_response_bytes": 2000000,
    "max_chars": 50000,
    "max_chars_cap": 200000,
    "cache_ttl": 900,
    "readability": True,
    "use_trusted_proxy": False,
    "user_agent": "",
}

# In-memory cache
_WEB_FETCH_CACHE: Dict[str, tuple[float, str]] = {}


def _load_native_web_config() -> Dict[str, Any]:
    """Read ``web.native`` from config.yaml, merged over built-in defaults."""
    merged = dict(_NATIVE_DEFAULTS)
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web_section = cfg.get("web") if isinstance(cfg, dict) else None
        native_section = web_section.get("native") if isinstance(web_section, dict) else None
        if isinstance(native_section, dict):
            merged.update({k: v for k, v in native_section.items() if v is not None})
    except Exception as exc:  # noqa: BLE001 — config optional; fall back to defaults
        logger.debug("Could not load web.native config: %s", exc)
    return merged


def _cfg_int(cfg: Dict[str, Any], key: str) -> int:
    try:
        return int(cfg.get(key, _NATIVE_DEFAULTS[key]))
    except (TypeError, ValueError):
        return int(_NATIVE_DEFAULTS[key])


def _cfg_bool(cfg: Dict[str, Any], key: str) -> bool:
    val = cfg.get(key, _NATIVE_DEFAULTS[key])
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


def _clamp_max_chars(max_chars: Optional[int], cfg: Dict[str, Any]) -> int:
    if max_chars is None or max_chars < 1:
        max_chars = _cfg_int(cfg, "max_chars")
    return min(max_chars, _cfg_int(cfg, "max_chars_cap"))


_SSRF_BLOCKED_ERROR = "Blocked: URL targets a private or internal network address"


async def _fetch_single_url(
    url: str,
    max_chars: Optional[int] = None,
    extract_mode: str = "markdown",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fetch a single URL and extract readable content."""
    if not url or not isinstance(url, str):
        return {"url": str(url), "title": "", "content": "", "raw_content": "", "error": "URL is required"}

    if cfg is None:
        cfg = _load_native_web_config()

    url = url.strip()
    max_chars = _clamp_max_chars(max_chars, cfg)
    timeout = _cfg_int(cfg, "timeout")
    max_redirects = _cfg_int(cfg, "max_redirects")
    max_response_bytes = _cfg_int(cfg, "max_response_bytes")
    cache_ttl = _cfg_int(cfg, "cache_ttl")
    readability_enabled = _cfg_bool(cfg, "readability")
    user_agent = str(cfg.get("user_agent") or "").strip() or _DEFAULT_USER_AGENT

    # ── SSRF check ────────────────────────────────────────────────
    from tools.url_safety import async_is_safe_url, normalize_url_for_request

    try:
        normalized_url = normalize_url_for_request(url)
    except Exception:
        normalized_url = url
    if not await async_is_safe_url(normalized_url):
        return {
            "url": url, "title": "", "content": "", "raw_content": "",
            "error": _SSRF_BLOCKED_ERROR,
        }

    # ── Cache check ───────────────────────────────────────────────
    cache_key = normalized_url
    cached = _WEB_FETCH_CACHE.get(cache_key)
    if cached and (time_module.monotonic() - cached[0]) < cache_ttl:
        result = cached[1]
        raw = result
        if max_chars and len(result) > max_chars:
            result = result[:max_chars] + "\n\n[... truncated ...]"
        return {"url": url, "content": result, "raw_content": raw}

    # ── Proxy ─────────────────────────────────────────────────────
    proxy_url = None
    if _cfg_bool(cfg, "use_trusted_proxy"):
        proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or None

    # ── Fetch (manual redirect follow so each hop is SSRF-revalidated) ──
    # httpx's built-in follow_redirects would issue requests to redirect
    # targets before we could vet them, so we disable it and walk the chain
    # ourselves, re-checking async_is_safe_url on every Location before the
    # next request. Mirrors the Firecrawl final-URL re-check (2e12401ed).
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            proxy=proxy_url,
        ) as client:
            current_url = normalized_url
            response = None
            for _hop in range(max_redirects + 1):
                response = await client.get(current_url, headers=headers)
                if not response.is_redirect:
                    break
                location = response.headers.get("location")
                if not location:
                    break
                try:
                    next_url = normalize_url_for_request(urljoin(current_url, location))
                except Exception:
                    next_url = urljoin(current_url, location)
                if not await async_is_safe_url(next_url):
                    return {
                        "url": url, "title": "", "content": "", "raw_content": "",
                        "error": _SSRF_BLOCKED_ERROR,
                    }
                current_url = next_url
            else:
                return {
                    "url": url, "title": "", "content": "", "raw_content": "",
                    "error": f"Too many redirects (>{max_redirects})",
                }

            if response is None:
                return {"url": url, "title": "", "content": "", "raw_content": "", "error": "No response"}
            if response.is_redirect:
                return {"url": url, "title": "", "content": "", "raw_content": "", "error": f"Too many redirects (>{max_redirects})"}

            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/" not in content_type and "application/xhtml" not in content_type:
                body = response.text
                raw = body
                if max_chars and len(body) > max_chars:
                    body = body[:max_chars] + "\n\n[... truncated ...]"
                return {"url": url, "content": body, "raw_content": raw, "content_type": content_type}
            html = response.text
    except httpx.TimeoutException:
        return {"url": url, "title": "", "content": "", "raw_content": "", "error": f"Request timed out after {timeout}s"}
    except httpx.HTTPStatusError as e:
        return {"url": url, "title": "", "content": "", "raw_content": "", "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}"}
    except Exception as e:
        return {"url": url, "title": "", "content": "", "raw_content": "", "error": f"Fetch failed: {type(e).__name__}: {e}"}

    # ── Response size check ──────────────────────────────────────
    if len(html) > max_response_bytes:
        html = html[:max_response_bytes]

    # ── Extract readable content ──────────────────────────────────
    readable_title = ""
    readable_html = html

    if readability_enabled:
        try:
            from readability import Document

            doc = Document(html, url=normalized_url)
            readable_title = doc.title()
            readable_html = doc.summary()
        except Exception as e:
            try:
                from lxml import html as lhtml

                tree = lhtml.fromstring(html)
                for tag in tree.xpath("//script|//style|//nav|//footer|//header|//aside"):
                    tag.getparent().remove(tag)
                readable_html = lhtml.tostring(tree, encoding="unicode")
                readable_title = tree.findtext(".//title", default="")
            except Exception:
                return {"url": url, "title": "", "content": "", "raw_content": "", "error": f"Content extraction failed: {e}"}
    else:
        try:
            from lxml import html as lhtml

            tree = lhtml.fromstring(html)
            readable_title = tree.findtext(".//title", default="")
            for tag in tree.xpath("//script|//style"):
                tag.getparent().remove(tag)
            readable_html = lhtml.tostring(tree, encoding="unicode")
        except Exception:
            readable_title = ""
            readable_html = html

    # ── Convert to markdown ───────────────────────────────────────
    try:
        import html2text

        converter = html2text.HTML2Text()
        converter.body_width = 0
        converter.ignore_links = False
        converter.ignore_images = True
        converter.ignore_emphasis = False
        converter.protect_links = True
        converter.unicode_snob = True
        converter.skip_internal_links = True
        if extract_mode == "text":
            converter.ignore_links = True
            converter.ignore_emphasis = True
        content = converter.handle(readable_html)
    except Exception as e:
        return {"url": url, "title": "", "content": "", "raw_content": "", "error": f"Markdown conversion failed: {e}"}

    # ── Clean up ──────────────────────────────────────────────────
    content = re.sub(r"\n{4,}", "\n\n\n", content)
    content = content.strip()
    full_content = f"# {readable_title}\n\n{content}" if readable_title else content
    raw_content = full_content
    if max_chars and len(full_content) > max_chars:
        full_content = full_content[:max_chars] + "\n\n[... truncated ...]"

    _WEB_FETCH_CACHE[cache_key] = (time_module.monotonic(), raw_content)

    return {"url": url, "title": readable_title or "", "content": full_content, "raw_content": raw_content}


class WebFetchWebSearchProvider(WebSearchProvider):
    """Local HTTP fetch extract provider — no API key needed."""

    @property
    def name(self) -> str:
        return "native"

    @property
    def display_name(self) -> str:
        return "Native Web Fetch"

    def is_available(self) -> bool:
        try:
            import readability  # noqa: F401
            import html2text  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        extract_mode = "markdown"
        fmt = kwargs.get("format")
        if fmt and isinstance(fmt, str) and fmt.lower().strip() == "text":
            extract_mode = "text"

        cfg = _load_native_web_config()
        tasks = [_fetch_single_url(u, extract_mode=extract_mode, cfg=cfg) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final: List[Dict[str, Any]] = []
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                final.append({
                    "url": urls[i] if i < len(urls) else "",
                    "title": "", "content": "", "raw_content": "", "error": f"Internal error: {r}",
                })
            else:
                final.append(r)
        return final

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Local Web Fetch (web-fetch)",
            "badge": "free · no key · extract only",
            "tag": "Fetches content via httpx + readability — no API key. Pair with any search provider.",
            "env_vars": [],
        }