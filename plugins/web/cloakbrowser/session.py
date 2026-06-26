"""Shared CloakBrowser launch helpers for the Hermes web-search plugin."""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT_SECS = 30
_EXTRACT_TIMEOUT_SECS = 60
_DEFAULT_GOTO_WAIT = "domcontentloaded"
_MAX_BODY_CHARS = 120_000


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _ensure_cloakbrowser() -> None:
    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.cloakbrowser", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        raise ImportError(str(exc)) from exc
    try:
        import cloakbrowser  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cloakbrowser is not installed — run `hermes tools` and select "
            "CloakBrowser, or: uv pip install 'cloakbrowser>=0.4.3,<0.5'"
        ) from exc


def launch_options() -> dict[str, Any]:
    """Build kwargs for ``cloakbrowser.launch()`` from env + config."""
    proxy = os.getenv("CLOAKBROWSER_PROXY", "").strip() or None
    raw_headless = os.getenv("CLOAKBROWSER_HEADLESS")
    if raw_headless is None:
        headless = True
    else:
        headless = _env_truthy("CLOAKBROWSER_HEADLESS", default=True)

    humanize = _env_truthy("CLOAKBROWSER_HUMANIZE", default=False)
    geoip = _env_truthy("CLOAKBROWSER_GEOIP", default=False)

    opts: dict[str, Any] = {
        "headless": headless,
        "humanize": humanize,
        "geoip": geoip,
    }
    if proxy:
        opts["proxy"] = proxy
    return opts


def _trim_text(text: str, limit: int = _MAX_BODY_CHARS) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 20] + "\n…[truncated]"


def _parse_ddg_html_results(page: Any, limit: int) -> list[dict[str, Any]]:
    """Parse DuckDuckGo HTML search results from an open Playwright page."""
    web: list[dict[str, Any]] = []
    rows = page.locator(".result")
    row_count = rows.count()
    for idx in range(min(row_count, limit)):
        row = rows.nth(idx)
        link = row.locator("a.result__a").first
        if link.count() == 0:
            continue
        href = (link.get_attribute("href") or "").strip()
        title = (link.inner_text() or "").strip()
        snippet_loc = row.locator(".result__snippet").first
        description = (
            snippet_loc.inner_text().strip() if snippet_loc.count() else ""
        )
        if not href:
            continue
        web.append(
            {
                "title": title,
                "url": href,
                "description": description,
                "position": len(web) + 1,
            }
        )
    return web


def search_duckduckgo_sync(query: str, limit: int) -> list[dict[str, Any]]:
    """Run a stealth DuckDuckGo HTML search in a disposable browser."""
    _ensure_cloakbrowser()
    from cloakbrowser import launch

    safe_limit = max(1, min(int(limit), 20))
    opts = launch_options()
    browser = launch(**opts)
    try:
        page = browser.new_page()
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        page.goto(
            search_url,
            wait_until=_DEFAULT_GOTO_WAIT,
            timeout=_SEARCH_TIMEOUT_SECS * 1000,
        )
        return _parse_ddg_html_results(page, safe_limit)
    finally:
        browser.close()


def extract_urls_sync(
    urls: list[str],
    *,
    format: str | None = None,
) -> list[dict[str, Any]]:
    """Extract readable content from URLs using one shared CloakBrowser session."""
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access

    _ensure_cloakbrowser()
    from cloakbrowser import launch

    opts = launch_options()
    browser = launch(**opts)
    results: list[dict[str, Any]] = []
    try:
        page = browser.new_page()
        for url in urls:
            blocked = check_website_access(url)
            if blocked:
                results.append(
                    {
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
                    }
                )
                continue

            try:
                page.goto(
                    url,
                    wait_until=_DEFAULT_GOTO_WAIT,
                    timeout=_EXTRACT_TIMEOUT_SECS * 1000,
                )
                final_url = page.url or url
                if not is_safe_url(final_url):
                    results.append(
                        {
                            "url": final_url,
                            "title": page.title() or "",
                            "content": "",
                            "raw_content": "",
                            "error": (
                                "Blocked: URL targets a private or internal "
                                "network address"
                            ),
                        }
                    )
                    continue

                final_blocked = check_website_access(final_url)
                if final_blocked:
                    results.append(
                        {
                            "url": final_url,
                            "title": page.title() or "",
                            "content": "",
                            "raw_content": "",
                            "error": final_blocked["message"],
                            "blocked_by_policy": {
                                "host": final_blocked["host"],
                                "rule": final_blocked["rule"],
                                "source": final_blocked["source"],
                            },
                        }
                    )
                    continue

                title = (page.title() or "").strip()
                if format == "html":
                    raw = page.content() or ""
                else:
                    raw = page.inner_text("body") or ""
                content = _trim_text(raw)
                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": final_url},
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("CloakBrowser extract failed for %s: %s", url, exc)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": str(exc),
                    }
                )
    finally:
        browser.close()
    return results
