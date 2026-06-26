"""CloakBrowser stealth web search + extract — Hermes plugin form.

Uses `cloakbrowser` (stealth Chromium / Playwright drop-in) for:

  - **web_search** — DuckDuckGo HTML results via a real browser (bot-resistant)
  - **web_extract** — navigate + extract body text or HTML with SSRF/policy gates

Upstream: https://github.com/zapabob/CloakBrowser (PyPI: ``cloakbrowser``).
No API key required; optional ``CLOAKBROWSER_PROXY`` for residential egress.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as _cf
import logging
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

from plugins.web.cloakbrowser.session import (
    _SEARCH_TIMEOUT_SECS,
    _ensure_cloakbrowser,
    extract_urls_sync,
    search_duckduckgo_sync,
)

logger = logging.getLogger(__name__)


class CloakBrowserWebSearchProvider(WebSearchProvider):
    """Stealth-browser web search and content extraction via CloakBrowser."""

    @property
    def name(self) -> str:
        return "cloakbrowser"

    @property
    def display_name(self) -> str:
        return "CloakBrowser (stealth)"

    def is_available(self) -> bool:
        try:
            _ensure_cloakbrowser()
            return True
        except ImportError:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            _ensure_cloakbrowser()
        except ImportError as exc:
            return {"success": False, "error": str(exc)}

        safe_limit = max(1, int(limit))
        pool = _cf.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(search_duckduckgo_sync, query, safe_limit)
            try:
                web_results = future.result(timeout=_SEARCH_TIMEOUT_SECS)
            except _cf.TimeoutError:
                logger.warning(
                    "CloakBrowser search timed out after %ds for query: %r",
                    _SEARCH_TIMEOUT_SECS,
                    query,
                )
                return {
                    "success": False,
                    "error": (
                        f"CloakBrowser search timed out after "
                        f"{_SEARCH_TIMEOUT_SECS}s — try again or reduce load."
                    ),
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("CloakBrowser search error: %s", exc)
            return {"success": False, "error": f"CloakBrowser search failed: {exc}"}
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        logger.info(
            "CloakBrowser search %r: %d results (limit %d)",
            query,
            len(web_results),
            safe_limit,
        )
        return {"success": True, "data": {"web": web_results}}

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        try:
            _ensure_cloakbrowser()
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": str(exc)} for u in urls
            ]

        fmt = kwargs.get("format")
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(extract_urls_sync, urls, format=fmt),
                timeout=max(60, 30 * len(urls)),
            )
        except asyncio.TimeoutError:
            logger.warning("CloakBrowser extract batch timed out (%d URLs)", len(urls))
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": "CloakBrowser extract timed out",
                }
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "CloakBrowser (stealth)",
            "badge": "free · no key · search + extract",
            "tag": (
                "Stealth Chromium for bot-resistant search and crawling "
                "(DuckDuckGo HTML + page extract). Pair with CLOAKBROWSER_PROXY "
                "for strict anti-bot sites."
            ),
            "env_vars": [
                {
                    "key": "CLOAKBROWSER_PROXY",
                    "prompt": "Optional HTTP/S proxy (residential recommended)",
                    "url": "https://github.com/zapabob/CloakBrowser#install",
                    "password": True,
                },
            ],
            "post_setup": "cloakbrowser",
        }
