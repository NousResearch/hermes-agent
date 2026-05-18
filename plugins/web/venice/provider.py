"""Venice web search + extract — plugin form.

Backed by Venice's two augment endpoints:

  POST /augment/search   — web search (Brave default, Google opt-in)
  POST /augment/scrape   — single-URL → markdown

Both use the shared ``VENICE_API_KEY`` and respect ``VENICE_BASE_URL``
so the plugin transparently targets either Venice direct or the
HermesOS managed-Venice proxy.

Config keys::

    web:
      search_backend: "venice"
      extract_backend: "venice"
      venice:
        search_provider: "brave" | "google"   # default brave
        limit: 10                              # 1-20

Env vars::

    VENICE_API_KEY=...
    VENICE_BASE_URL=...   # optional override

Cost: $0.01 / request for both endpoints (settled via Venice billing).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


DEFAULT_VENICE_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_SEARCH_PROVIDER = "brave"
VALID_SEARCH_PROVIDERS = {"brave", "google"}


def _resolve_base_url() -> str:
    env = os.environ.get("VENICE_BASE_URL", "").strip()
    return (env or DEFAULT_VENICE_BASE_URL).rstrip("/")


def _load_venice_web_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web_section = cfg.get("web") if isinstance(cfg, dict) else None
        venice_section = web_section.get("venice") if isinstance(web_section, dict) else None
        return venice_section if isinstance(venice_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load web.venice config: %s", exc)
        return {}


def _venice_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "hermes-agent/web-venice",
    }


class VeniceWebSearchProvider(WebSearchProvider):
    """Venice ``/augment/search`` + ``/augment/scrape``."""

    @property
    def name(self) -> str:
        return "venice"

    @property
    def display_name(self) -> str:
        return "Venice"

    def is_available(self) -> bool:
        return bool(os.environ.get("VENICE_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def supports_crawl(self) -> bool:
        # Venice doesn't ship a native crawl — leave this False so the
        # crawl tool falls back to its auxiliary-model summarization path.
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Web search via Venice /augment/search."""
        import httpx

        api_key = os.environ.get("VENICE_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "VENICE_API_KEY is not set"}

        venice_cfg = _load_venice_web_config()
        search_provider = (
            venice_cfg.get("search_provider") or DEFAULT_SEARCH_PROVIDER
        ).strip().lower()
        if search_provider not in VALID_SEARCH_PROVIDERS:
            search_provider = DEFAULT_SEARCH_PROVIDER

        clamped_limit = max(1, min(20, int(limit) if isinstance(limit, int) else 5))

        payload = {
            "query": str(query)[:400],
            "limit": clamped_limit,
            "search_provider": search_provider,
        }

        try:
            response = httpx.post(
                f"{_resolve_base_url()}/augment/search",
                headers=_venice_headers(api_key),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            return {"success": False, "error": f"Venice search failed ({exc.response.status_code}): {detail}"}
        except Exception as exc:
            return {"success": False, "error": f"Venice search error: {exc}"}

        results = body.get("results") or []
        web: List[Dict[str, Any]] = []
        for i, r in enumerate(results, start=1):
            if not isinstance(r, dict):
                continue
            web.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "description": r.get("content") or "",
                    "position": i,
                }
            )

        return {"success": True, "data": {"web": web}}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract page content via Venice /augment/scrape.

        Venice's scrape endpoint takes one URL per request; we fan-out
        sequentially and accumulate. Each failure is captured per-URL so
        a partial result still returns useful data.
        """
        import httpx

        api_key = os.environ.get("VENICE_API_KEY", "").strip()
        if not api_key:
            return [{"url": "", "title": "", "content": "", "raw_content": "",
                     "metadata": {}, "error": "VENICE_API_KEY is not set"}]

        if not isinstance(urls, list):
            return []

        base_url = _resolve_base_url()
        out: List[Dict[str, Any]] = []
        with httpx.Client(timeout=60) as client:
            for url in urls:
                if not isinstance(url, str) or not url.strip():
                    continue
                try:
                    response = client.post(
                        f"{base_url}/augment/scrape",
                        headers=_venice_headers(api_key),
                        json={"url": url.strip()},
                    )
                    response.raise_for_status()
                    body = response.json()
                    content = body.get("content") or ""
                    out.append(
                        {
                            "url": body.get("url") or url,
                            "title": "",
                            "content": content,
                            "raw_content": content,
                            "metadata": {"format": body.get("format") or "markdown"},
                        }
                    )
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text[:300] if exc.response is not None else str(exc)
                    out.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "metadata": {},
                            "error": f"Venice scrape failed ({exc.response.status_code}): {detail}",
                        }
                    )
                except Exception as exc:
                    out.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "metadata": {},
                            "error": f"Venice scrape error: {exc}",
                        }
                    )
        return out

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Venice (search + scrape)",
            "badge": "paid",
            "tag": "$0.01/request — Brave (default) or Google; uses VENICE_API_KEY",
            "env_vars": [
                {
                    "key": "VENICE_API_KEY",
                    "prompt": "Venice API key",
                    "url": "https://venice.ai/settings/api",
                },
            ],
        }
