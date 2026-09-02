"""Keenable web search + content extraction — bundled plugin.

Keenable (https://keenable.ai) operates an independent web index for AI
apps with public keyless endpoints (rate-limited free tier; keyed access
via KEENABLE_API_KEY for higher limits). Integrated as a keyless-ring
member following the Exa/Parallel/Firecrawl pattern: fresh installs with
zero web credentials rotate across the ring vendors' free tiers.

Credit: Keenable integration originally proposed by Ilya Gusev (Keenable)
in PR #49758; the native provider form follows the salvage of that work
plus the keyless-ring design.

Config keys this provider responds to::

    web:
      search_backend: "keenable"      # explicit per-capability
      extract_backend: "keenable"     # explicit per-capability
      backend: "keenable"             # shared fallback
      provider_tier:
        keenable: free|paid           # pin the tier (unset = auto)

Env var::

    KEENABLE_API_KEY=...   # optional — keyless free tier works without it
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_KEENABLE_API_URL = "https://api.keenable.ai"


def _keenable_headers(api_key: str) -> Dict[str, str]:
    """Build Keenable request headers for keyed or keyless access.

    Their keyless tier structurally requires an app-identifier header
    (X-Keenable-Title); no user identifiers are sent.
    """
    headers = {"X-Keenable-Title": "hermes-agent"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class KeenableWebSearchProvider(WebSearchProvider):
    """Keenable search + extract provider (keyed or keyless)."""

    @property
    def name(self) -> str:
        return "keenable"

    @property
    def display_name(self) -> str:
        return "Keenable"

    def is_available(self) -> bool:
        """Return True when ``KEENABLE_API_KEY`` is set to a non-empty value."""
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("KEENABLE_API_KEY"))

    def is_keyless_available(self) -> bool:
        """Keenable serves anonymous free-tier calls via its public endpoints.

        Default-on ring member of the keyless free tier. False when the
        user pinned ``web.provider_tier.keenable: paid``.
        """
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return keyless_enabled() and provider_tier("keenable") != "paid"

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Keenable search (keyed path or keyless ring)."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import search_with_failover, use_keyless

            api_key = get_provider_env("KEENABLE_API_KEY")
            if use_keyless("keenable", api_key):
                logger.info(
                    "Keenable keyless search: '%s' (limit=%d)", query, limit
                )
                return search_with_failover("keenable", query, limit)

            import requests

            logger.info("Keenable search: '%s' (limit=%d)", query, limit)
            response = requests.post(
                f"{_KEENABLE_API_URL}/v1/search",
                json={"query": query, "max_results": min(max(1, int(limit)), 20)},
                headers=_keenable_headers(api_key),
                timeout=30,
            )
            if response.status_code >= 400:
                detail = (response.text or "").strip() or f"HTTP {response.status_code}"
                return {"success": False, "error": f"Keenable search failed: {detail}"}
            data = response.json()

            web_results = []
            for i, result in enumerate(data.get("results") or []):
                web_results.append(
                    {
                        "url": result.get("url") or "",
                        "title": result.get("title") or "",
                        "description": result.get("snippet")
                        or result.get("description")
                        or "",
                        "position": i + 1,
                    }
                )
            return {"success": True, "data": {"web": web_results}}
        except Exception as exc:  # noqa: BLE001 — surface as failure
            logger.warning("Keenable search error: %s", exc)
            return {"success": False, "error": f"Keenable search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content via Keenable's fetch endpoint (per-URL).

        Sync — the dispatcher wraps in a thread when the caller is async.
        Returns the legacy list-of-results shape; per-URL failures become
        items with an ``error`` field.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import extract_with_failover, use_keyless

            api_key = get_provider_env("KEENABLE_API_KEY")
            if use_keyless("keenable", api_key):
                logger.info("Keenable keyless extract: %d URL(s)", len(urls))
                return extract_with_failover("keenable", list(urls))

            import requests

            logger.info("Keenable extract: %d URL(s)", len(urls))
            results: List[Dict[str, Any]] = []
            for url in urls:
                try:
                    response = requests.get(
                        f"{_KEENABLE_API_URL}/v1/fetch",
                        params={"url": url},
                        headers=_keenable_headers(api_key),
                        timeout=30,
                    )
                    if response.status_code >= 400:
                        raise ValueError(
                            (response.text or "").strip()
                            or f"HTTP {response.status_code}"
                        )
                    data = response.json()
                    content = data.get("content") or ""
                    title = data.get("title") or ""
                    results.append(
                        {
                            "url": data.get("url") or url,
                            "title": title,
                            "content": content,
                            "raw_content": content,
                            "metadata": {"sourceURL": url, "title": title},
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — per-URL error entry
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": f"Keenable extract failed: {exc}",
                        }
                    )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Keenable extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "",
                 "error": f"Keenable extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Keenable · Free (keyless)",
            "badge": "free · no key",
            "tag": (
                "Independent web index for AI apps — fast search + page "
                "fetch on Keenable's anonymous free tier."
            ),
            "env_vars": [],
            "web_tier": "free",
            "variants": [
                {
                    "name": "Keenable · Paid (API key)",
                    "badge": "paid",
                    "tag": (
                        "Independent web index for AI apps. Keyed access "
                        "with higher limits and guaranteed service."
                    ),
                    "env_vars": [
                        {
                            "key": "KEENABLE_API_KEY",
                            "prompt": "Keenable API key",
                            "url": "https://keenable.ai",
                        },
                    ],
                    "web_tier": "paid",
                },
            ],
        }
