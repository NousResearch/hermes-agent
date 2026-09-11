"""Keenable (https://keenable.ai) web search + fetch — keyless-ring member.

Env: ``KEENABLE_API_KEY`` (optional; keyless free tier works without it).
Config: ``web.provider_tier.keenable: free|paid`` pins the tier (unset = auto).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from plugins.web._common import (
    SEARCH_LIMIT_CAP, BaseWebSearchProvider, document, http_status_detail, keyless_extract, keyless_search,
    keyless_variant_schema, page_error, provider_env, run_extract, run_search, search_fail, search_ok, use_keyless, web_hit,
)

logger = logging.getLogger(__name__)

_KEENABLE_API_URL = "https://api.keenable.ai"


def _keenable_headers(api_key: str) -> Dict[str, str]:
    # The keyless tier structurally requires an app-identifier header; no user identifiers are sent.
    headers = {"X-Keenable-Title": "hermes-agent"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class KeenableWebSearchProvider(BaseWebSearchProvider):
    """Keenable search + extract provider (keyed or keyless)."""

    NAME = "keenable"
    DISPLAY_NAME = "Keenable"
    KEY_ENV = "KEENABLE_API_KEY"
    EXTRACT = True
    KEYLESS = True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        def _body() -> Dict[str, Any]:
            api_key = provider_env("KEENABLE_API_KEY")
            if use_keyless("keenable", api_key):
                return keyless_search("Keenable", "keenable", query, limit, logger)
            import requests
            logger.info("Keenable search: '%s' (limit=%d)", query, limit)
            response = requests.post(
                f"{_KEENABLE_API_URL}/v1/search",
                json={"query": query, "max_results": min(max(1, int(limit)), SEARCH_LIMIT_CAP)},
                headers=_keenable_headers(api_key), timeout=30,
            )
            if response.status_code >= 400:
                return search_fail(f"Keenable search failed: {http_status_detail(response)}")
            return search_ok([
                web_hit(r.get("url") or "", r.get("title") or "", r.get("snippet") or r.get("description") or "", i + 1)
                for i, r in enumerate(response.json().get("results") or [])
            ])

        return run_search("Keenable", logger, _body, verbatim_value_error=False)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        def _body() -> List[Dict[str, Any]]:
            api_key = provider_env("KEENABLE_API_KEY")
            if use_keyless("keenable", api_key):
                return keyless_extract("Keenable", "keenable", urls, logger)
            import requests
            logger.info("Keenable extract: %d URL(s)", len(urls))
            results: List[Dict[str, Any]] = []
            for url in urls:
                try:
                    response = requests.get(f"{_KEENABLE_API_URL}/v1/fetch", params={"url": url}, headers=_keenable_headers(api_key), timeout=30)
                    if response.status_code >= 400:
                        raise ValueError(http_status_detail(response))
                    data = response.json()
                    results.append(document(data.get("url") or url, data.get("title") or "", data.get("content") or "", source_url=url))
                except Exception as exc:  # noqa: BLE001 — per-URL error entry
                    results.append(page_error(url, f"Keenable extract failed: {exc}"))
            return results

        return run_extract("Keenable", logger, urls, _body, verbatim_value_error=False)

    def get_setup_schema(self) -> Dict[str, Any]:
        return keyless_variant_schema(
            "Keenable", "KEENABLE_API_KEY", "https://keenable.ai",
            free_tag="Independent web index for AI apps — fast search + page fetch on Keenable's anonymous free tier.",
            paid_tag="Independent web index for AI apps. Keyed access with higher limits and guaranteed service.",
        )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'WebSearchProvider': ('agent.web_search_provider', 'WebSearchProvider'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
