"""Brave Search (free-tier Data-for-Search API) — search only, 2,000 queries/month.

Config: ``web.search_backend`` / ``web.backend: "brave-free"`` (hyphen form kept for
existing user configs). Env: ``BRAVE_SEARCH_API_KEY``. Pair with Firecrawl/Tavily/Exa
for ``web_extract``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveFreeWebSearchProvider(BaseWebSearchProvider):
    """Search-only Brave provider using the free-tier Data-for-Search API."""

    NAME = "brave-free"
    DISPLAY_NAME = "Brave Search (Free)"
    KEY_ENV = "BRAVE_SEARCH_API_KEY"

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        api_key = provider_env("BRAVE_SEARCH_API_KEY")
        if not api_key:
            return search_fail("BRAVE_SEARCH_API_KEY is not set")
        data, failure = http_get_json(
            "Brave Search", _BRAVE_ENDPOINT,
            params={"q": query, "count": max(1, min(int(limit), 20))},  # Brave caps count at 20
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            timeout=15, logger=logger,
        )
        if failure is not None:
            return failure
        raw_results = (data.get("web") or {}).get("results", []) or []
        web_results = titled_rows(raw_results[:limit], "description")
        logger.info("Brave Search '%s': %d results (from %d raw, limit %d)", query, len(web_results), len(raw_results), limit)
        return search_ok(web_results)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "Brave Search (Free)", "free", "Free-tier API key — 2k queries/mo, search only.",
            "BRAVE_SEARCH_API_KEY", "Brave Search API key (free tier)", "https://brave.com/search/api/",
        )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402


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
