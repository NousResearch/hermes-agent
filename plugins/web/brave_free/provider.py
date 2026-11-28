"""Brave Search (free-tier Data-for-Search API) — search only, 2,000 queries/month.

Config: ``web.search_backend`` / ``web.backend: "brave-free"`` (hyphen form kept for
existing user configs). Env: ``BRAVE_SEARCH_API_KEY``. Pair with Firecrawl/Tavily/Exa
for ``web_extract``. Optional ``web.brave`` keys localize results: ``country`` (2-letter,
e.g. ``CA``), ``search_lang`` (e.g. ``en``) and ``ui_lang`` (e.g. ``en-CA``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_LOCALE_KEYS = ("country", "search_lang", "ui_lang")  # passed through to Brave verbatim


def _load_brave_locale() -> dict[str, str]:
    """Read ``web.brave.{country,search_lang,ui_lang}`` from config.yaml; unset/blank keys are omitted."""
    try:
        from hermes_cli.config import load_config
        cfg: Any = load_config()
        for key in ("web", "brave"):
            cfg = cfg.get(key) if isinstance(cfg, dict) else None
    except Exception as exc:
        logger.debug("Could not load web.brave config: %s", exc)
        return {}
    if not isinstance(cfg, dict):
        return {}
    return {key: str(cfg[key]).strip() for key in _LOCALE_KEYS if isinstance(cfg.get(key), str) and cfg[key].strip()}


class BraveFreeWebSearchProvider(BaseWebSearchProvider):
    """Search-only Brave provider using the free-tier Data-for-Search API."""

    NAME = "brave-free"
    DISPLAY_NAME = "Brave Search (Free)"
    KEY_ENV = "BRAVE_SEARCH_API_KEY"

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        api_key = provider_env("BRAVE_SEARCH_API_KEY")
        if not api_key:
            return search_fail("BRAVE_SEARCH_API_KEY is not set")
        data, failure = http_get_json(
            "Brave Search", _BRAVE_ENDPOINT,
            params={"q": query, "count": max(1, min(int(limit), 20)), **_load_brave_locale()},  # Brave caps count at 20
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            timeout=15, logger=logger,
        )
        if failure is not None:
            return failure
        raw_results = (data.get("web") or {}).get("results", []) or []
        web_results = titled_rows(raw_results[:limit], "description")
        logger.info("Brave Search '%s': %d results (from %d raw, limit %d)", query, len(web_results), len(raw_results), limit)
        return search_ok(web_results)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "Brave Search (Free)", "free", "Free-tier API key — 2k queries/mo, search only.",
            "BRAVE_SEARCH_API_KEY", "Brave Search API key (free tier)", "https://brave.com/search/api/",
        )
