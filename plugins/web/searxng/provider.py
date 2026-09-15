"""SearXNG search via a user-hosted instance (``/search?format=json``).

Search-only — SearXNG aggregates upstream engines but does not fetch URLs.
Env: ``SEARXNG_URL=http://localhost:8080``. An empty result set is returned as a
failure when SearXNG also reports unresponsive engines, since a hit list of zero
is indistinguishable from "every engine that could have matched was down".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)

# Engines named in the failure error / log line; the rest collapse to "and N more" so the
# string is bounded by this constant, not by the instance's engine count.
_MAX_ENGINES_NAMED = 10


def _unresponsive_engines(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """SearXNG's ``unresponsive_engines`` is a list of ``[engine, reason]`` pairs
    (searx/webutils.py ``get_translated_errors``). Anything that is not a >=2-item
    list/tuple with a truthy first element is ignored, so a future shape change
    degrades to today's behaviour instead of blacking out a self-hosted instance."""
    raw = data.get("unresponsive_engines")
    if not isinstance(raw, list):
        return []
    return [
        {"engine": str(e[0]), "reason": str(e[1])}
        for e in raw
        if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0]
    ]


class SearXNGWebSearchProvider(BaseWebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    NAME = "searxng"
    DISPLAY_NAME = "SearXNG"
    KEY_ENV = "SEARXNG_URL"

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        base_url = provider_env("SEARXNG_URL").rstrip("/")
        if not base_url:
            return search_fail("SEARXNG_URL is not set")
        data, failure = http_get_json(
            "SearXNG", f"{base_url}/search", params={"q": query, "format": "json", "pageno": 1},
            headers={"Accept": "application/json"}, timeout=15, logger=logger, reach_target=f"SearXNG at {base_url}",
        )
        if failure is not None:
            return failure
        raw_results = data.get("results", [])
        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(raw_results, key=lambda r: float(r.get("score", 0)), reverse=True)[:limit]
        web_results = titled_rows(sorted_results, "content")
        unresponsive = _unresponsive_engines(data)
        detail = ", ".join(f"{u['engine']} ({u['reason']})" for u in unresponsive[:_MAX_ENGINES_NAMED])
        if len(unresponsive) > _MAX_ENGINES_NAMED:
            detail += f", and {len(unresponsive) - _MAX_ENGINES_NAMED} more"
        if not web_results and unresponsive:
            logger.warning("SearXNG search '%s': no results and %d unresponsive engine(s): %s", query, len(unresponsive), detail)
            return search_fail(f"SearXNG returned no results and {len(unresponsive)} engine(s) were unresponsive: {detail}")
        # Rows present: still a success. Engine suspensions persist across requests, so a partial
        # outage stays at INFO rather than turning every search into an errors.log entry.
        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d)%s", query, len(web_results), len(raw_results), limit,
            f"; {len(unresponsive)} unresponsive engine(s): {detail}" if unresponsive else "",
        )
        return search_ok(web_results)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "SearXNG", "free · self-hosted", "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "SEARXNG_URL", "SearXNG instance URL (e.g. http://localhost:8080)", "https://searx.space/",
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
