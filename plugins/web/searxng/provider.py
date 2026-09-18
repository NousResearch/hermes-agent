"""SearXNG search via a user-hosted instance (``/search?format=json``).

Search-only — SearXNG aggregates upstream engines but does not fetch URLs.
Env: ``SEARXNG_URL=http://localhost:8080``.
HTTP timeout: ``web.searxng_timeout`` in config.yaml (seconds, default 15).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)

DEFAULT_SEARXNG_TIMEOUT = 15.0


def _resolve_searxng_timeout() -> Tuple[Optional[float], Optional[str]]:
    """``web.searxng_timeout`` from config.yaml -> ``(timeout, None)`` or ``(None, error)``.

    Unset (or explicit YAML ``null``) falls back to :data:`DEFAULT_SEARXNG_TIMEOUT`, preserving
    the historical hardcoded 15-second behavior exactly. An explicitly configured value that is
    not a positive, finite number (zero, negative, NaN, Inf, a YAML boolean, an integer too large
    to represent as a float, or non-numeric) is rejected outright rather than silently
    substituted — a bad timeout should surface immediately as a config error, not manifest later
    as unpredictable search failures or a hang.
    """
    from tools.web_tools import _load_web_config  # lazy: tests patch tools.web_tools._load_web_config

    raw = _load_web_config().get("searxng_timeout")
    if raw is None:
        return DEFAULT_SEARXNG_TIMEOUT, None
    # bool is a subclass of int in Python, so `True`/`False` from YAML would otherwise pass
    # straight through float() as 1.0/0.0 — reject explicitly rather than accept a nonsense value.
    if isinstance(raw, bool):
        return None, f"web.searxng_timeout must be a positive number of seconds, got {raw!r}"
    try:
        value = float(raw)
    except OverflowError:
        return None, f"web.searxng_timeout must be a positive, finite number of seconds, got {raw!r}"
    except (TypeError, ValueError):
        return None, f"web.searxng_timeout must be a positive number of seconds, got {raw!r}"
    if not math.isfinite(value) or value <= 0:
        return None, f"web.searxng_timeout must be a positive, finite number of seconds, got {raw!r}"
    return value, None


class SearXNGWebSearchProvider(BaseWebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    NAME = "searxng"
    DISPLAY_NAME = "SearXNG"
    KEY_ENV = "SEARXNG_URL"

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        base_url = provider_env("SEARXNG_URL").rstrip("/")
        if not base_url:
            return search_fail("SEARXNG_URL is not set")
        timeout, timeout_error = _resolve_searxng_timeout()
        if timeout_error is not None or timeout is None:
            logger.warning("SearXNG search rejected: %s", timeout_error)
            return search_fail(timeout_error or "web.searxng_timeout is invalid")
        data, failure = http_get_json(
            "SearXNG", f"{base_url}/search", params={"q": query, "format": "json", "pageno": 1},
            headers={"Accept": "application/json"}, timeout=timeout, logger=logger,
            reach_target=f"SearXNG at {base_url}",
        )
        if failure is not None:
            return failure
        raw_results = data.get("results", [])
        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(raw_results, key=lambda r: float(r.get("score", 0)), reverse=True)[:limit]
        web_results = titled_rows(sorted_results, "content")
        logger.info("SearXNG search '%s': %d results (from %d raw, limit %d)", query, len(web_results), len(raw_results), limit)
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
