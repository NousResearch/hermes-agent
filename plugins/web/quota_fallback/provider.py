"""Quota-fallback web search provider — plugin form.

Tries child providers in priority order, falling through on quota
exhaustion, rate limits, empty results, and transient errors.  Uses
the existing registry so child providers are never re-implemented.

**Language-aware routing**: when a query contains CJK characters the
provider prefers Chinese-optimised backends (baidu) first; otherwise
it prefers English-language backends (brave-free, exa, tavily) first.
Configure via ``fallback_search.order_cjk`` and
``fallback_search.order_latin``.

Search-only (``supports_extract`` is False).

Config keys this provider responds to::

    web:
      search_backend: "quota-fallback"
      fallback_search:
        # Shared fallback chain (used when no language-specific order matches)
        order:
          - baidu
          - serper
          - brave-free
          - exa
          - tavily
          - searxng
        # Chinese-optimised order (input contains CJK)
        order_cjk:
          - baidu
          - serper
          - brave-free
          - exa
          - tavily
          - searxng
        # English-optimised order (purely latin / ascii query)
        order_latin:
          - brave-free
          - exa
          - tavily
          - serper
          - baidu
          - searxng
        empty_results_fallback: true
        cooldown_minutes:
          rate_limit: 30
          temporary_error: 10
          quota: 1440

Auth is inherited from each child provider — ``quota-fallback`` has no
credentials of its own.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Shared fallback (when neither cjk nor latin order is configured).
_DEFAULT_SHARED = ["baidu", "exa", "tavily", "searxng"]
# Chinese-optimised (default for queries with CJK characters).
_DEFAULT_CJK = ["baidu", "exa", "tavily", "searxng"]
# English-optimised (default for purely latin/ascii queries).
_DEFAULT_LATIN = ["tavily", "exa", "baidu", "searxng"]

_STATE_FILE = "web-fallback-state.json"

# Error-text patterns that signal a child provider should be skipped.
_QUOTA_PATTERNS = (
    "quota",
    "credit",
    "billing",
    "402",
    "insufficient",
    "exhausted",
)
_RATE_LIMIT_PATTERNS = ("rate limit", "429", "too many requests", "ratelimit")
_TEMPORARY_PATTERNS = (
    "timeout",
    "connection error",
    "connection refused",
    "name resolution",
    "temporary",
    "unreachable",
    "reset",
    "eof",
)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _has_cjk(text: str) -> bool:
    """Return True when *text* contains any CJK (Chinese / Japanese / Korean) character."""
    import unicodedata

    for ch in text:
        cat = unicodedata.category(ch)
        # Lo = Letter, Other  (CJK ideographs are Lo).
        # Also catch CJK Unified Ideographs block explicitly.
        if cat == "Lo" or (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3400" <= ch <= "\u4dbf"
            or "\uf900" <= ch <= "\ufaff"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Cooldown state helpers
# ---------------------------------------------------------------------------


def _state_path() -> Path:
    """Return the path to the cooldown state file."""
    hermes_home = os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))
    return Path(hermes_home) / _STATE_FILE


def _read_cooldowns() -> Dict[str, Dict[str, Any]]:
    """Read the cooldown state file, returning {provider_name: {type, until}}.

    Returns an empty dict on any read failure (missing/corrupt file).
    """
    path = _state_path()
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        cd = data.get("cooldowns", {})
        if not isinstance(cd, dict):
            return {}
        return cd
    except Exception as exc:
        logger.debug("Failed to read fallback state file %s: %s", path, exc)
        return {}


def _write_cooldowns(cooldowns: Dict[str, Dict[str, Any]]) -> None:
    """Atomically write the cooldown state file.

    No-op on write failure (non-fatal).
    """
    path = _state_path()
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"cooldowns": cooldowns}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:
        logger.debug("Failed to write fallback state file %s: %s", path, exc)


def _is_cooldown_active(cooldowns: Dict[str, Dict[str, Any]], name: str) -> bool:
    """Return True if *name* is inside a cooldown window."""
    entry = cooldowns.get(name)
    if entry is None:
        return False
    until = entry.get("until", 0)
    if not isinstance(until, (int, float)):
        return False
    return time.time() < until


def _record_failure(cooldowns: Dict[str, Dict[str, Any]], name: str, error: str, cfg: Dict[str, Any]) -> None:
    """Record a failure for *name* with the appropriate cooldown duration."""
    minutes = _cooldown_minutes_for(error, cfg)
    if minutes <= 0:
        return
    cooldowns[name] = {
        "type": _classify_error(error),
        "until": time.time() + minutes * 60,
    }


def _classify_error(error: str) -> str:
    """Classify an error string as 'quota', 'rate_limit', or 'temporary'."""
    lower = error.lower()
    if any(p in lower for p in _QUOTA_PATTERNS):
        return "quota"
    if any(p in lower for p in _RATE_LIMIT_PATTERNS):
        return "rate_limit"
    if any(p in lower for p in _TEMPORARY_PATTERNS):
        return "temporary"
    return "temporary"


def _cooldown_minutes_for(error: str, cfg: Dict[str, Any]) -> int:
    """Return cooldown minutes for the error, reading overrides from *cfg*."""
    cooldown_cfg = cfg.get("cooldown_minutes", {})
    kind = _classify_error(error)
    if kind == "quota":
        return int(cooldown_cfg.get("quota", 1440))
    if kind == "rate_limit":
        return int(cooldown_cfg.get("rate_limit", 30))
    return int(cooldown_cfg.get("temporary_error", 10))


# ---------------------------------------------------------------------------
# Error matching
# ---------------------------------------------------------------------------


def _should_fallback_on_error(error: str) -> Tuple[bool, str]:
    """Return (should_fallback, failure_type) given a child's error string."""
    lower = error.lower()

    # Network / transient errors → fallback
    if any(p in lower for p in _TEMPORARY_PATTERNS):
        return True, "temporary"

    # Definitive quota / billing errors → fallback
    if any(p in lower for p in _QUOTA_PATTERNS):
        return True, "quota"

    # Rate-limit errors → fallback
    if any(p in lower for p in _RATE_LIMIT_PATTERNS):
        return True, "rate_limit"

    # Unknown — still try next provider.
    return True, "unknown"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class QuotaFallbackWebSearchProvider(WebSearchProvider):
    """Search-only aggregate provider with quota-aware fallback chaining.

    Uses :func:`agent.web_search_registry.get_provider` to retrieve child
    providers by name, so there is no duplication of API-call logic and
    zero risk of recursive self-selection.

    **Language-aware routing**: queries with CJK characters prefer
    ``order_cjk`` (baidu first); latin-only queries prefer ``order_latin``
    (brave-free / exa / tavily first). Falls back to ``order`` when
    neither language-specific key is configured.
    """

    @property
    def name(self) -> str:
        return "quota-fallback"

    @property
    def display_name(self) -> str:
        return "Quota Fallback Search"

    def is_available(self) -> bool:
        """Always available — at least one child provider may be usable.

        ``is_available`` is a cheap local check (no network).  The provider
        is considered available if at least one child provider is registered
        (regardless of credential state — fallback will skip unconfigured
        children at search time).
        """
        # Defer to search-time lookup; avoid loading registry at tool-paint.
        return True

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _read_config(self) -> Dict[str, Any]:
        """Load the ``web.fallback_search`` config block from config.yaml.

        Returns an empty dict on any failure so the provider never crashes
        due to a config read error.
        """
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            web_cfg = cfg.get("web", {})
            fb = web_cfg.get("fallback_search", {})
            if isinstance(fb, dict):
                return fb
        except Exception as exc:
            logger.debug("Could not read fallback_search config: %s", exc)
        return {}

    def _get_order(self, cfg: Dict[str, Any], query: str = "") -> List[str]:
        """Return the ordered child-provider list for *query*.

        When *query* contains CJK characters the method prefers the
        ``order_cjk`` config key; otherwise ``order_latin``.  Falls back
        to ``order`` (shared) when the language-specific key is absent,
        then to the hard-coded defaults.
        """
        is_cjk = _has_cjk(query) if query else False

        if is_cjk:
            explicit = cfg.get("order_cjk") or cfg.get("order")
        else:
            explicit = cfg.get("order_latin") or cfg.get("order")

        if isinstance(explicit, list) and explicit:
            return [str(n).strip() for n in explicit if n and str(n).strip()]

        # Hard-coded default
        return list(_DEFAULT_CJK if is_cjk else _DEFAULT_LATIN)

    def _empty_results_fallback(self, cfg: Dict[str, Any]) -> bool:
        """Return True if empty results should trigger a fallback."""
        return bool(cfg.get("empty_results_fallback", True))

    # ------------------------------------------------------------------
    # Search (the main fallback loop)
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search with quota-aware fallback chaining.

        Routing is language-aware: CJK queries prefer Chinese-optimised
        backends; latin queries prefer English-language backends.

        Returns ``{success: True, data: {web: [...]}}`` on first success,
        or ``{success: False, error: str}`` when every child provider fails.
        """
        if not query or not query.strip():
            return {"success": False, "error": "Query is empty"}

        cfg = self._read_config()
        order = self._get_order(cfg, query=query)
        empty_fallback = self._empty_results_fallback(cfg)
        cooldowns = _read_cooldowns()
        errors: List[Tuple[str, str]] = []  # (provider_name, short_error)
        new_cooldowns: Dict[str, Dict[str, Any]] = dict(cooldowns)

        logger.debug(
            "quota-fallback: query=%r cjk=%s order=%s",
            query[:60], _has_cjk(query), order,
        )

        for child_name in order:
            # Skip cooldowned providers
            if _is_cooldown_active(cooldowns, child_name):
                errors.append((child_name, "skipped (cooldown active)"))
                logger.debug("quota-fallback: %s in cooldown, skipping", child_name)
                continue

            # Look up child provider
            from agent.web_search_registry import get_provider

            child = get_provider(child_name)
            if child is None:
                errors.append((child_name, "not registered"))
                logger.debug("quota-fallback: %s not registered, skipping", child_name)
                continue

            if not child.is_available():
                errors.append((child_name, "unavailable (check credentials/config)"))
                logger.debug("quota-fallback: %s unavailable, skipping", child_name)
                continue

            if not child.supports_search():
                errors.append((child_name, "does not support search"))
                continue

            # Try this child
            try:
                result = child.search(query, limit=limit)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)[:200]
                errors.append((child_name, f"exception: {msg}"))
                logger.warning("quota-fallback: %s raised %s", child_name, msg)
                _record_failure(new_cooldowns, child_name, msg, cfg)
                continue

            # Analyse result
            if not isinstance(result, dict):
                errors.append((child_name, "non-dict response"))
                continue

            success = result.get("success", False)
            data = result.get("data", {})

            if success:
                web_results = data.get("web", []) if isinstance(data, dict) else []
                if web_results:
                    # Real success — return immediately
                    _write_cooldowns(new_cooldowns)
                    return {"success": True, "data": {"web": web_results}}

                # Empty result set
                if not empty_fallback:
                    _write_cooldowns(new_cooldowns)
                    return {"success": True, "data": {"web": []}}

                errors.append((child_name, "empty results"))
                logger.debug("quota-fallback: %s returned empty results, falling back", child_name)
                continue

            # Failure branch
            child_error = result.get("error", "unknown error")[:200]
            should_fb, fb_type = _should_fallback_on_error(child_error)
            errors.append((child_name, f"{fb_type}: {child_error}" if fb_type != child_error else child_error))
            _record_failure(new_cooldowns, child_name, child_error, cfg)

            if not should_fb:
                logger.debug(
                    "quota-fallback: %s failed with non-fallback error, trying next: %s",
                    child_name, child_error,
                )
                continue

            logger.debug(
                "quota-fallback: %s failed (%s), trying next provider",
                child_name, child_error,
            )

        # All children exhausted
        _write_cooldowns(new_cooldowns)

        error_summary = "; ".join(f"{name}: {err}" for name, err in errors)
        logger.warning("quota-fallback: all providers failed: %s", error_summary)
        return {"success": False, "error": f"All search providers failed: {error_summary}"}

    # ------------------------------------------------------------------
    # Setup schema
    # ------------------------------------------------------------------

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Quota Fallback Search",
            "badge": "free · fallback",
            "tag": (
                "Aggregate provider with language-aware routing: CJK queries go to "
                "baidu first; latin queries go to brave-free / exa / tavily first. "
                "Falls through on quota/rate-limit/errors."
            ),
            "env_vars": [],
        }
