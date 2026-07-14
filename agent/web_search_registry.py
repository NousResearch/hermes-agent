"""
Web Search Provider Registry
============================

Central map of registered web providers. Populated by plugins at import-time
via :meth:`PluginContext.register_web_search_provider`; consumed by the
``web_search`` and ``web_extract`` tool wrappers in :mod:`tools.web_tools` to
dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by configuration with this precedence:

1. ``web.search_backend`` / ``web.extract_backend``
   (per-capability override).
2. ``web.backend`` (shared fallback).
3. If exactly one capability-eligible provider is registered AND available,
   use it.
4. Legacy preference order — ``firecrawl`` → ``parallel`` → ``tavily`` →
   ``exa`` → ``searxng`` → ``brave-free`` → ``ddgs`` — filtered by
   availability. Matches the historic ``tools.web_tools._get_backend()``
   candidate order so installs that never set a config key keep landing
   on the same provider they did before the plugin migration.
5. Otherwise ``None`` — the tool surfaces a helpful error pointing at
   ``hermes tools``.

The capability filter (``supports_search`` / ``supports_extract``) is
applied at every step so a search-only provider (``brave-free``)
configured as ``web.extract_backend`` correctly falls through to an
extract-capable backend.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


_providers: Dict[str, WebSearchProvider] = {}
_lock = threading.Lock()


def register_provider(provider: WebSearchProvider) -> None:
    """Register a web search/extract provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — makes hot-reload scenarios (tests, dev loops) behave
    predictably.
    """
    if not isinstance(provider, WebSearchProvider):
        raise TypeError(
            f"register_provider() expects a WebSearchProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Web provider .name must be a non-empty string")
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
    if existing is not None:
        logger.debug(
            "Web provider '%s' re-registered (was %r)",
            name, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered web provider '%s' (%s)",
            name, type(provider).__name__,
        )


def list_providers() -> List[WebSearchProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        items = list(_providers.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str) -> Optional[WebSearchProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    with _lock:
        return _providers.get(name.strip())


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------


def _read_config_key(*path: str) -> Optional[str]:
    """Resolve a dotted config key from ``config.yaml``. Returns None on miss."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        cur = cfg
        for segment in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(segment)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
    except Exception as exc:
        logger.debug("Could not read config %s: %s", ".".join(path), exc)
    return None


# Legacy preference order — preserves behaviour for users who set no
# ``web.backend`` / ``web.<capability>_backend`` config key at all. Matches
# the historic candidate order in :func:`tools.web_tools._get_backend`
# (paid providers first so existing paid setups don't get downgraded to
# a free tier on upgrade). Filtered by ``is_available()`` at walk time so
# we don't surface a provider the user has no credentials for.
_LEGACY_PREFERENCE = (
    "firecrawl",
    "parallel",
    "tavily",
    "exa",
    "searxng",
    "brave-free",
    "ddgs",
)


def _resolve(configured: Optional[str], *, capability: str) -> Optional[WebSearchProvider]:
    """Resolve the active provider for a capability ("search" | "extract").

    Resolution rules (in order):

    1. **Explicit config wins, ignoring availability.** If
       ``web.{capability}_backend`` or ``web.backend`` names a registered
       provider that supports *capability*, return it even if its
       :meth:`is_available` returns False — the dispatcher will surface a
       precise "X_API_KEY is not set" error to the user instead of silently
       routing somewhere else. Matches legacy
       :func:`tools.web_tools._get_backend` behavior for configured names.

    2. **Single-provider shortcut.** When only one registered provider
       supports *capability* AND ``is_available()`` reports True, return it.

    3. **Legacy preference walk, filtered by availability.** Walk the
       :data:`_LEGACY_PREFERENCE` order (firecrawl → parallel → tavily →
       exa → searxng → brave-free → ddgs) looking for a provider whose
       ``supports_<capability>()`` is True AND whose ``is_available()`` is
       True. Matches the historic ``tools.web_tools._get_backend()``
       candidate order so users with credentials but no explicit config
       key keep landing on the same provider as pre-migration. This is
       the path that fires when no config key is set — pick the
       highest-priority backend the user actually has credentials for.

    Returns None when no provider is configured AND no available provider
    matches the legacy preference; the dispatcher then returns a "set up a
    provider" error to the user.
    """
    with _lock:
        snapshot = dict(_providers)

    def _capable(p: WebSearchProvider) -> bool:
        if capability == "search":
            return bool(p.supports_search())
        if capability == "extract":
            return bool(p.supports_extract())
        return False

    def _is_available_safe(p: WebSearchProvider) -> bool:
        """Wrap ``is_available()`` so a buggy provider doesn't kill resolution."""
        try:
            return bool(p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider %s.is_available() raised %s", p.name, exc)
            return False

    # 1. Explicit config wins — return regardless of is_available() so the
    #    user gets a precise downstream error message rather than a silent
    #    backend switch. Matches _get_backend() in web_tools.py.
    if configured:
        provider = snapshot.get(configured)
        if provider is not None and _capable(provider):
            return provider
        if provider is None:
            logger.debug(
                "web backend '%s' configured but not registered; falling back",
                configured,
            )
        else:
            logger.debug(
                "web backend '%s' configured but does not support '%s'; falling back",
                configured, capability,
            )

    # 2. + 3. Fallback path — filter by availability so we don't surface
    #    a provider the user has no credentials for. Without this filter,
    #    a registered-but-unconfigured provider could end up "active" on
    #    a fresh install with no API keys at all.
    eligible = [
        p for p in snapshot.values()
        if _capable(p) and _is_available_safe(p)
    ]
    if len(eligible) == 1:
        return eligible[0]

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if (
            provider is not None
            and _capable(provider)
            and _is_available_safe(provider)
        ):
            return provider

    return None


def _disabled_web_plugin_for(configured: Optional[str] = None, *, capability: Optional[str] = None) -> Optional[str]:
    """Return the plugin key of a *disabled* bundled web plugin that would
    have provided the configured backend, or None.

    When a user sets ``web.extract_backend: firecrawl`` (or the search
    equivalent) but also lists ``web-firecrawl`` in ``plugins.disabled``,
    the provider never registers and the dispatcher would otherwise emit a
    misleading "No web extract provider configured. Set web.extract_backend
    to ..." error — even though the backend IS configured correctly. The
    real fix is to re-enable the plugin. This helper detects that case so
    the dispatcher can point the user at the actual cause (issue #40190
    follow-up: pi314's disabled-plugin symptom).

    Pass ``capability`` ("search" | "extract") to resolve the configured
    name straight from ``config.yaml`` (``web.<capability>_backend`` →
    ``web.backend``). This is more reliable than the resolved backend the
    dispatcher fell back to, since a disabled provider fails the
    ``_is_backend_available`` gate and the dispatcher silently drops to
    the shared default. An explicit ``configured`` name still wins when
    given.

    Matching is by convention: bundled web plugins live under the
    ``web/<vendor>`` key with the provider ``name`` differing only in
    hyphen/underscore (``brave-free`` provider ⇄ ``web/brave_free`` key,
    ``firecrawl`` ⇄ ``web/firecrawl``). We normalize both sides before
    comparing so every bundled provider is covered without hardcoding a
    per-vendor table.
    """
    def _norm(s: str) -> str:
        return s.strip().lower().replace("-", "_")

    if not configured and capability in ("search", "extract"):
        configured = (
            _read_config_key("web", f"{capability}_backend")
            or _read_config_key("web", "backend")
        )
    if not configured:
        return None

    want = _norm(configured)
    try:
        from hermes_cli.plugins import get_plugin_manager

        pm = get_plugin_manager()
        for key, loaded in pm._plugins.items():
            if not isinstance(key, str) or not key.startswith("web/"):
                continue
            if loaded.enabled:
                continue
            if loaded.error != "disabled via config":
                continue
            vendor = key.split("/", 1)[1]
            if _norm(vendor) == want:
                return key
    except Exception as exc:  # noqa: BLE001 — diagnostics are best-effort
        logger.debug("disabled-web-plugin lookup failed: %s", exc)
    return None


def get_active_search_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web search provider.

    Reads ``web.search_backend`` (preferred) or ``web.backend`` (shared
    fallback) from config.yaml; falls back per the module docstring.
    """
    explicit = _read_config_key("web", "search_backend") or _read_config_key("web", "backend")
    return _resolve(explicit, capability="search")


def get_active_extract_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web extract provider.

    Reads ``web.extract_backend`` (preferred) or ``web.backend`` (shared
    fallback) from config.yaml; falls back per the module docstring.
    """
    explicit = _read_config_key("web", "extract_backend") or _read_config_key("web", "backend")
    return _resolve(explicit, capability="extract")


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _providers.clear()


# ---------------------------------------------------------------------------
# Runtime fallback chain
# ---------------------------------------------------------------------------


def _is_provider_failure(result: Any) -> bool:
    """Return True when a provider result indicates the call failed.

    Detection rules:

    * ``None`` / empty dict / empty list -> failure.
    * ``success: False`` -> failure.
    * ``success: True`` but empty ``data.web`` -> failure (0 results).
    * List result where every entry has an ``error`` key -> failure.
    * List result where no entry has an ``error`` key AND no entry has
      usable content (empty/missing ``content``) -> failure.  This catches
      providers (e.g. Firecrawl) that return a row per URL with neither
      markdown nor HTML -- structurally "successful" but useless to the
      user.  At least one content-bearing row means partial success and
      does NOT trigger fallback.
    * List result with at least one content-bearing entry -> NOT a failure
      (partial success is good enough to return to the user).
    """
    if not result:
        return True
    if isinstance(result, dict):
        if result.get("success") is False:
            return True
        # Dict-shaped: {"success": True, "data": {"web": [...]}}
        data = result.get("data")
        if isinstance(data, dict):
            web = data.get("web")
            if isinstance(web, list) and not web:
                return True
        return False
    if isinstance(result, list):
        if not result:
            return True
        # All entries have an "error" key -> failure
        if all(isinstance(r, dict) and r.get("error") for r in result):
            return True
        # No entry has an "error" key, but none has usable content either
        # (e.g. Firecrawl returning empty content for every URL).  At least
        # one content-bearing row means partial success.
        if all(isinstance(r, dict) and not r.get("content") for r in result):
            return True
    return False


def resolve_fallback_chain(*, capability: str) -> List[WebSearchProvider]:
    """Return an ordered list of providers to try for *capability*.

    Used by the web_tools dispatcher to implement "try provider A, if it
    fails try provider B" semantics. The chain includes all registered
    providers that support *capability*, ordered by :data:`_LEGACY_PREFERENCE`
    with any unrecognised providers appended alphabetically.

    All providers in the chain are filtered by ``is_available()`` so we
    never waste a round-trip on a provider the user has no credentials for.
    When the configured provider is unavailable, the preflight resolver
    (:func:`_resolve` / :func:`get_active_search_provider`) already
    surfaces a typed credential error to the user -- the chain's job is
    runtime failures after a valid invocation, not config errors.
    """
    with _lock:
        snapshot = dict(_providers)

    def _capable(p: WebSearchProvider) -> bool:
        if capability == "search":
            return bool(p.supports_search())
        if capability == "extract":
            return bool(p.supports_extract())
        return False

    def _is_available_safe(p: WebSearchProvider) -> bool:
        try:
            return bool(p.is_available())
        except Exception:
            return False

    # Start with explicitly configured provider (if any)
    configured = (
        _read_config_key("web", f"{capability}_backend")
        or _read_config_key("web", "backend")
    )
    chain: List[WebSearchProvider] = []
    seen: set = set()

    if configured:
        p = snapshot.get(configured)
        if p and _capable(p) and _is_available_safe(p):
            chain.append(p)
            seen.add(p.name)

    # Add providers in legacy preference order (filtered by availability)
    for name in _LEGACY_PREFERENCE:
        p = snapshot.get(name)
        if p and p.name not in seen and _capable(p) and _is_available_safe(p):
            chain.append(p)
            seen.add(p.name)

    # Append any remaining capable+available providers alphabetically
    for p in sorted(snapshot.values(), key=lambda x: x.name):
        if p.name not in seen and _capable(p) and _is_available_safe(p):
            chain.append(p)
            seen.add(p.name)

    if chain:
        logger.debug(
            "Web %s fallback chain: %s",
            capability,
            " -> ".join(p.name for p in chain),
        )

    return chain
