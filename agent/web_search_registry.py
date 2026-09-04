"""Web Search Provider Registry.

Populated by plugins via :meth:`PluginContext.register_web_search_provider`;
consumed by the ``web_search`` / ``web_extract`` wrappers in :mod:`tools.web_tools`.

Active selection, in precedence order (the ``supports_search`` /
``supports_extract`` capability filter applies at every step, so a search-only
provider configured as ``web.extract_backend`` falls through):

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
from typing import Optional

from agent.provider_registry import ProviderRegistry, is_available_safe
from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


_registry: ProviderRegistry[WebSearchProvider] = ProviderRegistry(
    label="Web", provider_cls=WebSearchProvider, logger=logger,
)
_registry.export(globals())


def _read_config_key(*path: str) -> Optional[str]:
    """Resolve a dotted config key from ``config.yaml``. Returns None on miss."""
    try:
        from hermes_cli.config import load_config_readonly

        cur = load_config_readonly()
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

# Keyless free-tier walk — strictly LAST-resort, tried only after the
# availability-filtered legacy walk finds nothing (i.e. the user has zero
# web credentials and no importable ddgs). Ring vendors expose public
# anonymous free tiers (see plugins/web/keyless_mcp.py). Unpinned keyless
# traffic round-robins across the ring per request (the ring cursor lives
# in keyless_mcp; an explicit `hermes tools` pick bypasses this walk
# entirely, and rate-limited requests fail over to the next ring vendor).
# Disable the tier with ``web.keyless_fallback: false``.
_KEYLESS_PREFERENCE = (
    "exa",
    "parallel",
    "tavily",
    "firecrawl",
    "keenable",
)


def _keyless_preference() -> tuple:
    """Return the keyless walk order for resolution.

    Delegates the entry-vendor choice to the ring cursor in
    :mod:`plugins.web.keyless_mcp` (round-robin per request, seeded by the
    per-process random session id) so resolution and dispatch agree on
    which vendor a fresh install starts at. The remaining vendors follow
    in ring order as fallbacks for registration gaps.
    """
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING, _ring_cursor

        start = _ring_cursor % len(_KEYLESS_RING)
        return tuple(
            _KEYLESS_RING[(start + i) % len(_KEYLESS_RING)]
            for i in range(len(_KEYLESS_RING))
        )
    except Exception as exc:  # noqa: BLE001 — ring optional in stripped envs
        logger.debug("keyless ring order unavailable: %s", exc)
    return _KEYLESS_PREFERENCE


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
    snapshot = _registry.merged()

    def _capable(p: WebSearchProvider) -> bool:
        return bool(
            p.supports_search() if capability == "search"
            else p.supports_extract() if capability == "extract" else False
        )

    def _available(p: WebSearchProvider) -> bool:
        return is_available_safe(p, logger, "provider %s.is_available() raised %s")

    if configured:
        provider = snapshot.get(configured)
        if provider is not None and _capable(provider):
            return provider
        if provider is None:
            logger.debug("web backend '%s' configured but not registered; falling back", configured)
        else:
            logger.debug(
                "web backend '%s' configured but does not support '%s'; falling back", configured, capability
            )

    # Fallbacks are availability-filtered so a registered-but-keyless provider
    # never becomes "active" on a fresh install.
    eligible = [p for p in snapshot.values() if _capable(p) and _available(p)]
    if len(eligible) == 1:
        return eligible[0]

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if provider is not None and provider in eligible:
            return provider

    # 4. Keyless free-tier walk — the user has NO credentialed/importable
    #    backend at all. Fall back to providers that can serve anonymously
    #    (public MCP free tiers), unless disabled via
    #    ``web.keyless_fallback: false``. This tier never pre-empts a keyed
    #    setup: it is only reachable when the legacy walk found nothing.
    if _keyless_tier_enabled():
        for name in _keyless_preference():
            provider = snapshot.get(name)
            if provider is None or not _capable(provider):
                continue
            try:
                if provider.is_keyless_available():
                    return provider
            except Exception as exc:  # noqa: BLE001 — buggy provider skipped
                logger.debug(
                    "provider %s.is_keyless_available() raised %s", name, exc
                )

    return None


def _keyless_tier_enabled() -> bool:
    """Read ``web.keyless_fallback`` from config.yaml (default: enabled)."""
    try:
        from hermes_cli.config import load_config

        web_cfg = load_config().get("web") or {}
        return bool(web_cfg.get("keyless_fallback", True))
    except Exception as exc:  # noqa: BLE001 — config layer optional
        logger.debug("keyless_fallback config read failed: %s", exc)
        return True


def _disabled_web_plugin_for(configured: Optional[str] = None, *, capability: Optional[str] = None) -> Optional[str]:
    """Plugin key of a *disabled* bundled web plugin that would have provided
    the configured backend (``web.<capability>_backend`` → ``web.backend``), or None.

    Lets the dispatcher say "re-enable web-firecrawl" instead of "No web extract
    provider configured". Resolved from config.yaml rather than the resolved
    backend because a disabled provider fails the availability gate and silently
    drops to the default. Bundled web plugins live under ``web/<vendor>`` with
    the provider name differing only by hyphen/underscore, so both are normalized.

    When a user sets ``web.extract_backend: firecrawl`` (or the search equivalent) but also lists
    ``web-firecrawl`` in ``plugins.disabled``, the provider never registers and the dispatcher would
    otherwise emit a misleading "No web extract provider configured. Set web.extract_backend to ..." error —
    even though the backend IS configured correctly. This helper detects that case so the dispatcher can
    point the user at the actual cause (issue #40190 follow-up: pi314's disabled-plugin symptom).
    """
    def _norm(s: str) -> str:
        return s.strip().lower().replace("-", "_")

    if not configured and capability in ("search", "extract"):
        configured = _configured_backend(capability)
    if not configured:
        return None

    want = _norm(configured)
    try:
        from hermes_cli.plugins import get_plugin_manager

        pm = get_plugin_manager()
        for key, loaded in pm._plugins.items():
            if (
                isinstance(key, str) and key.startswith("web/") and not loaded.enabled
                and loaded.error == "disabled via config" and _norm(key.split("/", 1)[1]) == want
            ):
                return key
    except Exception as exc:  # noqa: BLE001 — diagnostics are best-effort
        logger.debug("disabled-web-plugin lookup failed: %s", exc)
    return None


def get_active_search_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web search provider."""
    return _resolve(_configured_backend("search"), capability="search")


def get_active_extract_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web extract provider."""
    return _resolve(_configured_backend("extract"), capability="extract")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
from typing import List  # noqa: F401,E402
import threading  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'hermes_home_key': ('hermes_constants', 'hermes_home_key'),
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
