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
4. Registry fallback order, filtered by availability. This central order is
   the only place fallback provider precedence is encoded; tool wrappers
   should not maintain their own provider-name lists.
5. Otherwise ``None`` — the tool surfaces a helpful error pointing at
   ``hermes tools``.

The capability filter (``supports_search`` / ``supports_extract``) is
applied at every step so capability-specific config resolves through the
same provider contract for search and extraction.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

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


def _configured_name_from_web_config(config: Optional[dict], capability: str) -> Optional[str]:
    """Return the configured provider name for a web config block."""
    if not isinstance(config, dict):
        return None
    specific = config.get(f"{capability}_backend")
    if isinstance(specific, str) and specific.strip():
        return specific.strip().lower()
    shared = config.get("backend")
    if isinstance(shared, str) and shared.strip():
        return shared.strip().lower()
    return None


def get_configured_provider_from_web_config(
    config: Optional[dict], *, capability: str
) -> Optional[WebSearchProvider]:
    """Return the explicitly configured provider object, ignoring capability.

    This is useful for wrappers that need to surface a capability-specific
    error for a configured backend before falling back.
    """
    configured = _configured_name_from_web_config(config, capability)
    if not configured:
        return None
    return get_provider(configured)


def resolve_provider_from_web_config(
    config: Optional[dict], *, capability: str
) -> Optional[WebSearchProvider]:
    """Resolve the active provider from a loaded ``web:`` config block."""
    return _resolve(_configured_name_from_web_config(config, capability), capability=capability)


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


# Registry fallback preference — preserves behaviour for users who set no
# ``web.backend`` / ``web.<capability>_backend`` config key at all. Keep this
# as the single fallback-order table for web providers; tool wrappers should
# ask the registry to resolve providers instead of keeping their own lists.
# Explicit-key providers stay first so a managed Firecrawl gateway token does
# not pre-empt a deliberate Tavily/Exa/Parallel setup. Filtered by
# ``is_available()`` at walk time.
_REGISTRY_PREFERENCE = (
    "tavily",
    "exa",
    "parallel",
    "firecrawl",
    "camofox",
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

    2. **Single-provider shortcut.** When only one fallback-eligible provider
       supports *capability* AND ``is_available()`` reports True, return it.

    3. **Registry preference walk, filtered by availability.** Walk the
       :data:`_REGISTRY_PREFERENCE` order looking for a provider whose
       ``supports_<capability>()`` is True AND whose ``is_available()`` is
       True. This central table is the only fallback preference order;
       wrappers must not duplicate provider-name lists.

    Returns None when no provider is configured AND no available provider
    matches the registry preference; the dispatcher then returns a "set up a
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
    fallback_eligible = [p for p in eligible if p.name in _REGISTRY_PREFERENCE]
    if len(fallback_eligible) == 1:
        return fallback_eligible[0]

    for preferred in _REGISTRY_PREFERENCE:
        provider = snapshot.get(preferred)
        if (
            provider is not None
            and _capable(provider)
            and _is_available_safe(provider)
        ):
            return provider

    # Preserve the legacy no-credential default: return Firecrawl even when it
    # is not currently available so dispatch raises Firecrawl's precise setup
    # error (FIRECRAWL_API_KEY / FIRECRAWL_API_URL) instead of claiming no web
    # provider exists.
    firecrawl = snapshot.get("firecrawl")
    if firecrawl is not None and _capable(firecrawl):
        return firecrawl

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
