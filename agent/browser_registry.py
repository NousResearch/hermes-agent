"""
Browser Provider Registry
=========================

Central map of registered cloud browser providers. Populated by plugins at
import-time via :meth:`PluginContext.register_browser_provider`; consumed by
:func:`tools.browser_tool._get_cloud_provider` to route each cloud-mode
``browser_*`` tool call to the active backend.

Active selection
----------------
The active provider is chosen by configuration with this precedence:

1. ``browser.cloud_provider`` in ``config.yaml`` (explicit override).
2. Legacy preference order — ``browser-use`` → ``browserbase`` — filtered by
   availability. Matches the historic auto-detect order in
   :func:`tools.browser_tool._get_cloud_provider` (Browser Use checked first
   because it covers both the managed Nous gateway and direct API key path;
   Browserbase as the older direct-credentials fallback). ``firecrawl`` is
   intentionally NOT in the legacy walk — users only get Firecrawl as a
   cloud browser when they explicitly set ``browser.cloud_provider:
   firecrawl``, matching pre-migration behaviour where Firecrawl was never
   auto-selected.
3. Otherwise ``None`` — the dispatcher falls back to local browser mode.

The explicit-config branch (rule 1) intentionally ignores ``is_available()``
so the dispatcher surfaces a typed "X_API_KEY is not set" error to the user
instead of silently switching backends. Matches the legacy
:func:`tools.browser_tool._get_cloud_provider` behaviour for configured names.

Note: there is no "capability" split here (unlike the web subsystem, which
has search/extract/crawl). Every browser provider implements the full
:class:`agent.browser_provider.BrowserProvider` lifecycle; the registry's
job is purely selection, not capability routing.

Rendered-page integrations without persistent CDP/WebSocket sessions do not
belong in this registry. For example, MrScraper registers
``mrscraper_fetch_rendered_html`` as a native tool and registers separately as
a web search/extract provider; it deliberately is not a
``browser.cloud_provider``.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.browser_provider import BrowserProvider
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)


_providers: Dict[str, BrowserProvider] = {}
_scoped_providers: Dict[str, Dict[str, BrowserProvider]] = {}
_generation = 0
_scoped_generations: Dict[str, int] = {}
_lock = threading.Lock()


def register_provider(provider: BrowserProvider, *, scope: Optional[str] = None) -> None:
    """Register a cloud browser provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — makes hot-reload scenarios (tests, dev loops) behave
    predictably.
    """
    if not isinstance(provider, BrowserProvider):
        raise TypeError(
            f"register_provider() expects a BrowserProvider instance, "
            f"got {type(provider).__name__}"
        )
    raw_name = provider.name
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Browser provider .name must be a non-empty string")
    name = raw_name.strip()
    global _generation
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        existing = target.get(name)
        target[name] = provider
        if scope is None:
            _generation += 1
        else:
            _scoped_generations[scope] = _scoped_generations.get(scope, 0) + 1
    if existing is not None:
        logger.debug(
            "Browser provider '%s' re-registered (was %r)",
            name, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered browser provider '%s' (%s)",
            name, type(provider).__name__,
        )


def list_providers(*, scope: Optional[str] = None) -> List[BrowserProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        merged = dict(_providers)
        merged.update(_scoped_providers.get(scope or hermes_home_key(), {}))
        items = list(merged.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str, *, scope: Optional[str] = None) -> Optional[BrowserProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    with _lock:
        key = name.strip()
        return _scoped_providers.get(scope or hermes_home_key(), {}).get(key) or _providers.get(key)


def snapshot_registration(
    name: str, *, scope: Optional[str] = None
) -> Optional[BrowserProvider]:
    with _lock:
        target = _providers if scope is None else _scoped_providers.get(scope, {})
        return target.get(name.strip())


def registry_generation(*, scope: Optional[str] = None) -> tuple[int, int]:
    """Return a cache fingerprint for the global base and one profile."""
    active_scope = scope or hermes_home_key()
    with _lock:
        return _generation, _scoped_generations.get(active_scope, 0)


def restore_registration(
    name: str,
    current: BrowserProvider,
    previous: Optional[BrowserProvider],
    *,
    scope: Optional[str] = None,
) -> bool:
    """Restore a plugin registration only when *current* is still installed."""
    key = name.strip()
    global _generation
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        if target.get(key) is not current:
            return False
        if previous is None:
            target.pop(key, None)
        else:
            target[key] = previous
        if scope is None:
            _generation += 1
        else:
            _scoped_generations[scope] = _scoped_generations.get(scope, 0) + 1
            if not target:
                _scoped_providers.pop(scope, None)
    return True


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------


# Legacy auto-detect order — used when no ``browser.cloud_provider`` is set.
# Matches the pre-migration walk in :func:`tools.browser_tool._get_cloud_provider`.
# Firecrawl is intentionally absent so users with ``FIRECRAWL_API_KEY`` set
# for web-extract don't get silently routed to a paid cloud browser. See
# :func:`_resolve` for the full rationale.
_LEGACY_PREFERENCE = (
    "browser-use",
    "browserbase",
)


def _resolve(configured: Optional[str]) -> Optional[BrowserProvider]:
    """Resolve the active browser provider.

    Resolution rules (in order):

    1. **Explicit "local".** Returns None — the dispatcher disables cloud
       mode entirely. Mirrors legacy short-circuit in
       :func:`tools.browser_tool._get_cloud_provider`.
    2. **Explicit config wins, ignoring availability.** If ``configured``
       names a registered provider, return it even if its
       :meth:`is_available` returns False — the dispatcher will surface a
       precise "X_API_KEY is not set" error instead of silently routing
       somewhere else.
    3. **Legacy preference walk, filtered by availability.** Walk
       :data:`_LEGACY_PREFERENCE` (``browser-use`` → ``browserbase``) looking
       for a provider whose ``is_available()`` is True.

    There is intentionally NO "single-eligible shortcut" rule here (unlike
    :func:`agent.web_search_registry._resolve`). Pre-migration, the
    auto-detect branch in ``tools.browser_tool._get_cloud_provider`` only
    considered Browser Use and Browserbase; Firecrawl was reachable only
    via an explicit ``browser.cloud_provider: firecrawl`` config key.
    Preserving that gate matters because Firecrawl shares its API key with
    the *web* extract plugin (``plugins/web/firecrawl/``), so users who set
    ``FIRECRAWL_API_KEY`` for web extract must NOT get silently routed to a
    paid cloud browser on a fresh install. Third-party browser-provider
    plugins added under ``~/.hermes/plugins/browser/<vendor>/`` are subject
    to the same gate — they must be explicitly configured to take effect.

    Returns None when no provider is configured AND no available provider
    matches the legacy preference; the dispatcher then falls back to local
    browser mode.
    """
    with _lock:
        snapshot = dict(_providers)
        snapshot.update(_scoped_providers.get(hermes_home_key(), {}))

    def _is_available_safe(p: BrowserProvider) -> bool:
        """Wrap ``is_available()`` so a buggy provider doesn't kill resolution."""
        try:
            return bool(p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Browser provider %s.is_available() raised %s — treating as unavailable",
                p.name, exc, exc_info=True,
            )
            return False

    # 1. Explicit "local" short-circuit.
    if configured == "local":
        return None

    # 2. Explicit config wins — return regardless of is_available() so the
    #    user gets a precise downstream error message rather than a silent
    #    backend switch. Matches _get_cloud_provider() in browser_tool.py.
    if configured:
        provider = snapshot.get(configured)
        if provider is not None:
            return provider
        logger.debug(
            "browser cloud_provider '%s' configured but not registered; "
            "falling back to auto-detect",
            configured,
        )

    # 3. Legacy preference walk — only providers in _LEGACY_PREFERENCE are
    #    auto-eligible. Filtered by availability so we don't surface a
    #    provider the user has no credentials for. See docstring for why
    #    we do NOT fall back to "any single-eligible registered provider".
    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if provider is not None and _is_available_safe(provider):
            return provider

    return None


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    global _generation
    with _lock:
        _providers.clear()
        _scoped_providers.clear()
        _scoped_generations.clear()
        _generation += 1
