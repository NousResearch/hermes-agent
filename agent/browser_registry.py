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
2. If exactly one registered provider is available, use it.
3. Legacy preference order — ``browser-use`` → ``browserbase`` — filtered by
   availability. Matches the historic auto-detect order in
   :func:`tools.browser_tool._get_cloud_provider` (Browser Use checked first
   because it covers both the managed Nous gateway and direct API key path;
   Browserbase as the older direct-credentials fallback). ``firecrawl`` is
   intentionally NOT in the legacy walk — users only get Firecrawl as a
   cloud browser when they explicitly set ``browser.cloud_provider:
   firecrawl``, matching pre-migration behaviour where Firecrawl was never
   auto-selected.
4. Otherwise ``None`` — the dispatcher falls back to local browser mode.

The explicit-config branch (rule 1) intentionally ignores ``is_available()``
so the dispatcher surfaces a typed "X_API_KEY is not set" error to the user
instead of silently switching backends. Matches the legacy
:func:`tools.browser_tool._get_cloud_provider` behaviour for configured names.

Note: there is no "capability" split here (unlike the web subsystem, which
has search/extract/crawl). Every browser provider implements the full
:class:`agent.browser_provider.BrowserProvider` lifecycle; the registry's
job is purely selection, not capability routing.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.browser_provider import BrowserProvider

logger = logging.getLogger(__name__)


_providers: Dict[str, BrowserProvider] = {}
_lock = threading.Lock()


def register_provider(provider: BrowserProvider) -> None:
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
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Browser provider .name must be a non-empty string")
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
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


def list_providers() -> List[BrowserProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        items = list(_providers.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str) -> Optional[BrowserProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    with _lock:
        return _providers.get(name.strip())


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------


# Legacy preference order — preserves behaviour for users who set no
# ``browser.cloud_provider`` config key. Matches the historic auto-detect
# order in :func:`tools.browser_tool._get_cloud_provider` (Browser Use first
# because it covers both managed Nous gateway and direct API key; Browserbase
# second as the older direct-credentials fallback). Filtered by
# ``is_available()`` at walk time so we don't surface a provider the user
# has no credentials for.
#
# Note: ``firecrawl`` is intentionally absent. Pre-migration, the auto-detect
# branch only considered Browser Use → Browserbase; Firecrawl was reachable
# only via an explicit ``browser.cloud_provider: firecrawl`` config key.
# Preserving that gate prevents users with a ``FIRECRAWL_API_KEY`` set for
# web-extract from accidentally getting routed to a (paid) cloud browser.
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
    3. **Single-provider shortcut.** When only one registered provider
       reports ``is_available() == True``, return it.
    4. **Legacy preference walk, filtered by availability.** Walk
       :data:`_LEGACY_PREFERENCE` (``browser-use`` → ``browserbase``) looking
       for a provider whose ``is_available()`` is True.

    Returns None when no provider is configured AND no available provider
    matches the legacy preference; the dispatcher then falls back to local
    browser mode.
    """
    with _lock:
        snapshot = dict(_providers)

    def _is_available_safe(p: BrowserProvider) -> bool:
        """Wrap ``is_available()`` so a buggy provider doesn't kill resolution."""
        try:
            return bool(p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider %s.is_available() raised %s", p.name, exc)
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

    # 3. + 4. Auto-detect path — filter by availability so we don't surface
    #    a provider the user has no credentials for.
    eligible = [p for p in snapshot.values() if _is_available_safe(p)]
    if len(eligible) == 1:
        return eligible[0]

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if provider is not None and _is_available_safe(provider):
            return provider

    return None


def get_active_browser_provider() -> Optional[BrowserProvider]:
    """Resolve the currently-active cloud browser provider.

    Reads ``browser.cloud_provider`` from config.yaml; falls back per the
    module docstring. Returns None for local mode or when no provider is
    available.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
    except Exception as exc:
        logger.debug("Could not read browser config: %s", exc)
        browser_cfg = {}

    configured: Optional[str] = None
    if isinstance(browser_cfg, dict) and "cloud_provider" in browser_cfg:
        try:
            from tools.tool_backend_helpers import normalize_browser_cloud_provider

            configured = normalize_browser_cloud_provider(
                browser_cfg.get("cloud_provider")
            )
        except Exception as exc:
            logger.debug("normalize_browser_cloud_provider failed: %s", exc)
            configured = None

    return _resolve(configured)


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _providers.clear()
