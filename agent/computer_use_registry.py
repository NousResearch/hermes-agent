"""Computer Use Provider Registry
================================

Central map of registered computer-use providers. Populated by plugins at
import-time via :func:`register_provider`; consumed by
:func:`tools.computer_use.tool._get_active_cu_provider` to route each
``computer_use`` tool call to a per-task backend supplied by the active
provider.

Active selection
----------------
The active provider is chosen by configuration with this precedence:

1. ``computer_use.provider`` in ``config.yaml`` (explicit override).
   ``local`` (or the builtin names ``cua`` / ``cua-driver``) short-circuits
   to ``None`` — the dispatcher uses the legacy host-spawned singleton
   ``CuaDriverBackend``.
2. Otherwise ``None`` — the dispatcher falls back to the legacy singleton.

The explicit-config branch (rule 1) intentionally ignores
:meth:`is_available` so the dispatcher surfaces a typed provider error to
the user instead of silently falling back to the host singleton. This
mirrors :func:`agent.browser_registry._resolve` (PR #25214) bit-for-bit:
a configured-but-unavailable provider is a misconfiguration the user should
hear about, not a silent mode switch.

There is intentionally NO legacy preference walk here (unlike the browser
registry): computer_use had no pre-existing auto-detected cloud providers,
so any provider must be explicitly configured to take effect. Third-party
plugins added under ``~/.hermes/plugins/computer_use/<vendor>/`` are subject
to the same explicit-config gate — they never auto-activate.

Note: there is no "capability" split here. Every computer-use provider
implements the full
:class:`agent.computer_use_provider.ComputerUseProvider` lifecycle; the
registry's job is purely selection, not capability routing.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.computer_use_provider import ComputerUseProvider

logger = logging.getLogger(__name__)


_providers: Dict[str, ComputerUseProvider] = {}
_lock = threading.Lock()


# Config values that mean "use the legacy host-spawned singleton, not a
# registered provider." ``local`` mirrors the browser registry's
# short-circuit; ``cua`` / ``cua-driver`` / empty cover the names users
# already use for the builtin backend.
_LEGACY_SENTINELS = {"", "local", "cua", "cua-driver", "builtin"}


def register_provider(provider: ComputerUseProvider) -> None:
    """Register a computer-use provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — makes hot-reload scenarios (tests, dev loops) behave
    predictably. Mirrors :func:`agent.browser_registry.register_provider`.
    """
    if not isinstance(provider, ComputerUseProvider):
        raise TypeError(
            f"register_provider() expects a ComputerUseProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Computer use provider .name must be a non-empty string")
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
    if existing is not None:
        logger.debug(
            "Computer use provider '%s' re-registered (was %r)",
            name, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered computer use provider '%s' (%s)",
            name, type(provider).__name__,
        )


def list_providers() -> List[ComputerUseProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        items = list(_providers.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str) -> Optional[ComputerUseProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    with _lock:
        return _providers.get(name.strip())


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------


def _resolve(configured: Optional[str]) -> Optional[ComputerUseProvider]:
    """Resolve the active computer-use provider.

    Resolution rules (in order):

    1. **Legacy sentinel / unset.** Returns None — the dispatcher uses the
       host-spawned singleton ``CuaDriverBackend``. Covers ``local``,
       ``cua``, ``cua-driver``, ``builtin``, and unset.
    2. **Explicit config wins, ignoring availability.** If ``configured``
       names a registered provider, return it even if its
       :meth:`is_available` returns False — the dispatcher will surface a
       precise provider error instead of silently falling back to the host
       singleton. Mirrors :func:`agent.browser_registry._resolve`.

    Returns None when no provider is configured (or the value is a legacy
    sentinel); the dispatcher then falls back to the host singleton.
    """
    with _lock:
        snapshot = dict(_providers)

    if not configured or configured in _LEGACY_SENTINELS:
        return None

    provider = snapshot.get(configured)
    if provider is not None:
        return provider
    logger.debug(
        "computer_use provider '%s' configured but not registered; "
        "falling back to host singleton",
        configured,
    )
    return None


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _providers.clear()
