"""Registry of computer-use providers, and the rule for picking one.

Providers register at import time — the built-in host backend from
:mod:`tools.computer_use.host_provider`, third-party runtimes from their
plugin's ``register(ctx)`` via
:meth:`hermes_cli.plugins.PluginContext.register_computer_use_provider`.
:func:`resolve_provider` then answers which one services a call, from the
``computer_use.provider`` key in ``config.yaml``.

Selection is explicit, never inferred. computer_use has no history of
auto-detected cloud backends to preserve, and guessing here would silently
move where the agent's clicks land — so an unset key means the host backend
and nothing else auto-activates.

A configured name that nobody registered raises rather than falling back.
Quietly reverting to the host backend would drive the user's own desktop when
they had asked for a container, which is the one outcome worth crashing over;
:func:`tools.computer_use.tool.handle_computer_use` turns it into a message
naming the missing provider.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.computer_use_provider import ComputerUseProvider

logger = logging.getLogger(__name__)

#: The built-in provider: cua-driver spawned on whatever host runs the gateway.
HOST_PROVIDER_NAME = "local"

#: Names users and older configs already use for the built-in backend. Unset
#: lands here too, so the default is the behavior that predates the registry.
_HOST_ALIASES = {"", "local", "cua", "cua-driver", "builtin", "host"}

_providers: Dict[str, ComputerUseProvider] = {}
_lock = threading.Lock()


class UnknownComputerUseProvider(LookupError):
    """``computer_use.provider`` names a provider that is not registered."""

    def __init__(self, configured: str, available: List[str]):
        self.configured = configured
        self.available = available
        known = ", ".join(available) or "none"
        super().__init__(
            f"computer_use.provider is set to {configured!r}, but no such provider "
            f"is registered (available: {known}). Install the plugin that provides "
            f"it, or set computer_use.provider to '{HOST_PROVIDER_NAME}' in config.yaml."
        )


def register_provider(provider: ComputerUseProvider) -> None:
    """Register a provider. Re-registering the same name replaces it."""
    if not isinstance(provider, ComputerUseProvider):
        raise TypeError(
            "register_provider() expects a ComputerUseProvider instance, "
            f"got {type(provider).__name__}"
        )

    name = provider.name

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Computer use provider .name must be a non-empty string")

    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider

    if existing is not None:
        logger.debug("Computer use provider %r replaced %s", name, type(existing).__name__)


def list_providers() -> List[ComputerUseProvider]:
    """Every registered provider, sorted by name."""
    with _lock:
        return sorted(_providers.values(), key=lambda p: p.name)


def get_provider(name: str) -> Optional[ComputerUseProvider]:
    """The provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None

    with _lock:
        return _providers.get(name.strip())


def snapshot_registration(name: str) -> Optional[ComputerUseProvider]:
    """What is registered under *name* right now, for unload restore."""
    return get_provider(name)


def restore_registration(
    name: str,
    current: ComputerUseProvider,
    previous: Optional[ComputerUseProvider],
) -> None:
    """Undo one registration when its plugin unloads.

    A no-op unless *current* is still the live entry: a third plugin that
    registered the same name afterwards owns it now, and rolling back to our
    predecessor would silently take its provider away.
    """
    with _lock:
        if _providers.get(name) is not current:
            return

        if previous is None:
            _providers.pop(name, None)
        else:
            _providers[name] = previous


def resolve_provider(configured: Optional[str]) -> ComputerUseProvider:
    """Return the provider that should service calls.

    Raises :class:`UnknownComputerUseProvider` when *configured* names one
    that is not registered — including the host provider, whose absence means
    ``tools.computer_use`` was never imported and the caller has a real
    layering bug rather than a typo.
    """
    name = (configured or "").strip().lower()

    if name in _HOST_ALIASES:
        name = HOST_PROVIDER_NAME

    provider = get_provider(name)

    if provider is None:
        raise UnknownComputerUseProvider(name, [p.name for p in list_providers()])

    return provider


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _providers.clear()
