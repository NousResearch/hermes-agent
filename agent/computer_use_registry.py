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
from typing import List, Optional

from agent.computer_use_provider import ComputerUseProvider
from agent.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)

#: The built-in provider: cua-driver spawned on whatever host runs the gateway.
HOST_PROVIDER_NAME = "local"

#: Names users and older configs already use for the built-in backend. Unset
#: lands here too, so the default is the behavior that predates the registry.
_HOST_ALIASES = {"", "local", "cua", "cua-driver", "builtin", "host"}

_registry: ProviderRegistry[ComputerUseProvider] = ProviderRegistry(
    label="Computer use", provider_cls=ComputerUseProvider, logger=logger,
)
_registry.export(globals())


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

    provider = _registry.get_provider(name)

    if provider is None:
        raise UnknownComputerUseProvider(name, [p.name for p in _registry.list_providers()])

    return provider
