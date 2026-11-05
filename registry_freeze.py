"""Freezing the process-wide registries once a hosted boot has finished.

A hosted process serves many tenants' agents from one set of registries: tools, MCP servers,
model providers and plugins. A registration made while one agent's turn runs would change what
every other agent sees. So the hosted boot loads everything it serves, then calls
:func:`freeze_registries`; from then on each registry's mutating entry points raise
:class:`RegistryFrozen`, and reads keep working. A per-agent toggle (a disabled tool, a switched-on
plugin) is a filter applied from the bound scope at read time, never a registration.

A local install never calls :func:`freeze_registries`: plugins, MCP servers and providers keep
loading, reloading and unloading at runtime as they always have. The freeze is one-way; there is
no thaw.
"""

from __future__ import annotations

_frozen = False


class RegistryFrozen(RuntimeError):
    """A registry was changed after the hosted boot froze the registries."""


def freeze_registries() -> None:
    """Make the tool, MCP, provider and plugin registries immutable for the rest of the process.
    Called once by the hosted boot after it has loaded everything it serves."""
    global _frozen
    _frozen = True


def registries_frozen() -> bool:
    return _frozen


def refuse_if_frozen(registry: str) -> None:
    """Raise :class:`RegistryFrozen` if the registries are frozen; *registry* names the one the
    caller was about to change."""
    if _frozen:
        raise RegistryFrozen(
            f"the {registry} is frozen: the hosted boot has finished, so registrations are fixed "
            "for the process. Load it during boot, or filter it per agent from the bound scope")
