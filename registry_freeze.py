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

import weakref
from typing import Callable

_frozen = False
# Registries that hold payloads callers can reach (the tool registry's schemas and entries, the
# provider profiles) seal them at the freeze. A registry instance's bound method is held weakly so a
# registry built in a test does not live for the process; a module function is held as is.
_sealers: list[Callable[[], Callable[[], None] | None]] = []


# Registries whose boot can leave work that only completes by registering later (a deferred
# platform loader registers its adapter on first lookup) report what is still pending; the freeze
# refuses while anything is, because that registration would then be refused and the entry lost.
_completeness_checks: list[Callable[[], list[str]]] = []


class RegistryFrozen(RuntimeError):
    """A registry was changed after the hosted boot froze the registries."""


class RegistriesIncomplete(RuntimeError):
    """:func:`freeze_registries` was called while a registry still had registrations pending."""


def check_at_freeze(pending: Callable[[], list[str]]) -> None:
    """Have :func:`assert_registries_complete` (and so :func:`freeze_registries`) call *pending*,
    which names the registrations a registry still owes (empty when complete)."""
    _completeness_checks.append(pending)


def assert_registries_complete() -> None:
    """Raise :class:`RegistriesIncomplete` naming every registration still pending (an unresolved
    deferred platform loader). The hosted boot materialises each platform it serves, and drops the
    rest, before it freezes."""
    pending = sorted(name for check in _completeness_checks for name in check())
    if pending:
        raise RegistriesIncomplete(
            "cannot freeze the registries while registrations are pending: " + ", ".join(pending)
            + ". Resolve them during boot (a deferred platform: look it up once) or drop them first")


def freeze_registries() -> None:
    """Make the tool, MCP, provider and plugin registries immutable for the rest of the process.
    Called once by the hosted boot after it has loaded everything it serves.

    Refuses (:class:`RegistriesIncomplete`, nothing frozen) while a registration is still pending,
    such as a deferred platform loader nobody has looked up: after the freeze its adapter could not
    register. Hooks are global too: a per-agent memory provider constructed after the freeze must
    not register global hooks (``ctx.register_hook``); its plugin registers them during boot, and
    stable dispatch to per-agent provider instances is a follow-up, not something the freeze
    provides."""
    global _frozen
    if _frozen:
        return
    assert_registries_complete()
    _frozen = True  # first: a registration racing the freeze is refused, so sealing sees the final set
    for ref in _sealers:
        seal = ref()
        if seal is not None:
            seal()


def seal_at_freeze(seal: Callable[[], None]) -> None:
    """Have :func:`freeze_registries` call *seal* (a registry's bound method, or a module function),
    which replaces the registry's reachable payloads with read-only copies. Runs at once when the
    registries are already frozen."""
    _sealers[:] = [ref for ref in _sealers if ref() is not None]
    _sealers.append(weakref.WeakMethod(seal) if hasattr(seal, "__self__") else (lambda: seal))
    if _frozen:
        seal()


def registries_frozen() -> bool:
    return _frozen


def refuse_if_frozen(registry: str) -> None:
    """Raise :class:`RegistryFrozen` if the registries are frozen; *registry* names the one the
    caller was about to change."""
    if _frozen:
        raise RegistryFrozen(
            f"the {registry} is frozen: the hosted boot has finished, so registrations are fixed "
            "for the process. Load it during boot, or filter it per agent from the bound scope")
