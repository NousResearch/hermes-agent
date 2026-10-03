"""Heal mixed ``sys.modules`` after an in-place checkout update.

Pre-reexec updaters (Hermes ≤ v2026.9.14) purged only package prefixes
(``hermes_cli``, ``gateway``, ``tools``, ``tui_gateway``, ``agent``) and left
root modules like ``utils`` cached in the updater process. The post-pull
gateway-restart phase then imports new ``hermes_cli.gateway`` /
``hermes_cli.config`` into that process; those need symbols the stale
``utils`` lacks (``file_signature``), and ``hermes update`` exits 1 with
``gateway auto-restart failed: cannot import name 'file_signature' from
'utils'``.

Post-swap hand-off (``hermes_cli.update_handoff``) makes this class dead for
updaters that already include it. This module is the bridge for the one
upgrade from a pre-handoff release onto a tree that needs new root symbols:
freshly imported ``hermes_cli`` code drops the incomplete cache before
importing ``utils``.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Mapping, Sequence

# Root modules the narrow purge left behind, keyed by attributes that must
# exist on the on-disk copy after this release. Extend when a new root-level
# symbol would otherwise break the pre-handoff upgrade path.
_ROOT_MODULE_REQUIRED_ATTRS: dict[str, tuple[str, ...]] = {
    "utils": ("file_signature",),
}


def drop_stale_root_modules(
    required: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Drop cached root modules missing required attrs. Returns dropped names."""
    checks = _ROOT_MODULE_REQUIRED_ATTRS if required is None else required
    dropped: list[str] = []
    for name, attrs in checks.items():
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if any(not hasattr(mod, attr) for attr in attrs):
            sys.modules.pop(name, None)
            dropped.append(name)
    return dropped


# Packages the pre-handoff purge could not evict: it pops their ``sys.modules`` entries but the
# package objects themselves (and so the submodule attributes the import system set on them) live on.
_BRIDGED_PACKAGES = ("hermes_cli",)


def drop_stale_package_bindings() -> list[str]:
    """Drop submodule attributes that alias a module no longer in ``sys.modules``.

    ``from hermes_cli import main_dashboard`` resolves through ``getattr(package, name)`` and only
    imports when that attribute is *absent* (``importlib._bootstrap._handle_fromlist``), so a
    binding that outlived the purge hands fresh code the PRE-pull module object instead of
    re-importing it. That is how the pulled ``dashboard_procs._kill_stale_dashboard_processes``
    called ``_loaded_launchd_backend_jobs`` on the old ``main_dashboard`` and killed the update
    after ``✓ Update complete!`` (#115091). Deleting the stale attribute is enough: the next
    ``from ... import`` re-imports the name from the pulled tree. Returns dropped names.
    """
    dropped: list[str] = []
    for package in _BRIDGED_PACKAGES:
        pkg = sys.modules.get(package)
        if pkg is None:
            continue
        for name, value in list(vars(pkg).items()):
            full = f"{package}.{name}"
            # In sync (the attribute IS the live module) — leave it alone; a live entry with a
            # different object is the fresh one, so only the attribute is stale.
            if sys.modules.get(full) is value:
                continue
            if isinstance(value, ModuleType):
                delattr(pkg, name)
                dropped.append(full)
    return dropped


def drop_stale_modules() -> list[str]:
    """Import-time bridge entry point for the one upgrade off a pre-handoff updater release.

    Runs before any post-pull import chain can bind a symbol from the OLD tree. Both halves belong
    to every caller: root modules (``utils.file_signature``) and stale package bindings
    (``hermes_cli.main_dashboard``) are the same failure, one module apart. Returns dropped names.
    """
    return drop_stale_root_modules() + drop_stale_package_bindings()
