"""Pinned tool acquisition for PM's private Python engine; never worker API."""
from __future__ import annotations

from pathlib import Path

from pm.filesystem import native
from pm.package import InstallError
from pm.registry import get_package
from pm.store import current_target


def _toolchain(*, realize: bool = True, explicit: bool = False) -> tuple[Path, Path] | None:
    """Resolve uv and Python without host discovery or recursive worker dispatch.

    A read-only probe never installs. A sealed payload builds on its own
    interpreter: a venv's redirectors then name a path inside the package, so
    nothing may run them (``pm.environments.venv_command`` enters such a venv).
    """
    from pm.install import _installed_location, _lockfile, ensure

    if realize:
        ensure("uv", explicit=explicit)

    lockfile = _lockfile()
    target = current_target()
    binaries = {}
    for name in ("uv", "python"):
        package = get_package(name)
        location = _installed_location(package, lockfile, target)
        if location is None:
            if not realize:
                return None
            raise InstallError(name, "pinned tool is unavailable", "run `hermes pm install`")
        facts, store = location
        binary = package.binary(store.entry(facts.get(name)["entry"]), target)
        if binary is None or not binary.is_file():
            if not realize:
                return None
            raise InstallError(name, "installed binary is missing", "run `hermes pm install`")
        binaries[name] = binary
    # Callers execute and compare these outside PM's own file calls: the ordinary spelling, not the store's.
    return Path(native(binaries["uv"])), Path(native(binaries["python"]))
