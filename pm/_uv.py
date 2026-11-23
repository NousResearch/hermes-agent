"""Pinned tool acquisition for PM's private Python engine; never worker API."""
from __future__ import annotations

from pathlib import Path

from pm.filesystem import native
from pm.package import InstallError
from pm.registry import get_package
from pm.store import current_target


def _toolchain(*, realize: bool = True, explicit: bool = False) -> tuple[Path, Path] | None:
    """Resolve uv and Python without host discovery or recursive worker dispatch.

    A read-only probe never installs. A sealed Windows payload builds on a verified writable
    copy of its Python: uv starts a venv's ``Scripts\\python.exe`` to query it, and that
    redirector runs outside the package, so its target must too. A redirector naming the
    packaged interpreter dies with exit 101 (#135236). Hermes itself still enters those venvs
    on the packaged interpreter (``pm.environments.venv_command``).
    """
    from pm.install import _installed_location, _lockfile, ensure, sealed
    from pm.user_facing import writable_copy

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
        if name == "python" and target.startswith("win32") and sealed():
            location = writable_copy(name, realize=realize, explicit=explicit,
                                     why="a writable Python is required for bundled builds")
            if location is None:
                return None
        facts, store = location
        binary = package.binary(store.entry(facts.get(name)["entry"]), target)
        if binary is None or not binary.is_file():
            if not realize:
                return None
            raise InstallError(name, "installed binary is missing", "run `hermes pm install`")
        binaries[name] = binary
    # Callers execute and compare these outside PM's own file calls: the ordinary spelling, not the store's.
    return Path(native(binaries["uv"])), Path(native(binaries["python"]))
