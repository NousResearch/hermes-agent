"""Pinned tool acquisition for PM's private Python engine; never worker API."""
from __future__ import annotations

from pathlib import Path

from pm import paths
from pm.filesystem import long_root, native
from pm.lock import Facts
from pm.package import InstallError
from pm.registry import get_package
from pm.store import Store, current_target


def _toolchain(*, realize: bool = True, explicit: bool = False) -> tuple[Path, Path] | None:
    """Resolve uv and Python without host discovery or recursive worker dispatch.

    A read-only probe never installs. A sealed Windows payload builds on a verified writable
    copy of its Python: uv starts a venv's ``Scripts\\python.exe`` to query it, and that
    redirector runs outside the package, so its target must too. A redirector naming the
    packaged interpreter dies with exit 101 (#135236). Hermes itself still enters those venvs
    on the packaged interpreter (``pm.environments.venv_command``).
    """
    from pm.install import (
        _install, _installed_location, _lockfile, _refuse_lazy,
        ensure, lazy_installs_allowed, sealed,
    )

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
            # Spelled like Store spells its root, so the comparison and the copy's file calls agree.
            writable = long_root(paths.writable_store_root())
            if location[1].root != writable:
                copied = _installed_location(package, lockfile, target, verify=explicit, roots=(writable,))
                if copied is None:
                    if not realize:
                        return None
                    if not explicit and not lazy_installs_allowed():
                        raise _refuse_lazy(name, "a writable Python is required for bundled builds")
                    copy_store = Store(writable)
                    copy_facts = Facts(writable / "facts.json")
                    _install(package, lockfile, copy_facts, copy_store, target, copy_from=location)
                    copied = copy_facts, copy_store
                location = copied
        facts, store = location
        binary = package.binary(store.entry(facts.get(name)["entry"]), target)
        if binary is None or not binary.is_file():
            if not realize:
                return None
            raise InstallError(name, "installed binary is missing", "run `hermes pm install`")
        binaries[name] = binary
    # Callers execute and compare these outside PM's own file calls: the ordinary spelling, not the store's.
    return Path(native(binaries["uv"])), Path(native(binaries["python"]))
