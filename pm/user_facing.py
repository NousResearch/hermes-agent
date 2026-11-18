"""The terminal's PATH: the user's own tools first, Hermes's after them.

Hermes's process puts its store's tool dirs first (``pm.install.activate``), so Hermes's own
code resolves the pinned binaries. Terminal commands belong to the user and the agent, so they
run the tools the user installed, and Hermes's tools are the fallback for a machine without
them. Hermes's own operations must not depend on this PATH: they resolve pinned tools
through PM (``pm.env_for``, ``pm.installed_package``).

A sealed Windows payload's store sits under ``WindowsApps``. In the process tree that the
app's execution alias starts, a program whose own executable is outside the package cannot
start one inside it: cmd.exe running npm's ``npx.cmd`` gets "Access is denied" from the
packaged ``node.exe``, and a venv's ``Scripts\\python.exe`` redirector exits 101 when it
starts the packaged Python. A copy of the same files outside the package starts without
error, so there :func:`copies` swaps those dirs for PM's verified copies.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)

# Packages whose packaged dirs a sealed Windows payload swaps for writable copies.
COPIED = ("npm", "node", "python")


def applies() -> bool:
    """True for a sealed payload on Windows, the only layout with the restriction above."""
    from pm.install import sealed
    from pm.store import current_target

    return current_target().startswith("win32") and sealed()


def writable_copy(name: str, *, realize: bool = True, explicit: bool = False,
                  why: str = "a copy outside the package is required"):
    """``(facts, store)`` of PM's verified copy of the installed *name* in the writable store.

    The first call copies the installed entry and checks its digest. It downloads nothing.
    If the install already lives in the writable store, that install is the answer. Returns
    ``None`` when *name* is not installed, or when no copy exists and *realize* is false.
    Raises the lazy-install refusal, naming *why*, when a copy is needed and lazy installs
    are off.
    """
    from pm import paths
    from pm.filesystem import long_root
    from pm.install import (_install, _installed_location, _lockfile, _refresh_found_env, _refuse_lazy,
                            lazy_installs_allowed)
    from pm.lock import Facts
    from pm.registry import get_package
    from pm.store import Store, current_target

    package, lockfile, target = get_package(name), _lockfile(), current_target()
    location = _installed_location(package, lockfile, target)
    if location is None:
        return None
    # Spelled like Store spells its root, so the comparison and the copy's file calls agree.
    writable = long_root(paths.writable_store_root())
    if location[1].root == writable:
        return location
    copied = _installed_location(package, lockfile, target, verify=explicit, roots=(writable,))
    if copied is not None and realize:
        # Another install's copy of this pin: running on it makes this checkout an owner (#149497).
        _refresh_found_env(package, copied, target, None)
    if copied is not None or not realize:
        return copied
    if not explicit and not lazy_installs_allowed():
        raise _refuse_lazy(name, why)
    copy_store, copy_facts = Store(writable), Facts(writable / "facts.json")
    _install(package, lockfile, copy_facts, copy_store, target, copy_from=location)
    return copy_facts, copy_store


def _dirs(facts, store, name: str) -> list[str]:
    """The PATH dirs that *store*'s install of *name* contributes, natively spelled."""
    from pm.filesystem import native

    dirs = facts.env_for(name, store.root).get("PATH") or []
    return [native(d) for d in ([dirs] if isinstance(dirs, str) else dirs) if d]


def _sealed_dirs(name: str) -> list[str]:
    from pm.install import _facts, _store

    facts, store = _facts(), _store()
    return _dirs(facts, store, name) if facts.installed(name, None, store.root) else []


def _copy_dirs(name: str) -> list[str]:
    """The PATH dirs of *name*'s writable copy, made on first use, or ``[]`` when PM may not make one."""
    from pm.package import InstallError

    try:
        location = writable_copy(name)
    except (InstallError, OSError) as exc:
        LOG.debug("user-facing %s stays the packaged copy: %s", name, exc)
        return []
    return _dirs(*location, name) if location is not None else []


def copies(dirs: list[str]) -> list[str]:
    """*dirs* with each packaged npm, node and python dir swapped for its writable copy on a
    sealed Windows payload. A packaged dir stays when PM may not copy it (lazy installs off)."""
    if not applies():
        return list(dirs)
    swap: dict[str, list[str]] = {}
    for name in COPIED:
        sealed = _sealed_dirs(name)
        copied = _copy_dirs(name) if sealed else []
        if copied:
            swap.update((os.path.normcase(d), copied) for d in sealed)
    return list(dict.fromkeys(c for d in dirs for c in swap.get(os.path.normcase(d), [d])))


# The app-alias dir's verdict, keyed by the dir and its mtime: the glob it answers with is
# 0.6ms, and the terminal asks before every command.
_STUB_DIRS: dict[tuple[str, int], bool] = {}


def _python_stub_dir(directory: str) -> bool:
    """True for the Windows app-alias dir when its ``python.exe`` is only the Store stub.

    The stub opens the Microsoft Store instead of running Python. A real Store Python puts its
    aliases in the same dir, and it also ships a ``python3.N.exe``, which the stub does not.

    Asked for every PATH entry before each command, and answering it globs the alias dir: the
    verdict is remembered until that dir's entries change.
    """
    # The shape check below builds Paths for every entry; a plain substring test is what
    # keeps the common (non-alias) entry free.
    if "windowsapps" not in directory.lower():
        return False
    parts = [os.path.normcase(part) for part in Path(directory).parts[-2:]]
    if parts != [os.path.normcase("Microsoft"), os.path.normcase("WindowsApps")]:
        return False
    try:
        stamp = os.stat(directory).st_mtime_ns
    except OSError:
        # No readable dir means no real Python beside the alias.
        return True
    key = (os.path.normcase(directory), stamp)
    known = _STUB_DIRS.get(key)
    if known is None:
        try:
            known = not any(Path(directory).glob("python3.*.exe"))
        except OSError:
            known = True
        if len(_STUB_DIRS) >= 32:
            _STUB_DIRS.clear()
        _STUB_DIRS[key] = known
    return known


def _python_dirs() -> list[str]:
    """PM's Python dirs, memoized: ``env_for`` parses the lock and the installed state and
    stats the whole dependency chain, and the terminal asks for this before every command."""
    from pm import paths
    from pm.install import _memo_lookup, _memo_store, _state_key, _witness, env_for

    key = ("python_dirs", *_state_key(), env_for)
    cached = _memo_lookup("python_dirs", key)
    if cached is not None:
        return list(cached)
    dirs = tuple(d for d in
                 env_for("python", base_env={"PATH": ""}).get("PATH", "").split(os.pathsep) if d)
    _memo_store("python_dirs", key, dirs,
                (_witness(paths.facts_path()), _witness(paths.lockfile_path()),
                 *(_witness(directory) for directory in dirs)))
    return list(dirs)


def user_facing_path(path: str) -> str:
    """*path* with Hermes's store tool dirs (their writable copies, see :func:`copies`) moved
    behind the user's own entries.

    Hermes's Python stops just before the app-alias dir that holds the Store stub, so a
    machine without Python still runs a real one.
    """
    from pm.install import _store_path_dirs

    store = _store_path_dirs()
    if not store:
        return path
    ours = copies(store)
    keys = {os.path.normcase(d) for d in (*store, *ours)}
    kept = [p for p in path.split(os.pathsep) if p and os.path.normcase(p) not in keys]
    python = {os.path.normcase(d) for d in copies(_python_dirs())}
    cut = next((i for i, p in enumerate(kept) if _python_stub_dir(p)), len(kept))
    early = [d for d in ours if os.path.normcase(d) in python]
    late = [d for d in ours if os.path.normcase(d) not in python]
    return os.pathsep.join(dict.fromkeys([*kept[:cut], *early, *kept[cut:], *late]))
