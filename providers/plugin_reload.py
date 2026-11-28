"""Load ``$HERMES_HOME`` provider plugins into a home layer, and swap in an updated one whole (#153300).

A provider plugin used to be imported once per process. An update then never reached a running gateway, and a
plugin importing part of its package on first use (inside ``create_client``) read that one file from the new
tree against siblings loaded from the old one.

Each tree is imported from its own frozen copy, under its own module name (the plugin's base name plus a
fingerprint of its files), into a staging layer, and only a complete import is published: the plugin's previous
registrations stay live until then, and stay live if the new tree fails to import. The previous package stays
in ``sys.modules`` with its ``__path__`` on its own copy, so anything built from it (a cached agent's client)
imports even its not-yet-loaded files from that version: every object runs one version end to end, never a mix.

The copies are read for the whole life of the process, so they live under ``<root>/runtime/provider-plugins`` (the
Hermes root, which deleting a profile never removes) and not in a temp dir that is cleaned by age (#156493). A
process that takes the setup lock makes an ``owned-`` root and holds an OS lock on its ``owner.lock`` until it exits;
the next such process reclaims the owned roots whose lock file nobody holds: their process is gone, killed or
crashed included. Creating a root and that sweep run under the setup lock, so a sweep never sees a root its owner has
made but not locked yet. A process that cannot take the setup lock makes an ``unswept-`` root instead, which no sweep
matches: removed on its normal exit, left behind by a crash.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from hermes_cli.home_data_layout import PROVIDER_PLUGIN_SNAPSHOT_PARTS
from hermes_constants import get_default_hermes_root, mkdir_under_hermes_home

logger = logging.getLogger(__name__)

# Directories inside a plugin tree that never hold code the provider imports.
_SKIP_DIRS = frozenset({"__pycache__", ".git", "node_modules", "tests", "evals", ".venv"})
_SNAPSHOT_IGNORE = shutil.ignore_patterns("__pycache__", ".git", "node_modules", ".venv")
# The directory is Hermes-private, and these names are reserved in it. A sweep reclaims "owned-" roots (made under
# the setup lock) once their owner.lock exists and nobody holds it, and finishes "_REAPING_PREFIX" claims; entries
# under any other name are never touched.
_OWNED_PREFIX = "owned-"
_UNSWEPT_PREFIX = "unswept-"
_REAPING_PREFIX = f".reaping-{_OWNED_PREFIX}"
_OWNER_LOCK = "owner.lock"
_SETUP_LOCK = ".setup.lock"
_SETUP_WAIT_SECONDS = 30
# One directory per process for the frozen copies; removed on a normal exit, reclaimed later otherwise.
_snapshot_root: Path | None = None
_snapshot_lock = threading.Lock()


def _owner_gone(root: Path) -> bool:
    """Whether *root* has an owner lock file that nobody holds. A root without one is not provably ours (a directory
    someone made by hand, or an owner that died between creating its root and locking it) and is left alone."""
    from pm.filesystem import lock_fd, unlock_fd

    try:
        fd = os.open(root / _OWNER_LOCK, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        if lock_fd(fd, wait=False):
            unlock_fd(fd)
            return True
        return False
    finally:
        os.close(fd)


def _claim_dead_roots(parent: Path) -> list[Path]:
    """Rename the owned roots whose owner is gone, and the claims a sweeper killed mid-delete left behind, to fresh
    names; the caller deletes them after releasing the setup lock, so a large dead root never holds up other
    processes starting. A claim another sweeper is still deleting can be taken again and deleted twice: both deletes
    ignore errors, so the cost is the duplicated work."""
    claimed = []
    with contextlib.suppress(OSError):
        for entry in parent.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            with contextlib.suppress(OSError):  # unreadable, or another user's: not ours to judge
                leftover = entry.name.startswith(_REAPING_PREFIX)
                if leftover or (entry.name.startswith(_OWNED_PREFIX) and _owner_gone(entry)):
                    claim = parent / f"{_REAPING_PREFIX}{uuid.uuid4().hex}"
                    os.rename(entry, claim)
                    claimed.append(claim)
    return claimed


def _release_root(root: Path, owner_fd: int | None, creator_pid: int) -> None:
    # A forked child inherits this hook and shares the flock (it belongs to the open file): its exit must
    # neither delete the parent's live copy nor drop the parent's lock.
    if os.getpid() != creator_pid:
        return
    if owner_fd is not None:
        with contextlib.suppress(OSError):
            os.close(owner_fd)
    shutil.rmtree(root, ignore_errors=True)


def _new_root() -> Path:
    """This process's root: ``owned-`` with its owner lock held until the process exits, or ``unswept-`` without one
    when the setup lock is unavailable."""
    from pm.filesystem import lock_path

    parent = mkdir_under_hermes_home(get_default_hermes_root().joinpath(*PROVIDER_PLUGIN_SNAPSHOT_PARTS))
    try:
        setup_fd = lock_path(parent / _SETUP_LOCK, timeout=_SETUP_WAIT_SECONDS)
    except PermissionError:  # another user's lock file: a home shared across UIDs
        setup_fd = None
    if setup_fd is None:
        # Without the setup lock a sweep could see this root before its owner lock is taken, so make it under a
        # name no sweep matches. It is still removed on a normal exit; after a crash it stays.
        logger.warning("Could not take %s; this process's provider plugin snapshots will not be reclaimed if it "
                       "crashes, and it reclaims no others", parent / _SETUP_LOCK)
        root = Path(tempfile.mkdtemp(prefix=_UNSWEPT_PREFIX, dir=parent))
        atexit.register(_release_root, root, None, os.getpid())
        return root
    claimed: list[Path] = []
    try:
        claimed = _claim_dead_roots(parent)
        root = Path(tempfile.mkdtemp(prefix=_OWNED_PREFIX, dir=parent))
        try:
            owner_fd = lock_path(root / _OWNER_LOCK, timeout=0)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        if owner_fd is None:
            shutil.rmtree(root, ignore_errors=True)
            raise OSError(f"{root / _OWNER_LOCK} is held by another process")
    finally:
        os.close(setup_fd)
        for claim in claimed:
            shutil.rmtree(claim, ignore_errors=True)
    atexit.register(_release_root, root, owner_fd, os.getpid())
    return root


def _snapshot(plugin_dir: Path, module_name: str) -> Path:
    """A frozen copy of *plugin_dir* for *module_name* (per home, plugin and tree), named like the original
    (load errors key on it). Copied into a private directory and published by one rename, so a concurrent scan
    never imports a half-written copy; a failed copy leaves nothing behind."""
    global _snapshot_root
    with _snapshot_lock:
        if _snapshot_root is None:
            _snapshot_root = _new_root()
    target = _snapshot_root / module_name / plugin_dir.name
    if target.is_dir():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix=".copy-", dir=target.parent))
    try:
        shutil.copytree(plugin_dir, private / plugin_dir.name, ignore=_SNAPSHOT_IGNORE)
        os.rename(private / plugin_dir.name, target)
    except OSError:
        if not target.is_dir():  # another scan publishing the same copy first is fine
            raise
    finally:
        shutil.rmtree(private, ignore_errors=True)
    return target


def tree_fingerprint(plugin_dir: Path) -> str:
    """Digest of the plugin's Python files and manifest (path, size, mtime): it moves whenever an update
    rewrites them, however the plugin was installed."""
    entries = []
    for root, dirs, files in os.walk(plugin_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".py") or name in ("plugin.yaml", "plugin.yml"):
                path = os.path.join(root, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                entries.append(f"{os.path.relpath(path, plugin_dir)}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha1("\n".join(entries).encode("utf-8")).hexdigest()[:12]


def import_into_layer(plugin_dir: Path, source: str, base_name: str) -> None:
    """Import *plugin_dir* into the layer being scanned unless that exact tree is already live there; an
    updated tree replaces the live one only once it imported completely."""
    import providers as P

    layer = P._REGISTRATION_TARGET.get()
    fingerprint = tree_fingerprint(plugin_dir)
    live = layer.trees.get(base_name)
    if live is not None and live[0] == fingerprint:
        return
    module_name = f"{base_name}_{fingerprint}"
    staging = P._HomeLayer()
    token = P._REGISTRATION_TARGET.set(staging)
    try:
        # In-process code (including a hosted plugin's in-process profiles) reads the frozen copy; the plugin
        # host keeps the installed path, the stable key of its extraction cache.
        P._import_user_module(plugin_dir, source, module_name, code_dir=_snapshot(plugin_dir, module_name))
    except OSError as exc:  # the frozen copy could not be made: keep serving the previous version
        logger.warning("Could not stage provider plugin %s: %s", plugin_dir.name, exc)
        staging.errors[plugin_dir.name] = f"{type(exc).__name__}: {exc}"
    finally:
        P._REGISTRATION_TARGET.reset(token)
    if module_name in sys.modules or module_name in staging.hosted:
        _publish(layer, base_name, module_name, fingerprint, staging, replacing=live is not None)
        # A hosted plugin can load with one of its profiles refused (plugins.isolation: host): that reason stays,
        # with the profiles this publish actually gave the layer.
        if refused := staging.errors.get(plugin_dir.name):
            if isinstance(refused, P.ProfileRefusal):
                refused = P.ProfileRefusal(refused, tuple(sorted(
                    name for name in staging.registry if layer.owner.get(name) == base_name)))
            layer.errors[plugin_dir.name] = refused
        else:
            layer.errors.pop(plugin_dir.name, None)
        if live is not None:
            logger.info("Provider plugin %s changed on disk; new sessions use the updated version",
                        plugin_dir.name)
        return
    # A failed import leaves the previous version serving; drop what it half-loaded so a retry is clean.
    prefix = module_name + "."
    for name in [m for m in sys.modules if m.startswith(prefix)]:
        del sys.modules[name]
    if error := staging.errors.get(plugin_dir.name):
        layer.errors[plugin_dir.name] = error


def _publish(layer, base_name: str, module_name: str, fingerprint: str, staging, *, replacing: bool) -> None:
    """Swap *base_name*'s registrations in *layer* for *staging*'s. Lookups take no lock, so a name kept by the
    new release is overwritten in place and the alias table is replaced whole: a reader sees the old entry or
    the new one, never neither. A first import keeps the scan's last-writer-wins order; a reload never takes a
    name another plugin currently owns."""
    import providers as P

    with P._HOME_LAYERS_LOCK:
        previous = {n for n, owner in layer.owner.items() if owner == base_name}
        for name, profile in staging.registry.items():
            if replacing and layer.owner.get(name, base_name) != base_name:
                continue
            layer.registry[name] = profile
            layer.owner[name] = base_name
        # The plugin's aliases go with its registrations: one the new release dropped stops resolving.
        aliases = {a: t for a, t in layer.aliases.items() if t not in previous}
        aliases.update({a: t for a, t in staging.aliases.items() if layer.owner.get(t) == base_name})
        layer.aliases = aliases
        for name in previous - staging.registry.keys():
            layer.registry.pop(name, None)
            del layer.owner[name]
        live = layer.trees.get(base_name)
        if live is not None:
            layer.hosted.discard(live[1])
        layer.hosted.update(staging.hosted)
        layer.trees[base_name] = (fingerprint, module_name)
        P._REGISTRY_GENERATION += 1
