"""Reclaim per-checkout dependency state whose checkout is gone.

Dependency state lives under ``<home>/installs/<install_key>`` where the key hashes the
checkout path (:func:`pm.environments.install_key`). Every ``hermes -w`` worktree, review tree
or scratch clone that boots the agent therefore commits a full environment (venv, test venv,
PM runtime, ~200 MB each), and nothing removed it when the tree was deleted: one developer
host carried 402 such orphans (80 GB) after a fortnight of worktree campaigns.

The state dir records the checkout it belongs to in ``inputs/.project-root`` (written by
:func:`pm.environments.record_activation_inputs`). A missing path alone does not prove the
checkout is gone: a data root shared with a container (``-v ~/.hermes:/opt/data``) or an
unmounted volume records paths this process cannot see. So a dir is an orphan only when its
checkout sat where this process can see that it was deleted -- under the data root that owns
``installs/`` (scratch clones, the runtime checkout) or directly in a ``.worktrees/`` dir that
still exists. Removal still fails closed on anything a live process could hold: the install
lock, or a dependency / PM runtime generation lease.

A checkout elsewhere (``~/.cache/review-1/head``, a scratch clone outside the data root) can never
be proved deleted, so its missing checkout alone would keep the state forever. It is reclaimed
once the state has also gone unused for ``worktrees.install_state_retention_days``: every sync
recreates ``inputs/`` and every launch takes a generation lease (moving its
``.leases`` dir), so a checkout that still exists behind a mount boundary and is used at all keeps
its state. One unused for longer is rebuilt by ``hermes pm repair`` / the next activation.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _provably_deleted(root: Path, data_root: Path) -> bool:
    if root.exists():
        return False
    return root.is_relative_to(data_root) or (root.parent.name == ".worktrees" and root.parent.is_dir())


def _last_used(state: Path) -> float:
    """Newest mtime among what a sync (``inputs/`` is recreated, ``facts.json`` rewritten) or a
    launch (a generation's ``.leases`` dir) moves. Not the state dir itself: a part-way removal
    moves it, and that must not restart the clock for the retry."""
    paths = [state / "inputs", state / "facts.json"]
    for generations in (state / "environments", state / "pm-runtime" / "generations"):
        if generations.is_dir():
            paths += [generation / ".leases" for generation in generations.iterdir()]
    mtimes = [0.0]
    for path in paths:
        with contextlib.suppress(OSError):
            mtimes.append(path.lstat().st_mtime)
    return max(mtimes)


def _retention_seconds(data_root: Path) -> float | None:
    """The store-wide retention, from the config of the data root that owns ``installs/``.
    Every profile of that root shares the store, so the active profile's config must not decide
    for the others: a named profile without the key would apply the default to a state the root
    opted out of reclaiming."""
    from hermes_cli.config import cfg_get, load_config_readonly
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(data_root)
    try:
        days = cfg_get(load_config_readonly(), "worktrees", "install_state_retention_days", default=14)
    finally:
        reset_hermes_home_override(token)
    return days * 86400 if isinstance(days, (int, float)) and days > 0 else None


def _is_orphan(state: Path, data_root: Path, retention: float | None) -> bool:
    record = state / "inputs" / ".project-root"
    if not state.is_dir() or not record.is_file():
        return False
    try:
        root = record.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return False
    if not root or Path(root).exists():
        return False
    return _provably_deleted(Path(root), data_root) or (
        retention is not None and time.time() - _last_used(state) > retention)


def orphan_install_states(installs: Path) -> list[Path]:
    """State dirs whose recorded checkout is provably deleted, or missing and unused past the
    retention period. Dirs without a record, and dirs whose checkout exists, are left alone."""
    if not installs.is_dir():
        return []
    data_root = installs.resolve().parent
    retention = _retention_seconds(installs.parent)
    return [state for state in sorted(installs.iterdir()) if _is_orphan(state, data_root, retention)]


@contextlib.contextmanager
def _install_lock(state: Path):
    """Hold the state's install lock (``runtime_lock``'s) without waiting: yields False while
    another process holds it, or when it cannot even be opened."""
    from pm.filesystem import lock_path

    try:
        fd = lock_path(state / ".install.lock", timeout=0)
    except OSError:
        fd = None
    if fd is None:
        yield False
        return
    try:
        yield True
    finally:
        os.close(fd)


def _leased(state: Path) -> bool:
    """True while a generation lease (dependency env or PM runtime) is held."""
    from hermes_cli.runtime_state import leases_held

    for generations in (state / "environments", state / "pm-runtime" / "generations"):
        if generations.is_dir():
            for generation in generations.iterdir():
                if generation.is_dir() and (generation / ".leases").is_dir() and leases_held(generation):
                    return True
    return False


_KEPT_UNTIL_LAST = frozenset({"inputs", ".install.lock"})


def _remove_contents(state: Path) -> bool:
    """Remove everything but the record and the held lock; True when nothing else is left.
    The record goes last, and only once everything else is gone: the startup prune runs on a
    daemon thread and a removal can fail part-way (an open file on Windows, a read-only dir), and
    a dir that lost its record first would never be recognised again."""
    for child in state.iterdir():
        if child.name in _KEPT_UNTIL_LAST:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                child.unlink()
    return not any(child.name not in _KEPT_UNTIL_LAST for child in state.iterdir())


def collect_orphan_install_states(installs: Path) -> list[Path]:
    """Remove every unheld orphan state dir under *installs*; returns what was removed.

    Each candidate is re-checked under its install lock right before removal: removing a large
    orphan takes long enough for another state to be synced or launched (lease taken and
    released) since the candidates were listed, and that use must keep it."""
    if not installs.is_dir():
        return []
    data_root = installs.resolve().parent
    retention = _retention_seconds(installs.parent)
    removed: list[Path] = []
    for state in [s for s in sorted(installs.iterdir()) if _is_orphan(s, data_root, retention)]:
        with _install_lock(state) as locked:
            if not locked or not _is_orphan(state, data_root, retention) or _leased(state):
                logger.debug("orphan install state %s is held or back in use; skipped", state.name)
                continue
            emptied = _remove_contents(state)
            if emptied:  # still under the lock: only the record is left, so it goes last
                shutil.rmtree(state / "inputs", ignore_errors=True)
                # POSIX: unlink while still holding it. A launcher already waiting on this file
                # then sees it gone once it gets the lock and retries on a fresh one (lock_path);
                # unlinked after release, the path could be a file that launcher just made.
                if os.name != "nt":
                    with contextlib.suppress(OSError):
                        (state / ".install.lock").unlink()
        if not emptied:
            logger.debug("orphan install state %s only partly removed; retried next pass", state.name)
            continue
        # Non-recursive: a launch from a checkout recreated at the same path may have taken the
        # lock the moment it was released and written here; its files make rmdir fail, not vanish.
        with contextlib.suppress(OSError):
            if os.name == "nt":  # an open file cannot be unlinked: a waiting launcher keeps it
                (state / ".install.lock").unlink()
            state.rmdir()
        if not state.exists():
            removed.append(state)
    return removed
