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
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _provably_deleted(root: Path, data_root: Path) -> bool:
    if root.exists():
        return False
    return root.is_relative_to(data_root) or (root.parent.name == ".worktrees" and root.parent.is_dir())


def orphan_install_states(installs: Path) -> list[Path]:
    """State dirs whose recorded checkout is provably deleted. Dirs without a record are left alone."""
    if not installs.is_dir():
        return []
    data_root = installs.resolve().parent
    orphans: list[Path] = []
    for state in sorted(installs.iterdir()):
        record = state / "inputs" / ".project-root"
        if not state.is_dir() or not record.is_file():
            continue
        try:
            root = record.read_text(encoding="utf-8-sig").strip()
        except OSError:
            continue
        if root and _provably_deleted(Path(root), data_root):
            orphans.append(state)
    return orphans


def _held(state: Path) -> bool:
    """True while a process may still be using this state dir: the install lock is taken or a
    generation lease is held. Any lock we cannot even open counts as held."""
    from hermes_cli.runtime_state import leases_held
    from pm.filesystem import lock_fd

    lock = state / ".install.lock"
    if lock.exists():
        try:
            fd = os.open(lock, os.O_RDWR)
        except OSError:
            return True
        try:
            if not lock_fd(fd, wait=False):
                return True
        finally:
            os.close(fd)
    for generations in (state / "environments", state / "pm-runtime" / "generations"):
        if generations.is_dir():
            for generation in generations.iterdir():
                if generation.is_dir() and (generation / ".leases").is_dir() and leases_held(generation):
                    return True
    return False


def collect_orphan_install_states(installs: Path) -> list[Path]:
    """Remove every unheld orphan state dir under *installs*; returns what was removed."""
    removed: list[Path] = []
    for state in orphan_install_states(installs):
        if _held(state):
            logger.debug("orphan install state %s is still held; skipped", state.name)
            continue
        shutil.rmtree(state, ignore_errors=True)
        if not state.exists():
            removed.append(state)
    return removed
