"""Stdlib-only recovery and lifetime protection for dependency generations."""
from __future__ import annotations

import base64
from collections.abc import Callable
from contextlib import contextmanager, suppress
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

from pm.environments import dependency_home_root, install_state_dir, runtime_facts_path
from pm.leases import collect_lease_dirs, lease_directory, leases_held
from pm.moved_locks import install_lock_paths, moved_lock
# Private aliases: this module calls them through its globals (tests patch ``_atomic_bytes``
# here) and updaters shipped before PM import them by these names mid-swap
# (tests/compat/old_updater_surface.json). New code imports the pm.filesystem names.
from pm.filesystem import (
    durable_write_bytes as _atomic_bytes,
    file_digest as _digest,
    lock_fd as _lock,
    read_bytes_or_none as _bytes,
)

LOG = logging.getLogger(__name__)


# How long a process that only needs to READ the install state waits for a writer. The holder can
# be another profile's backend rebuilding the whole dependency environment (measured: tens of
# seconds on a bundle, minutes when a sync falls back to an sdist build), and a backend that never
# binds its port is worse than one that binds against the previous generation. Losers skip — the
# same rule boot_bootstrap._RecordLock states for home maintenance.
INSTALL_LOCK_TIMEOUT_SECONDS = 10.0

# A lease file younger than this is never a prune candidate: its creator may still sit between
# ``O_CREAT|O_EXCL`` and ``flock`` (two syscalls; the window is microseconds, the margin generous).
@contextmanager
def runtime_lock(project: Path, *, timeout: float | None = INSTALL_LOCK_TIMEOUT_SECONDS):
    """Hold the per-install dependency lock; yields True when held, False when the wait expired.

    Callers decide what a lost race means: readers skip the work the lock guards and carry on
    (``activate_dependencies`` still selects and leases the committed generation), writers that
    cannot be skipped pass ``timeout=None`` — an install the user asked for is theirs to wait on.
    """
    old, new = install_lock_paths(install_state_dir(project))
    with moved_lock(old, new, timeout=timeout) as held:
        if not held:
            LOG.warning("dependency lock still held after %ss; continuing without it (%s)", timeout, new)
        yield held


def _recover_plugin_publication(project: Path, row: dict, journal: Path) -> None:
    from hermes_cli.fs_utils import rmtree_force

    target, backup, metadata = (Path(row[key]) for key in ("target", "backup", "metadata"))
    home = dependency_home_root().resolve()
    if (not target.resolve().is_relative_to(home) or target.parent.name != "plugins"
            or backup.parent != target.parent or not backup.name.startswith(".previous-")
            or metadata != target.parent / ".install-metadata.json"):
        raise ValueError("plugin publication paths escape their home")
    committed = row.get("committed") or _digest(runtime_facts_path(project)) != row["facts_before"]
    if committed:
        if backup.exists():
            rmtree_force(backup)
    else:
        old = base64.b64decode(row["metadata_before"], validate=True) if row["metadata_before"] is not None else None
        current = _bytes(metadata)
        new = base64.b64decode(row["metadata_after"], validate=True)
        if current not in (old, new):
            raise ValueError("plugin metadata changed after publication; preserve it for manual recovery")
        if backup.exists():
            if target.exists():
                rmtree_force(target)
            os.replace(backup, target)
        elif not row["target_existed"] and target.exists():
            rmtree_force(target)
        if old is None:
            metadata.unlink(missing_ok=True)
        else:
            _atomic_bytes(metadata, old)
    journal.unlink()


def recover_publication(project: Path) -> None:
    """Recover while holding runtime_lock, before activation or another write."""
    journal = install_state_dir(project) / "publication.json"
    data = _bytes(journal)
    if data is None:
        return
    try:
        row = json.loads(data)
        if row.get("kind") == "plugin":
            _recover_plugin_publication(project, row, journal)
            return
        # One row may carry several configs (a plugin eviction edits every home that enables it).
        entries = row["configs"] if "configs" in row else [row]
        configs = []
        for entry in entries:
            config = Path(entry["config"])
            if config.name != "config.yaml" or not config.resolve().is_relative_to(dependency_home_root().resolve()):
                raise ValueError("config path is outside Hermes state")
            previous = base64.b64decode(entry["previous"], validate=True) if entry["previous"] is not None else None
            configs.append((config, previous, entry.get("config_after")))
        if not row.get("committed") and _digest(runtime_facts_path(project)) == row["facts_before"]:
            for config, previous, after in configs:
                prior = hashlib.sha256(previous).hexdigest() if previous is not None else None
                if _digest(config) not in (prior, after):
                    raise ValueError("config changed after publication began; preserve it for manual recovery")
            # No selection was published. Roll back before any plugin is loaded.
            for config, previous, _after in configs:
                if previous is None:
                    config.unlink(missing_ok=True)
                else:
                    _atomic_bytes(config, previous)
        journal.unlink()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise RuntimeError(f"cannot recover dependency publication: {journal}: {exc}") from exc


def finish_publication(project: Path) -> None:
    """Persist a commit even when code/config changed without a new generation."""
    journal = install_state_dir(project) / "publication.json"
    row = json.loads(journal.read_bytes())
    row["committed"] = True
    _atomic_bytes(journal, json.dumps(row).encode())
    recover_publication(project)


def lease_generation(environment: Path) -> Callable[[], None]:
    """Hold a kernel lock until process exit; the returned callable releases it early.

    Call under ``runtime_lock`` at boot. Without the lock (``runtime_lock`` timed out) the
    caller must re-read the selection after leasing: an installer may have moved it in between,
    and an unselected, unleased generation is exactly what the collector removes.
    """
    return lease_directory(environment.parent)


def collect_generations(project: Path, *, min_age_seconds: float = 86400) -> list[Path]:
    """Remove unselected lease-managed generations after their readers exit."""
    from pm.environments import selected_venv
    removed = []
    root = install_state_dir(project)
    if not root.exists():
        return removed
    with runtime_lock(project) as held:
        if not held:
            return removed  # maintenance: another process owns the install, skip rather than queue
        recover_publication(project)
        selected = selected_venv(project).parent.resolve()
        generations = root / "environments"
        if not generations.is_dir():
            return removed
        for generation in generations.iterdir():
            if generation.is_symlink() or not generation.is_dir() or generation.resolve() == selected:
                continue
            marker = generation / ".lease-managed"
            if not marker.is_file() or time.time() - marker.stat().st_mtime < min_age_seconds:
                continue
            if not leases_held(generation):
                shutil.rmtree(generation)
                removed.append(generation)
        collect_lease_dirs(root / "leases" / "environments", generations)
    return removed
