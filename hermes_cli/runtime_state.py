"""Stdlib-only recovery and lifetime protection for dependency generations."""
from __future__ import annotations

import atexit
import errno
import base64
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from pm.environments import dependency_home_root, install_state_dir, runtime_facts_path

LOG = logging.getLogger(__name__)

# How long a process that only needs to READ the install state waits for a writer. The holder can
# be another profile's backend rebuilding the whole dependency environment (measured: tens of
# seconds on a bundle, minutes when a sync falls back to an sdist build), and a backend that never
# binds its port is worse than one that binds against the previous generation. Losers skip — the
# same rule boot_bootstrap._RecordLock states for home maintenance.
INSTALL_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_POLL_SECONDS = 0.05


def _lock(fd: int, *, wait: bool, timeout: float | None = None) -> bool:
    """Take the byte lock; ``timeout`` bounds the retry loop (None waits forever, 0 tries once)."""
    deadline = None if timeout is None else time.monotonic() + timeout
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
            if not wait or (deadline is not None and time.monotonic() >= deadline):
                return False
            time.sleep(_LOCK_POLL_SECONDS)
    else:
        import fcntl
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                pass
            if not wait or (deadline is not None and time.monotonic() >= deadline):
                return False
            time.sleep(_LOCK_POLL_SECONDS)


@contextmanager
def runtime_lock(project: Path, *, timeout: float | None = INSTALL_LOCK_TIMEOUT_SECONDS):
    """Hold the per-install dependency lock; yields True when held, False when the wait expired.

    Callers decide what a lost race means: readers skip the work the lock guards and carry on
    (``activate_dependencies`` still selects and leases the committed generation), writers that
    cannot be skipped pass ``timeout=None`` — an install the user asked for is theirs to wait on.
    """
    state = install_state_dir(project)
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not _lock(fd, wait=True, timeout=timeout):
            LOG.warning(
                "dependency lock still held after %ss; continuing without it (%s)",
                timeout, state / ".install.lock")
            yield False
            return
        yield True
    finally:
        os.close(fd)


def _bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _digest(path: Path) -> str | None:
    data = _bytes(path)
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
        config = Path(row["config"])
        if config.name != "config.yaml" or not config.resolve().is_relative_to(dependency_home_root().resolve()):
            raise ValueError("config path is outside Hermes state")
        previous = base64.b64decode(row["previous"], validate=True) if row["previous"] is not None else None
        if not row.get("committed") and _digest(runtime_facts_path(project)) == row["facts_before"]:
            current = _digest(config)
            prior = hashlib.sha256(previous).hexdigest() if previous is not None else None
            if current not in (prior, row.get("config_after")):
                raise ValueError("config changed after publication began; preserve it for manual recovery")
            # No selection was published. Roll back before any plugin is loaded.
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


def lease_generation(environment: Path) -> None:
    """Hold a kernel lock until process exit.

    Call under ``runtime_lock`` at boot; when that lock times out it is still safe to lease
    without it, because the collector only removes generations that are NOT selected and are
    older than a day — the generation being leased here is the selected one.
    """
    generation = environment.parent
    if not (generation / ".lease-managed").is_file():
        return  # Generations produced before leases stay conservatively retained.
    leases = generation / ".leases"
    leases.mkdir(exist_ok=True)
    fd = os.open(leases / uuid.uuid4().hex, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        _lock(fd, wait=True)
    except BaseException:
        os.close(fd)
        raise
    atexit.register(os.close, fd)


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
            active = False
            for lease in (generation / ".leases").glob("*"):
                fd = os.open(lease, os.O_RDWR)
                try:
                    if not _lock(fd, wait=False):
                        active = True
                        break
                finally:
                    os.close(fd)
            if not active:
                shutil.rmtree(generation)
                removed.append(generation)
    return removed
