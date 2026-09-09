"""Stdlib-only recovery and lifetime protection for dependency generations."""
from __future__ import annotations

import atexit
import errno
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from hermes_cli.runtime_paths import dependency_home_root, install_state_dir, runtime_facts_path


def _lock(fd: int, *, wait: bool) -> bool:
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
                if not wait:
                    return False
                time.sleep(0.05)
    else:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            return True
        except BlockingIOError:
            return False


@contextmanager
def runtime_lock(project: Path):
    state = install_state_dir(project)
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        _lock(fd, wait=True)
        yield
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


def recover_publication(project: Path) -> None:
    """Recover while holding runtime_lock, before activation or another write."""
    journal = install_state_dir(project) / "publication.json"
    data = _bytes(journal)
    if data is None:
        return
    try:
        row = json.loads(data)
        if row.get("kind") == "plugin":
            from hermes_cli.plugins_transaction import recover_plugin_publication

            recover_plugin_publication(project, row, journal)
            return
        config = Path(row["config"])
        if config.name != "config.yaml" or not config.resolve().is_relative_to(dependency_home_root().resolve()):
            raise ValueError("config path is outside Hermes state")
        previous = base64.b64decode(row["previous"], validate=True) if row["previous"] is not None else None
        if _digest(runtime_facts_path(project)) == row["facts_before"]:
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


class Publication:
    def __init__(self, project: Path, config: Path, proposed: bytes | None = None):
        self.project = project
        self.journal = install_state_dir(project) / "publication.json"
        previous = _bytes(config)
        row = {"config": str(config.resolve()), "previous": base64.b64encode(previous).decode() if previous is not None else None,
               "facts_before": _digest(runtime_facts_path(project)),
               "config_after": hashlib.sha256(proposed).hexdigest() if proposed is not None else None}
        _atomic_bytes(self.journal, json.dumps(row).encode())

    def __call__(self) -> None:
        recover_publication(self.project)

    def finish(self) -> None:
        self.journal.unlink(missing_ok=True)


def begin_publication(project: Path, config: Path, proposed: bytes | None = None) -> Publication:
    recover_publication(project)
    return Publication(project, config, proposed)


def lease_generation(environment: Path) -> None:
    """Hold a kernel lock until process exit; call under runtime_lock at boot."""
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
    from hermes_cli.runtime_paths import selected_venv
    removed = []
    root = install_state_dir(project)
    if not root.exists():
        return removed
    with runtime_lock(project):
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
