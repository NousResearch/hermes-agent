"""Read mounted dotenv streams once per process; ordinary files remain live.

Stdlib only: config-backend selection uses this before application bootstrap.
The adjacent advisory lock contains no data and is never unlinked: all Hermes
processes must lock the same inode while the secret producer serves a reader.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import select
import stat
import threading
import time

_FIFO_TIMEOUT = 10.0
_FIFO_MAX_BYTES = 1024 * 1024
_snapshots: dict[str, bytes] = {}
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()
_pid = os.getpid()


class MountedEnvError(RuntimeError):
    """A mounted environment could not be read completely; never include its contents."""


def is_fifo(path: Path) -> bool:
    try:
        return stat.S_ISFIFO(path.stat().st_mode)
    except FileNotFoundError:
        return False


def require_regular_env_target(path: Path) -> None:
    """Refuse before a writer reads, creates a temporary file or replaces a mount."""
    if is_fifo(path):
        raise ValueError(f"Cannot write mounted environment {path}; edit it in your secret manager.")


def invalidate_fifo_cache(path: Path | None = None) -> None:
    """Explicit refresh only: FIFO timestamps can change on each access/remount."""
    with _guard:
        if path is None:
            _snapshots.clear()
        else:
            _snapshots.pop(str(path.resolve()), None)


def _remaining(deadline: float, path: Path) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise MountedEnvError(
            f"Timed out reading mounted environment {path}; unlock/authorize your secret manager "
            "and close other readers, then retry.")
    return remaining


@contextmanager
def _process_read_lock(path: Path, deadline: float):
    import fcntl  # POSIX only; ordinary Windows files never enter this path.

    flags = os.O_CREAT | os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path.with_name(path.name + ".read.lock"), flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:  # windows-footgun: ok — fcntl path is POSIX-only
            raise MountedEnvError(f"Unsafe mounted-environment read lock for {path}")
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                threading.Event().wait(min(0.02, _remaining(deadline, path)))
        yield
    finally:
        os.close(fd)


def _read_fifo(path: Path, deadline: float) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        if not stat.S_ISFIFO(os.fstat(fd).st_mode):
            raise MountedEnvError(f"Mounted environment changed type while opening {path}; retry.")
        chunks: list[bytes] = []
        size = 0
        while True:
            ready, _, _ = select.select([fd], [], [], _remaining(deadline, path))
            if not ready:
                continue
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                if chunks:
                    return b"".join(chunks)
                # No writer yet (or an empty failed authorization). Never install
                # an empty mapping in place of previously available credentials.
                threading.Event().wait(min(0.02, _remaining(deadline, path)))
                continue
            size += len(chunk)
            if size > _FIFO_MAX_BYTES:
                raise MountedEnvError(f"Mounted environment exceeds {_FIFO_MAX_BYTES} bytes: {path}")
            chunks.append(chunk)
    finally:
        os.close(fd)


def read_env_bytes(path: Path) -> bytes:
    """Read a file normally, or share one complete FIFO snapshot in this process.

    A cold FIFO read on an asyncio loop is refused instead of freezing all peers.
    Async consumers must warm their profile's snapshot with ``asyncio.to_thread``.
    """
    global _pid, _guard, _locks, _snapshots
    if _pid != os.getpid():
        # Never inherit a parent's held thread lock or profile snapshots after fork.
        _pid, _guard, _locks, _snapshots = os.getpid(), threading.Lock(), {}, {}
    path = Path(path)
    if not is_fifo(path):
        return path.read_bytes()
    path = path.resolve()
    key = str(path)
    with _guard:
        snapshot = _snapshots.get(key)
        lock = _locks.setdefault(key, threading.Lock())
    if snapshot is not None:
        return snapshot
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise MountedEnvError(f"Mounted environment {path} needs loading in a worker thread before use.")
    deadline = time.monotonic() + _FIFO_TIMEOUT
    if not lock.acquire(timeout=_remaining(deadline, path)):
        raise MountedEnvError(f"Timed out waiting for mounted environment {path}")
    try:
        with _guard:
            snapshot = _snapshots.get(key)
        if snapshot is None:
            with _process_read_lock(path, deadline):
                snapshot = _read_fifo(path, deadline)
            with _guard:
                _snapshots[key] = snapshot
        return snapshot
    finally:
        lock.release()
