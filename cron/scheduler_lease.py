"""Cross-process ownership lease for one cron scheduler per HERMES_HOME."""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Literal

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_WINDOWS_LOCK_OFFSET = 1024 * 1024
_PROCESS_LEASES: set[str] = set()
_PROCESS_LEASES_LOCK = threading.Lock()


def _canonical_lock_path(hermes_home: Path) -> Path:
    return hermes_home.expanduser().resolve() / "cron" / ".scheduler-owner.lock"


def _process_start_time() -> int | None:
    try:
        import psutil

        return int(round(psutil.Process(os.getpid()).create_time() * 100))
    except Exception:
        return None


def _try_lock(handle: IO[str]) -> bool:
    try:
        if sys.platform == "win32":
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(handle: IO[str]) -> None:
    try:
        if sys.platform == "win32":
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class SchedulerOwnershipLease:
    """A kernel-authoritative, non-blocking scheduler ownership lease.

    The lock file is deliberately never unlinked. Removing a locked pathname can
    create a second inode and split ownership between two live processes.
    """

    def __init__(self, path: Path, handle: IO[str]) -> None:
        self.path = path
        self._key = str(path)
        self._handle: IO[str] | None = handle
        self._release_lock = threading.Lock()

    @classmethod
    def try_acquire(
        cls,
        *,
        hermes_home: Path,
        owner: Literal["gateway", "desktop"],
        provider: str,
    ) -> "SchedulerOwnershipLease | None":
        path = _canonical_lock_path(hermes_home)
        key = str(path)
        with _PROCESS_LEASES_LOCK:
            if key in _PROCESS_LEASES:
                return None
            # Reserve before the kernel operation so same-process contenders do
            # not rely on platform-specific flock semantics.
            _PROCESS_LEASES.add(key)

        handle: IO[str] | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "a+", encoding="utf-8")
            if not _try_lock(handle):
                handle.close()
                handle = None
                return None

            metadata = {
                "version": 1,
                "pid": os.getpid(),
                "process_start_time": _process_start_time(),
                "owner": owner,
                "provider": provider,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.seek(0)
            handle.truncate()
            json.dump(metadata, handle, sort_keys=True)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
            return cls(path, handle)
        except Exception:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                handle = None
            raise
        finally:
            if handle is None:
                with _PROCESS_LEASES_LOCK:
                    _PROCESS_LEASES.discard(key)

    def release(self) -> None:
        with self._release_lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _unlock(handle)
            try:
                handle.close()
            except OSError:
                pass
            finally:
                with _PROCESS_LEASES_LOCK:
                    _PROCESS_LEASES.discard(self._key)

    def __enter__(self) -> "SchedulerOwnershipLease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
