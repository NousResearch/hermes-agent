"""Shared lifecycle contract for gateway-triggered updates."""

import errno
import os
from pathlib import Path
from typing import Optional


UPDATE_TIMEOUT_SECONDS = 3600.0
UPDATE_ORPHAN_GRACE_SECONDS = 30.0


class UpdateHelperLockBusy(RuntimeError):
    """The live detached update helper owns its lifecycle lock."""


class UpdateHelperLock:
    """Cross-platform OS lock held for the detached helper's whole lifetime."""

    def __init__(self, path: Path, *, blocking: bool = True):
        self.path = path
        self.blocking = blocking
        self._fh = None

    def acquire(self) -> "UpdateHelperLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0, os.SEEK_END)
                if self._fh.tell() == 0:
                    self._fh.write(b"\0")
                    self._fh.flush()
                self._fh.seek(0)
                mode = msvcrt.LK_LOCK if self.blocking else msvcrt.LK_NBLCK
                msvcrt.locking(self._fh.fileno(), mode, 1)
            else:
                import fcntl

                mode = fcntl.LOCK_EX
                if not self.blocking:
                    mode |= fcntl.LOCK_NB
                fcntl.flock(self._fh.fileno(), mode)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            if not self.blocking and exc.errno in (errno.EACCES, errno.EAGAIN):
                raise UpdateHelperLockBusy from exc
            raise RuntimeError("update helper file lock unavailable") from exc
        except Exception as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError("update helper file lock unavailable") from exc
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "UpdateHelperLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def acquire_update_helper_probe(path: Path) -> Optional[UpdateHelperLock]:
    """Hold and return the unlocked helper lock, or None when a helper owns it.

    Lock inspection errors are raised so callers can fail closed. Keeping the
    returned handle held closes the probe-to-cleanup race with a late helper.
    """
    lock = UpdateHelperLock(path, blocking=False)
    try:
        return lock.acquire()
    except UpdateHelperLockBusy:
        return None
