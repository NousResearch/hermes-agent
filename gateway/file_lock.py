"""Cross-platform file locking utility.

Provides a proper lock using ``fcntl.flock`` (POSIX) or ``msvcrt.locking``
(Windows) with retry/timeout support.  Replaces ad-hoc file-based locks
that only use ``O_CREAT | O_EXCL`` (which do NOT auto-release on process
death and lack mutual-exclusion during read-check-acquire cycles).

Usage::

    from gateway.file_lock import FileLock, LockTimeout

    lock = FileLock("/path/to/lockfile")
    with lock.acquire(timeout=5):
        # critical section — mutual exclusion guaranteed
        ...
"""

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


class LockTimeout(Exception):
    """Raised when a lock cannot be acquired within the configured timeout."""

    def __init__(self, lock_path: Path, timeout: float) -> None:
        self.lock_path = lock_path
        self.timeout = timeout
        super().__init__(
            f"Could not acquire lock on {lock_path} within {timeout}s"
        )


_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import msvcrt
else:
    import fcntl

# Windows byte-range locks are mandatory for other readers.  Lock a byte
# well past any JSON payload so concurrent readers can still read the file
# while another process holds the mutual-exclusion lock.
_WINDOWS_LOCK_OFFSET = 1024 * 1024


class FileLock:
    """A cross-platform file lock using fcntl (POSIX) or msvcrt (Windows).

    The lock is automatically released when the file descriptor is closed
    (including on process death — OS-level cleanup).

    Parameters
    ----------
    path:
        Path to the lock file.  The parent directory will be created if
        it does not exist.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path) if not isinstance(path, Path) else path
        self._fd: Optional[int] = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self, *, timeout: float = 0, retry_interval: float = 0.05) -> "FileLock":
        """Acquire the exclusive lock.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  ``0`` means non-blocking (raise
            immediately if the lock is held).
        retry_interval:
            Seconds between retry attempts when ``timeout > 0``.

        Returns
        -------
        FileLock
            ``self`` for use as a context manager.

        Raises
        ------
        LockTimeout
            If the lock cannot be acquired within ``timeout`` seconds.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        deadline: Optional[float] = None
        if timeout > 0:
            deadline = time.monotonic() + timeout

        while True:
            fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o644)
            try:
                if _IS_WINDOWS:
                    # On Windows we need to ensure the file has content
                    # before locking a byte range past the current EOF.
                    file_size = os.fstat(fd).st_size
                    if file_size < _WINDOWS_LOCK_OFFSET + 1:
                        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                        os.write(fd, b"\n")
                        os.lseek(fd, 0, os.SEEK_SET)
                    os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    except (BlockingIOError, OSError):
                        os.close(fd)
                        self._fd = None
                        if deadline is None:
                            raise LockTimeout(self._path, 0)
                        if time.monotonic() >= deadline:
                            raise LockTimeout(self._path, timeout)
                        time.sleep(retry_interval)
                        continue
                else:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except (BlockingIOError, OSError):
                        os.close(fd)
                        self._fd = None
                        if deadline is None:
                            raise LockTimeout(self._path, 0)
                        if time.monotonic() >= deadline:
                            raise LockTimeout(self._path, timeout)
                        time.sleep(retry_interval)
                        continue

                self._fd = fd
                return self

            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                self._fd = None
                raise

    def release(self) -> None:
        """Release the lock and close the file descriptor."""
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            if _IS_WINDOWS:
                os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    @contextmanager
    def __call__(self, *, timeout: float = 0, retry_interval: float = 0.05) -> Iterator["FileLock"]:
        """Context manager for acquire/release.

        Usage::

            lock = FileLock("/path/to/lock")
            with lock(timeout=5):
                # critical section
                ...
        """
        self.acquire(timeout=timeout, retry_interval=retry_interval)
        try:
            yield self
        finally:
            self.release()

    def __del__(self) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "locked" if self._fd is not None else "unlocked"
        return f"FileLock({self._path!r}, {state})"
