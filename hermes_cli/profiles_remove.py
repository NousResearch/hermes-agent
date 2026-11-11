"""Profile directory removal: wait out the live owner, then rmtree with a short retry."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import shutil
import stat
import time

# A multiplexer answers a delete's rescan ``pending`` after 5 s and keeps tearing the profile down;
# its gateway.lock reservation is released last, once the profile's writers are gone.
OWNER_RELEASE_WAIT_SECONDS = 60.0


def wait_profile_unowned(profile_dir: Path, *, timeout: float | None = None) -> bool:
    """True once no gateway runtime holds ``profile_dir``'s ``gateway.lock`` (taken and released
    through ``exclusive_maintenance``); False when it is still held after ``timeout`` seconds."""
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance
    deadline = time.monotonic() + (OWNER_RELEASE_WAIT_SECONDS if timeout is None else timeout)
    while True:
        try:
            with exclusive_maintenance([profile_dir]):
                return True
        except OwnershipConflict:
            pass
        except OSError:
            # A read-only copy (0555 home, 0444 lock) cannot host a reservation: probe it instead.
            if _lock_unheld_read_only(profile_dir / 'gateway.lock'):
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _lock_unheld_read_only(path: Path) -> bool:
    """A missing lock, or one our own flock through a read-only descriptor wins, has no holder."""
    if os.name == 'nt':
        return False  # a sharing violation there is a live holder
    import fcntl
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK)
    except FileNotFoundError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _rmtree_make_writable(func, path, exc):
    """onexc/onerror handler: add +w on PermissionError so rmtree can proceed. Covers NixOS-
    style read-only copies where the path itself (0444) or its parent (0555) isn't writable."""
    # onexc(func, path, exc_instance) on 3.12+; onerror(func, path, exc_info_tuple) on 3.11.
    if isinstance(exc, tuple):
        exc = exc[1]
    if not isinstance(exc, PermissionError):
        raise
    for target in (path, os.path.dirname(path)):  # parent needed for unlink/rmdir
        if target:
            with contextlib.suppress(OSError):
                os.chmod(target, os.stat(target).st_mode | stat.S_IWUSR)
    func(path)


def _rmtree_with_retry(profile_dir: Path, onexc_handler) -> None:
    """``shutil.rmtree`` with a short retry loop: a just-terminated process can leave in-flight
    writes (SQLite -wal/-shm checkpoints, sandbox temp files) landing after rmtree walked
    past a directory — ENOTEMPTY on POSIX, transient PermissionError on Windows."""
    attempts = 3
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            try:
                shutil.rmtree(profile_dir, onexc=onexc_handler)
            except TypeError:  # ``onexc`` is 3.12+; 3.11 has ``onerror``
                shutil.rmtree(profile_dir, onerror=onexc_handler)
            return
        except OSError as e:
            last_exc = e
            if not profile_dir.exists():
                return
            if attempt < attempts - 1:
                time.sleep(0.3 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
