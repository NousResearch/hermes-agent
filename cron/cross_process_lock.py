# Windows cross-process lock helper shared by the cron locks.
#
# msvcrt.locking(LK_LOCK) retries internally for only ~10 seconds (10 tries
# x 1s) before raising OSError(EDEADLK). The POSIX branches poll with a
# 30-second budget (#60703 discipline: bounded wait, then fail closed but
# ALIVE). This helper gives the msvcrt backend the same bounded-polling
# semantics so both platforms wait for the same window before giving up.

import time

try:
    import fcntl  # POSIX
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt  # Windows
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


def lock_exclusive_bounded(fd, timeout_seconds: float, *, poll_interval: float = 0.1) -> bool:
    """Acquire an exclusive lock on ``fd`` within ``timeout_seconds``.

    POSIX: fcntl.flock(LOCK_EX | LOCK_NB) polled until the deadline.
    Windows: msvcrt.locking(LK_NBLCK) polled until the deadline
    (LK_LOCK alone retries only ~10s internally, which is shorter than
    the callers' budget and makes Windows fail closed early).

    Returns True when the lock is held by the caller.
    """
    if fcntl is not None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (OSError, IOError):
                if time.monotonic() >= deadline:
                    return False
                time.sleep(poll_interval)
    if msvcrt is not None:
        fileno = fd.fileno() if hasattr(fd, "fileno") else fd
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                msvcrt.locking(fileno, msvcrt.LK_NBLCK, 1)
                return True
            except (OSError, IOError):
                if time.monotonic() >= deadline:
                    return False
                time.sleep(poll_interval)
    return False  # no backend: caller decides (fail closed)


def unlock_quietly(fd) -> None:
    """Best-effort release matching :func:`lock_exclusive_bounded`."""
    try:
        fileno = fd.fileno() if hasattr(fd, "fileno") else fd
        if fcntl is not None:
            fcntl.flock(fileno, fcntl.LOCK_UN)
        elif msvcrt is not None:
            msvcrt.locking(fileno, msvcrt.LK_UNLCK, 1)
    except (OSError, IOError):
        pass
