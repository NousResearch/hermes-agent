"""The unusable scoped-lock dir report (#134889 W2). :mod:`gateway.status` raises it from
``acquire_scoped_lock``; nothing here reads back into that module."""
from __future__ import annotations

import logging
from pathlib import Path

from hermes_dir_acl import unusable_dir_remedy

log = logging.getLogger("gateway.status")


class ScopedLockDirUnusable(PermissionError):
    """This token cannot create, read or replace locks in the scoped-lock dir (#134889 W2).

    On Windows an elevated install/update creates ``gateway-locks`` owned by Administrators with
    no entry for the user; the same user's non-elevated gateway then gets ``Permission denied`` on
    every token lock. The adapter reports it once and names the repair; it stays retryable (the
    dir is shared by every token-locked adapter), so a repair is picked up without a restart. A
    ``PermissionError`` subclass so existing callers keep their handling.
    """


# Lock dirs already reported unusable by this process: WARNING once per dir, DEBUG after, since the
# reconnect watcher and every adapter sharing the dir re-enter the acquire path.
_warned: set[Path] = set()


def unusable_lock_dir(lock_dir: Path, exc: PermissionError) -> ScopedLockDirUnusable:
    """Log *lock_dir* as unusable (WARNING once, then DEBUG) and return the error to raise."""
    level = logging.DEBUG if lock_dir in _warned else logging.WARNING
    _warned.add(lock_dir)
    log.log(level, "Gateway lock directory %s is not usable by this process (%s); it was likely created "
               "by an elevated (administrator) Hermes. Platforms that need a token lock stay down until it "
               "is repaired. To repair, %s", lock_dir, exc, unusable_dir_remedy(lock_dir))
    return ScopedLockDirUnusable(exc.errno, f"gateway lock directory is not usable: {exc}", str(lock_dir))


def unusable_lock_dir_recovered(lock_dir: Path) -> None:
    """A lock was taken in *lock_dir*: warn again if it ever becomes unusable later."""
    _warned.discard(lock_dir)


def reset_unusable_lock_dir_reports() -> None:
    """Forget every report (a fresh process's view; tests)."""
    _warned.clear()


def unusable_lock_dir_reported(lock_dir: Path) -> bool:
    return lock_dir in _warned
