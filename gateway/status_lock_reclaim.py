"""Reclaiming abandoned gateway lock files without stealing a live one.

Both the scoped lock (one bot token across homes) and the runtime lock self-heal by removing a
leftover file. A check-then-remove on a shared path is a race: by the time the file is removed it
may already be a different, live lock. These helpers only remove a file once it is proven stale.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

# A lock file is empty between its creator's O_EXCL create (or truncate) and the record dump, so
# an empty or unparseable file this young may belong to a live writer.
LOCK_WRITE_GRACE_S = 5.0


def lock_file_within_write_grace(lock_path: Path) -> bool:
    """True while ``lock_path`` is young enough to be a live writer mid-dump."""
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return age < LOCK_WRITE_GRACE_S


def reclaim_stale_scoped_lock(lock_path: Path) -> bool:
    """Remove a scoped lock judged stale, unless it was replaced by a live lock meanwhile.

    The caller's staleness verdict is about the record it read; a racer may have reclaimed that
    record and installed its own lock since, and renaming the path would steal it (both starters
    then win the O_EXCL create). So rename to a tombstone only this process uses, re-judge the file
    actually captured, and put a live one back with a non-clobbering ``os.link`` (same inode, so a
    writer mid-dump still lands in it).

    Returns False when the captured file is a live lock (contention); True when the caller may race
    the O_EXCL create (stale file removed, or someone else already moved it)."""
    from gateway import status

    tombstone = lock_path.with_name(f"{lock_path.name}.stale.{os.getpid()}.{time.time_ns()}")
    try:
        os.replace(lock_path, tombstone)
    except OSError:
        return True  # another racer claimed the stale file first; O_EXCL picks one winner
    record = status._read_json_file(tombstone)
    if record is None:
        stale = not lock_file_within_write_grace(tombstone)
    else:
        stale = status._scoped_lock_record_is_stale(record, status._pid_from_record(record))
    if not stale:
        try:
            os.link(tombstone, lock_path)
        except FileExistsError:
            pass  # a newer lock was created while this one was renamed away; it stands
        except OSError:
            with contextlib.suppress(OSError):
                os.replace(tombstone, lock_path)  # filesystem without hard links
    status._unlink_quietly(tombstone)
    return stale


def unopenable_runtime_lock_may_be_held(lock_path: Path) -> bool:
    """True when a runtime lock we cannot open (root-owned, #42685) may still belong to a gateway.

    Held when its recorded owner is alive (start-time PID-reuse guard), or, with no readable
    record, while the file is young enough to be mid-write. A dead owner or an aged unreadable
    file is a launchd leftover and safe to replace."""
    from gateway import status

    record = status._read_json_file(lock_path, bare_pid_ok=True)
    if record is not None:
        return status._live_pid_from_record(record) is not None
    return lock_file_within_write_grace(lock_path)
