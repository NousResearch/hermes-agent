"""Who holds the SQLite write lock on state.db right now (Linux ``/proc/locks``).

The "database is locked for over Ns" failure names the victim but not the holder, and the
open-descriptor scan (``hermes_state_holders``) cannot tell a reader from the one writer: every
Hermes process has the DB open. SQLite's unix VFS takes ``fcntl`` byte-range locks whose offsets
encode the lock kind, and the kernel exports them with the owning pid, so the holder can be named
at the moment the deadline passes.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# sqlite/src/os_unix.c: UNIX_SHM_BASE = (22 + SQLITE_SHM_NLOCK) * 4 = 120; WAL lock i sits at BASE + i.
_SHM_LOCK_NAMES = {
    120: "WAL write",
    121: "WAL checkpoint",
    122: "WAL recover",
    128: "shm dead-man switch",
}
# Rollback-journal / exclusive locks on the main file live in the pending-byte page (1 GiB).
_PENDING_BYTE = 0x40000000
_MAIN_LOCK_NAMES = {
    _PENDING_BYTE: "PENDING",
    _PENDING_BYTE + 1: "RESERVED",
}


def _describe_range(sidecar: str, start: int, end: int) -> str:
    if sidecar == "-shm":
        if start in _SHM_LOCK_NAMES and end == start:
            return _SHM_LOCK_NAMES[start]
        if 123 <= start <= 127 and end == start:
            return f"WAL read slot {start - 123}"
        return f"shm bytes {start}-{end}"
    if start == end and start in _MAIN_LOCK_NAMES:
        return _MAIN_LOCK_NAMES[start]
    if start == _PENDING_BYTE + 2:
        return "SHARED range"
    return f"db bytes {start}-{end}"


def parse_proc_locks(text: str, inodes: dict[tuple[int, int], str]) -> list[tuple[int, str, str]]:
    """``(pid, lock kind, sidecar)`` for every WRITE lock on one of ``inodes``.

    ``inodes`` maps ``(st_dev, st_ino)`` to ``""`` (main file), ``"-wal"`` or ``"-shm"``. Read locks
    are dropped: they never block a writer in WAL mode. An OFD lock is reported with pid ``-1``
    (the kernel does not export its owner).
    """
    found: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        fields = line.split()
        # "N: POSIX ADVISORY WRITE <pid> MAJ:MIN:INO <start> <end|EOF>"; a blocked waiter is "N: -> POSIX ...".
        if len(fields) < 8 or fields[1] == "->":
            continue
        _, _kind, _cls, access, pid_s, ident, start_s, end_s = fields[:8]
        if access != "WRITE":
            continue
        try:
            major_s, minor_s, ino_s = ident.split(":")
            dev = os.makedev(int(major_s, 16), int(minor_s, 16))
            key = (dev, int(ino_s))
            pid = int(pid_s)
            start = int(start_s)
            end = start if end_s == "EOF" else int(end_s)
        except ValueError:
            continue
        sidecar = inodes.get(key)
        if sidecar is None:
            continue
        found.append((pid, _describe_range(sidecar, start, end), sidecar))
    return found



def _translated_device_write_locks(
    text: str, inodes: dict[tuple[int, int], str],
) -> list[tuple[int, str, str]]:
    """Attribute kernel write locks across container device-number translation.

    Linux can expose different MAJ:MIN values for the same mounted SQLite file
    in stat() and /proc/locks. A shared inode number and an open target FD do
    NOT prove the lock belongs to the target: two filesystems can reuse inodes,
    and a process can have both files open.

    Require both (a) the owning PID's FD resolves to the target's exact
    (st_dev, st_ino) and (b) that *same FD's* /proc/PID/fdinfo lists the exact
    kernel lock (owner, device, inode, byte range). If /proc is inaccessible,
    remain inconclusive rather than guessing.
    """
    by_inode: dict[int, list[tuple[int, str]]] = {}
    for (dev, ino), sidecar in inodes.items():
        by_inode.setdefault(ino, []).append((dev, sidecar))

    found: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[1] == "->" or fields[3] != "WRITE":
            continue
        try:
            major_s, minor_s, ino_s = fields[5].split(":")
            dev = os.makedev(int(major_s, 16), int(minor_s, 16))
            ino = int(ino_s)
            pid = int(fields[4])
            start = int(fields[6])
            end = start if fields[7] == "EOF" else int(fields[7])
        except ValueError:
            continue
        if pid <= 0 or (dev, ino) in inodes or ino not in by_inode:
            continue

        try:
            with os.scandir(f"/proc/{pid}/fd") as fds:
                for fd in fds:
                    try:
                        st = fd.stat()
                        if (st.st_dev, st.st_ino) not in inodes:
                            continue
                        if st.st_ino != ino:
                            continue
                        with open(f"/proc/{pid}/fdinfo/{fd.name}", encoding="ascii", errors="replace") as info:
                            # The fdinfo prefix is "lock:", followed by the same
                            # /proc/locks record. Ignore its per-snapshot sequence number.
                            matching_lock = any(
                                row.startswith("lock:") and row.split()[4:9] == fields[3:8]
                                for row in info
                            )
                        if not matching_lock:
                            continue
                    except OSError:
                        continue  # FD closed or proc access restricted during observation
                    sidecar = inodes[(st.st_dev, st.st_ino)]
                    entry = (pid, _describe_range(sidecar, start, end), sidecar)
                    if entry not in found:
                        found.append(entry)
        except OSError:
            continue
    return found


def state_db_write_lock_holders(db_path) -> list[str]:
    """Operator-facing lines naming the processes that hold a write-class lock on ``db_path``.

    Empty when nothing is held or the platform has no ``/proc/locks``.
    """
    if not sys.platform.startswith("linux"):
        return []
    base = os.path.realpath(os.fspath(db_path))
    inodes: dict[tuple[int, int], str] = {}
    for sidecar in ("", "-wal", "-shm"):
        try:
            st = os.stat(base + sidecar)
        except OSError:
            continue
        inodes[(st.st_dev, st.st_ino)] = sidecar
    # /proc/locks is host-wide and served over several read()s: lock churn in other processes
    # shifts it mid-read and can skip the holder. A skip rarely repeats, so union three passes.
    held: list[tuple[int, str, str]] = []
    for _ in range(3):
        try:
            with open("/proc/locks", encoding="ascii", errors="replace") as handle:
                text = handle.read()
        except OSError:
            return []
        entries = parse_proc_locks(text, inodes)
        entries.extend(_translated_device_write_locks(text, inodes))
        held.extend(entry for entry in entries if entry not in held)
    from hermes_state_holders import describe_holder_pid

    lines = []
    for pid, kind, sidecar in held:
        who = describe_holder_pid(pid) if pid > 0 else "OFD lock, owner pid not exported by the kernel"
        lines.append(f"{who} holds {kind} lock on {Path(base + sidecar).name}")
    return lines


def log_write_lock_holders(db_path, patience_s: float) -> None:
    """One WARNING naming the write-lock holders at the moment a writer gave up waiting."""
    holders = state_db_write_lock_holders(db_path)
    if holders:
        detail = " | ".join(holders)
    else:
        detail = "no identifiable write-class lock at the deadline (holder may have released or be unobservable)"
    logger.warning(
        "state.db write lock unavailable for %.0fs (%s): %s", patience_s, Path(db_path).name, detail
    )
