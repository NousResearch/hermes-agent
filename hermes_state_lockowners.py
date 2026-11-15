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


def state_db_write_lock_holders(db_path) -> list[str]:
    """Operator-facing lines naming the processes that hold a write-class lock on ``db_path``.

    Empty when nothing is held (Linux). Non-Linux always returns candidate lines — see below.

    Non-Linux (macOS/Windows) has no /proc/locks equivalent — fcntl byte-range lock
    owners are not enumerable there — but the 2026-09-28 RCA showed the empty fallback
    lying by omission: 38 "no write-class lock held at the deadline" lines while the
    holder existed and was later named by the in-process holder warning. On those
    platforms we name every FOREIGN process with the db (or a WAL sidecar) open as an
    open-file CANDIDATE — honestly labeled, never claimed lock-proof — and say plainly
    that a remaining in-process holder is the likeliest culprit (foreign_state_db_holders
    deliberately skips this process's own handles, so same-process holders can never
    appear as candidates; the "held the lock via X" holder warning covers that half).
    """
    if not sys.platform.startswith("linux"):
        from hermes_state_holders import describe_holder_pid, foreign_state_db_holders

        candidates = foreign_state_db_holders(db_path)
        unproven: list[str] = []
        by_pid: dict[int, list[str]] = {}
        for pid, target in candidates:
            if pid <= 0:
                # pid=-1 rows are foreign_state_db_holders' *unproven* sentinels (scan
                # failure / scan unavailable) — the target is a reason string, not a
                # path. Never render those as "has X open": say unproven, say why.
                entry = f"unproven: {target} (scan failure, not a named holder)"
                if entry not in unproven:
                    unproven.append(entry)
                continue
            name = Path(os.path.realpath(target)).name
            names = by_pid.setdefault(pid, [])
            if name not in names:
                names.append(name)
        candidate_lines = list(unproven)
        for pid, names in by_pid.items():
            who = describe_holder_pid(pid)
            candidate_lines.append(
                f"{who} has {', '.join(names)} open "
                f"(open-file candidate, not lock-proof)"
            )
        lines = candidate_lines[:6]
        if len(candidate_lines) > 6:
            lines.append(f"... and {len(candidate_lines) - 6} more open-file candidates")
        if not candidate_lines:
            lines.append(
                "no foreign process holds the db open; the holder is likely THIS "
                "process (see the 'held the lock' holder warning for the operation)"
            )
        return lines
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
        held.extend(entry for entry in parse_proc_locks(text, inodes) if entry not in held)
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
        detail = "no write-class lock held at the deadline (the holder released just before it)"
    logger.warning(
        "state.db write lock unavailable for %ss (%s): %s",
        f"{patience_s:g}",
        Path(db_path).name,
        detail,
    )
