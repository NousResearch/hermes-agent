"""Hold a state.db writer's WAL-mode file locks in a form a stray ``close()`` cannot cancel.

SQLite protects a live WAL generation with two POSIX advisory locks: a SHARED lock on the main
file's lock range and a shared lock on the DMS byte of ``state.db-shm``. A sibling process may
checkpoint and unlink ``-wal``/``-shm`` at its close only after taking both EXCLUSIVE. POSIX locks
are per process, so any ``open()``/``close()`` of those two files inside the holder — a raw probe,
a plugin, a tool reading ``~/.hermes`` — cancels both (sqlite.org/howtocorrupt.html §2.2) and the
next foreign close strands the holder on a deleted generation (``DeletedWalGenerationError``).

This module adds the same two ranges as *open file description* locks (``F_OFD_SETLK``) on the
descriptors SQLite itself holds. OFD locks belong to the description, not the process: a stray
``close()`` elsewhere cannot cancel them, they die with the connection's own descriptor (nothing
extra to track or retire), and they conflict with a foreign EXCLUSIVE exactly like SQLite's own,
so the sibling's close-time unlink is refused while a guarded handle is open. The guard is
lifted before the handle's own close so a true last close still ends the generation normally.
No-op on Windows and on runtimes without OFD locks.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger("hermes_state")

# SQLite's unix VFS lock geometry (os_unix.c): the SHARED range on the main file and the
# deadman-switch byte of the -shm file.
_PENDING_BYTE = 0x40000000
_SHARED_FIRST = _PENDING_BYTE + 2
_SHARED_SIZE = 510
_SHM_DMS_BYTE = 128

try:
    import fcntl
    # CPython exports F_OFD_SETLK only from 3.12. The kernel ABI values are stable: 37 on every
    # Linux arch (asm-generic/fcntl.h), 90 on XNU (bsd/sys/fcntl.h, documented in fcntl(2)).
    _F_OFD_SETLK: Optional[int] = getattr(
        fcntl, "F_OFD_SETLK", {"linux": 37, "darwin": 90}.get(sys.platform.rstrip("0123456789")))
    _F_RDLCK, _F_UNLCK, _SEEK_SET = fcntl.F_RDLCK, fcntl.F_UNLCK, os.SEEK_SET
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    _F_OFD_SETLK = None
    _F_RDLCK = _F_UNLCK = _SEEK_SET = 0

# struct flock differs per libc: glibc/musl put type+whence first, Darwin/BSD last.
_FLOCK_FORMAT = "@qqihh" if sys.platform == "darwin" or "bsd" in sys.platform else "@hhqqi"

Identity = Tuple[int, int]
Held = Dict[int, Identity]  # fd -> (st_dev, st_ino) it referenced when locked

# Several handles in one process share the same inodes (and see each other's descriptors), so a
# lock on a given (fd, inode) is reference-counted: only the last holder unlocks it.
_LOCK = threading.Lock()
_REFS: Dict[Tuple[int, Identity], int] = {}


def supported() -> bool:
    return _F_OFD_SETLK is not None


def _flock(lock_type: int, start: int, length: int) -> bytes:
    if _FLOCK_FORMAT == "@qqihh":
        return struct.pack(_FLOCK_FORMAT, start, length, 0, lock_type, _SEEK_SET)
    return struct.pack(_FLOCK_FORMAT, lock_type, _SEEK_SET, start, length, 0)


def _ofd_lock(fd: int, lock_type: int, start: int, length: int) -> bool:
    """Apply a non-blocking OFD lock; False when the range is held EXCLUSIVE elsewhere."""
    assert fcntl is not None and _F_OFD_SETLK is not None
    try:
        fcntl.fcntl(fd, _F_OFD_SETLK, _flock(lock_type, start, length))
    except BlockingIOError:
        return False
    return True


def _identity(path: str) -> Optional[Identity]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _own_fds_for(identities: Dict[Identity, Tuple[int, int]]):
    """Yield ``(fd, identity, (start, length))`` for every descriptor of this process on one of
    *identities* (SQLite's own connection descriptors; the cached header-probe fd too, harmless)."""
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for name in names:
            if not name.isdigit():
                continue
            fd = int(name)
            try:
                st = os.fstat(fd)
            except OSError:
                continue
            ident = (st.st_dev, st.st_ino)
            rng = identities.get(ident)
            if rng is not None:
                yield fd, ident, rng
        return


def hold(db_path, held: Optional[Held] = None) -> Held:
    """Lock the guard ranges on every descriptor this process has open on ``state.db`` and its
    ``-shm``; returns the record :func:`release` needs (pass it back to extend an existing one:
    a ``-shm`` minted after open, a reopened connection). Safe to repeat."""
    held = {} if held is None else held
    if not supported():
        return held
    base = os.fspath(db_path)
    wanted: Dict[Identity, Tuple[int, int]] = {}
    for path, rng in ((base, (_SHARED_FIRST, _SHARED_SIZE)), (base + "-shm", (_SHM_DMS_BYTE, 1))):
        ident = _identity(path)
        if ident is not None:
            wanted[ident] = rng
    try:
        with _LOCK:
            for fd, ident, (start, length) in _own_fds_for(wanted):
                if held.get(fd) == ident:
                    continue
                if _REFS.get((fd, ident)) or _ofd_lock(fd, _F_RDLCK, start, length):
                    held[fd] = ident
                    _REFS[(fd, ident)] = _REFS.get((fd, ident), 0) + 1
    except OSError:
        logger.debug("WAL lock guard unavailable for %s", base, exc_info=True)
    return held


def release(held: Held) -> None:
    """Unlock what :func:`hold` locked, on descriptors that still reference the same inode (a
    number recycled onto another file is left alone). Call BEFORE the handle's own close so
    SQLite's close-time reset sees only real holders: a sibling process's intact locks still
    refuse the unlink, and a true last close ends the generation, so a later ``state.db``
    replace never pairs with a stale WAL."""
    if not supported():
        return
    with _LOCK:
        for fd, ident in list(held.items()):
            remaining = _REFS.get((fd, ident), 1) - 1
            if remaining > 0:
                _REFS[(fd, ident)] = remaining
                continue
            _REFS.pop((fd, ident), None)
            try:
                st = os.fstat(fd)
                if (st.st_dev, st.st_ino) == ident:
                    _ofd_lock(fd, _F_UNLCK, _SHARED_FIRST, _SHARED_SIZE)
                    _ofd_lock(fd, _F_UNLCK, _SHM_DMS_BYTE, 1)
            except OSError:
                pass
        held.clear()
