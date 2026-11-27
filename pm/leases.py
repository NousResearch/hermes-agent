"""Generation leases: take them, test them, prune abandoned ones, collect empty lease dirs.

Its own module, not ``hermes_cli.runtime_state``: a process started on v0.23.3 loaded that at boot
(``hermes_bootstrap`` activation imports it) and keeps its version, whose ``leases_held`` only reads
``<generation>/.leases``. Collectors it imports lazily after ``hermes update`` moved the checkout
(#158723) must still see leases current processes take outside the generation, so they import these
from here; it uses only names v0.23.3's ``pm.filesystem`` has.
"""
from __future__ import annotations

import atexit
from collections.abc import Callable
from contextlib import suppress
import os
from pathlib import Path
import uuid

from pm.filesystem import lock_fd as _lock


def lease_dirs(generation: Path) -> list[Path]:
    """Where *generation*'s leases are taken now, then where releases before that took them.

    An install's generations lease from ``installs/<key>/leases/<kind>/<generation>``, outside the
    dependency tree, so that tree can be mounted read-only. Processes started on an earlier
    release still lease ``<generation>/.leases`` until they exit.
    """
    old = generation / ".leases"
    parent = generation.parent
    if parent.name == "environments":
        return [parent.parent / "leases" / "environments" / generation.name, old]
    if parent.name == "generations" and parent.parent.name == "pm-runtime":
        return [parent.parent.parent / "leases" / "pm-runtime" / generation.name, old]
    return [old]


def lease_directory(generation: Path) -> Callable[[], None]:
    """Pin a lease-managed generation directory for this process's lifetime."""
    if not (generation / ".lease-managed").is_file():
        return lambda: None  # Generations produced before leases stay conservatively retained.
    leases = lease_dirs(generation)[0]
    leases.mkdir(parents=True, exist_ok=True)
    _prune_unlocked_leases(leases)
    while True:
        lease = leases / uuid.uuid4().hex
        fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            _lock(fd, wait=True)
        except BaseException:
            os.close(fd)
            raise
        # Create and flock are two syscalls; a peer's prune (pm workers and launch run outside
        # ``runtime_lock``) can lock-and-unlink the file in between, and we would then hold an
        # unlinked inode invisible to every collector. The pruner only unlinks while holding the
        # lock, so once we hold it the path is either still ours or already gone for good.
        if lease.exists():
            break
        os.close(fd)

    def release() -> None:
        atexit.unregister(release)
        os.close(fd)
        # Best effort: the collector ignores unlocked lease files, but one per
        # invocation would otherwise accumulate for the life of the generation.
        with suppress(OSError):
            lease.unlink()

    atexit.register(release)
    return release


def leases_held(generation: Path) -> bool:
    """True while any process still holds a lease taken by ``lease_directory``."""
    return any([_prune_unlocked_leases(leases) for leases in lease_dirs(generation)])


def collect_lease_dirs(leases: Path, generations: Path) -> None:
    """Remove the lease directories under *leases* whose generation is gone and nothing holds."""
    if not leases.is_dir():
        return
    for directory in leases.iterdir():
        if not (generations / directory.name).exists() and not _prune_unlocked_leases(directory):
            with suppress(OSError):
                directory.rmdir()  # a lease taken since the prune keeps it


def _open_lease_probe(lease: Path) -> int:
    """Let a Windows pruner unlink while still holding its exclusion lock.

    Creator handles deliberately keep their ordinary sharing: an open creator
    between O_EXCL and locking must still prevent a peer from deleting its lease.
    """
    if os.name != "nt":
        return os.open(lease, os.O_RDWR)

    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | WRITE | DELETE, OPEN_EXISTING.
    handle = kernel32.CreateFileW(str(lease), 0xC0000000, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # The CRT owns the handle after conversion; os.close releases it and its lock.
        return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def _prune_unlocked_leases(leases: Path) -> bool:
    """Remove abandoned lease files and report whether any live lock remains.

    Kernel locks disappear even when ``execv``, ``os._exit`` or a crash bypasses
    ``atexit``. Cleaning those unlocked files whenever a reader arrives bounds leaks in
    the selected generation too, which generation GC intentionally never visits.

    Fail closed: a lease this user cannot open or remove is held, never a boot failure.
    """
    held = False
    for lease in leases.glob("*"):
        try:
            fd = _open_lease_probe(lease)
        except FileNotFoundError:
            continue
        except OSError:
            held = True
            continue
        try:
            if not _lock(fd, wait=False):
                held = True
            else:
                try:
                    lease.unlink()
                except FileNotFoundError:
                    pass  # A peer already removed it.
                except OSError:
                    held = True
        finally:
            os.close(fd)
    return held
