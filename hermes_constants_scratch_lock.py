"""Scratch owner locks: which processes the scratch prune must treat as Hermes' own.

A Hermes process holds an exclusive OS lock on ``<default root>/locks/scratch-owners/<pid>.lock``
for as long as it lives. The operating system drops that lock the instant the holder dies (exit,
SIGKILL, TerminateProcess, OOM), so the prune never has to guess who is alive: a held lock is a
live holder, and clocks, reboots and pid reuse cannot fake one.

POSIX uses ``flock``: the lock belongs to the open file description, so the holder closing some
other descriptor of the same file (a profile clone or backup reading ``locks/`` in-process) does
not drop it, as it would a ``lockf``/``fcntl`` record lock. Hermes opens it non-inheritable, so an
exec'd child never shares it, and a fork child closes its inherited copies at once. Windows uses ``msvcrt.locking`` on a non-inheritable handle. Either
way a held file names a live process Hermes owns: the holder itself, or an unreaped child the
holder spawned and vouches for (an unreaped child's pid cannot be reused).

Who holds one: every long-lived entry point (``process_identity.register_self``, or
:func:`hold_owner_lock` directly) for itself, and the process registry for each background session
it spawned or re-adopted (:func:`vouch_for_child`, ``<child>.vouch-<spawner>-<n>.lock``: a file of
its own, so the child taking ``<child>.lock`` for itself never waits on it). The owner pid is the
part of the name before the first dot. Every other process is owned through its live parent chain
(:func:`owning_holder`).
"""
from __future__ import annotations

import atexit
import contextlib
import errno
import itertools
import logging
import os
import sys
import time
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)

OWNERS_SUBDIR = ("locks", "scratch-owners")
_SUFFIX = ".lock"
# A prune probe holds a lock for microseconds: an owner retries for up to ~2 s before giving up.
_ACQUIRE_TRIES = 200
_ACQUIRE_PAUSE = 0.01
# The lock is held by someone else. Any other error (ENOLCK on an NFS mount without lockd,
# EOPNOTSUPP) means this filesystem cannot lock at all: retrying cannot help.
_BUSY = {errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, getattr(errno, "EDEADLK", -1),
         getattr(errno, "EDEADLOCK", -1)}

# lock file path -> fd: this process's own lock (one per root; tests use many roots) and the locks
# it holds for children (:func:`vouch_for_child`).
_held: dict[str, int] = {}
_vouch_seq = itertools.count(1)


def _ours(path: str) -> bool:
    return path in _held


def _forget_inherited() -> None:
    """In a fork child: close the inherited lock descriptors, neither unlocking nor unlinking. The
    parent's flock lives on through its own descriptor, while a fork child that outlived its parent
    would otherwise keep a dead holder's lock held (a false owner once that pid is reused)."""
    for fd in _held.values():
        with contextlib.suppress(OSError):
            os.close(fd)
    _held.clear()


if not _IS_WINDOWS:
    os.register_at_fork(after_in_child=_forget_inherited)


def owners_dir(home: str | Path | None = None) -> Path:
    """The owners dir of *home*'s machine root (all profiles of an install share it)."""
    from hermes_constants import get_default_hermes_root

    root = get_default_hermes_root() if home is None else get_default_hermes_root(home=home)
    return root.joinpath(*OWNERS_SUBDIR)


def owner_pid(name: str) -> int | None:
    """The pid a lock file name speaks for: ``<pid>.lock`` or ``<pid>.vouch-...lock``."""
    stem = name.split(".", 1)[0]
    # ASCII digits only: str.isdigit() also accepts other scripts' digits ("١٢").
    return int(stem) if name.endswith(_SUFFIX) and stem.isascii() and stem.isdigit() else None


def _try_lock(fd: int) -> bool:
    """Take the lock without waiting: False when someone else holds it. Raises ``OSError`` when
    the filesystem cannot lock at all."""
    try:
        if _IS_WINDOWS:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _BUSY:
            return False
        raise
    return True


def _unlock(fd: int) -> None:
    with contextlib.suppress(OSError):
        if _IS_WINDOWS:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)


def _still_named(fd: int, path: str) -> bool:
    """False when a prune removed *path* between our open and our lock (we locked a dead inode)."""
    try:
        return os.path.samestat(os.fstat(fd), os.stat(path))
    except OSError:
        return False


def _acquire(path: str) -> str | None:
    """Lock *path* from this process; *path*, or None when another holder keeps it. Raises
    ``OSError`` when the file cannot be created or the filesystem cannot lock."""
    if _ours(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(_ACQUIRE_TRIES):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            locked = _try_lock(fd) and _still_named(fd, path)
        except OSError:
            os.close(fd)
            raise
        if locked:
            _held[path] = fd
            return path
        os.close(fd)
        time.sleep(_ACQUIRE_PAUSE)
    return None


def _release(path: str) -> None:
    """Drop a lock this process holds, and its file. A fork child (atexit included) holds none of
    the parent's locks (:func:`_forget_inherited`) and must not touch the parent's files."""
    if not _ours(path):
        return
    fd = _held.pop(path)
    if not _IS_WINDOWS:
        with contextlib.suppress(OSError):
            os.unlink(path)  # while still holding: no probe can see it free first
    os.close(fd)
    if _IS_WINDOWS:
        with contextlib.suppress(OSError):
            os.unlink(path)


def hold_owner_lock(home: str | Path | None = None) -> bool:
    """Take this process's owner lock for *home*'s root and hold it until the process exits.
    Idempotent. False (logged) when it cannot be taken: the prune will not know this process.
    Never raises: entry points call it first thing, and no lock failure may stop one starting."""
    try:
        path = os.path.join(owners_dir(home), f"{os.getpid()}{_SUFFIX}")
        already = _ours(path)
        taken = _acquire(path)
    # health: allow BLE001 -- best-effort by contract: a home that cannot resolve (RuntimeError from Path.home()) must not abort the entry point
    except Exception as exc:
        logger.warning("scratch owner lock not taken (%s): the scratch prune will not know this process", exc)
        return False
    if taken is None:
        logger.warning("scratch owner lock %s is held elsewhere: the scratch prune will not know this process", path)
        return False
    if not already:
        atexit.register(_release, path)
    return True


def vouch_for_child(pid: int, home: str | Path | None = None) -> str | None:
    """Hold a lock naming *pid*, a child this process spawned (or re-adopted) and still tracks,
    until :func:`release_vouch` with the returned handle. It cannot outlive this process. Each call
    gets its own file, so a release only ever drops the vouch it names. Best-effort: None when the
    lock cannot be taken. Never raises: the registry calls it right after a spawn or during
    checkpoint recovery, where an exception would kill the fresh child or abort the recovery."""
    try:
        path = os.path.join(owners_dir(home), f"{pid}.vouch-{os.getpid()}-{next(_vouch_seq)}{_SUFFIX}")
        return _acquire(path)
    # health: allow BLE001 -- best-effort by contract: a failed vouch only leaves the child unprotected from the prune
    except Exception as exc:
        logger.warning("scratch owner vouch for pid %s not taken (%s): the scratch prune will not know it", pid, exc)
        return None


def release_vouch(handle: str | None) -> None:
    """Stop vouching: drop the lock :func:`vouch_for_child` returned. No-op for None or a handle
    this process does not hold (already released, or inherited across a fork)."""
    if handle:
        _release(handle)


def _check_usable(directory: Path) -> None:
    """Raise ``OSError`` unless owner locks can be taken in *directory* right now. An owners dir
    that cannot be created, written or locked means holders may have failed to register: the
    caller must treat ownership as unknown rather than as \"nobody\"."""
    directory.mkdir(parents=True, exist_ok=True)
    probe = os.path.join(directory, f".probe-{os.getpid()}")
    fd = os.open(probe, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if not _try_lock(fd):
            raise OSError(errno.EAGAIN, "probe lock unexpectedly busy", probe)
        _unlock(fd)
    finally:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(probe)


def live_owner_pids(home: str | Path | None = None) -> dict[int, str]:
    """Pid -> lock file for every owner lock held under *home*'s root, read straight from the OS
    lock state. A file whose lock this pass can take is stale (its holder died): it is removed.
    Raises ``OSError`` when ownership cannot be known: the owners dir cannot be created, listed or
    written, or its filesystem cannot lock (the caller fails closed and reaps nothing)."""
    directory = owners_dir(home)
    try:
        _check_usable(directory)
        names = os.listdir(directory)
    except OSError as exc:
        raise OSError(exc.errno, f"scratch owner locks unavailable in {directory}: {exc}") from exc
    owners: dict[int, str] = {}
    for name in sorted(names):
        pid = owner_pid(name)
        if pid is None:
            continue
        path = os.path.join(directory, name)
        if _ours(path) or _held_elsewhere(path):
            owners.setdefault(pid, path)
    return owners


def _held_elsewhere(path: str) -> bool:
    """Probe *path*: True when another process holds it (or it cannot be probed: an unknown is
    never read as free). A free file is stale and removed."""
    for _ in range(_ACQUIRE_TRIES):
        verdict = _probe(path)
        if verdict is not None:
            return verdict
    return True


def _probe(path: str) -> bool | None:
    """One probe of :func:`_held_elsewhere`; None when the inode we locked is no longer the one
    *path* names (a concurrent prune removed it and a new holder may have recreated the name)."""
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return False  # its holder exited cleanly while we listed
    except OSError:
        return True
    try:
        if not _try_lock(fd):
            return True
        if _IS_WINDOWS:
            _unlock(fd)
        elif not _still_named(fd, path):
            return None  # unlinking by name now would remove the new holder's live file
        else:
            with contextlib.suppress(OSError):
                os.unlink(path)  # under our lock: a new holder re-checks the inode and retries
    except OSError:
        return True
    finally:
        os.close(fd)
    if _IS_WINDOWS:
        # Removal fails while anyone (a new holder of this name) has it open: left alone.
        with contextlib.suppress(OSError):
            os.unlink(path)
    return False


def ancestry(pid: int) -> list[int]:
    """*pid* and its ancestors through raw ``ppid()`` links, nearest first.

    Never ``psutil.parents()``: it drops a parent whose start time is later than the child's,
    which a backward clock step produces. On Linux/macOS a live child's ppid is a live process
    (orphans are reparented). On Windows a ppid can name a dead parent whose pid was reused; the
    chain then errs toward keeping. The seen-set stops a reuse cycle."""
    import psutil

    chain: list[int] = []
    current: int | None = pid
    while current and current not in chain:
        chain.append(current)
        try:
            current = psutil.Process(current).ppid()
        except (psutil.Error, OSError):
            break
    return chain


def owning_holder(pid: int, owners: dict[int, str] | set[int]) -> int | None:
    """The owner-lock holder covering *pid*: itself or the nearest holder on its parent chain.
    pid 1 never owns: a Hermes running as a container init adopts every orphan."""
    return next((p for p in ancestry(pid) if p in owners and p != 1), None)
