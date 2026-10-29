"""Nonblocking, all-or-nothing reservations of canonical profile homes.

A held lock inode is never removed: unlinking it creates a second owner. The one exception is a
stale inode this user cannot write (left by a gateway that ran as root, #42685) that a flock through
a read-only descriptor proves unheld; it is replaced once, and every winner checks that the path
still names the inode it locked.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import stat
import threading
import time
from hermes_dir_acl import make_user_dir

logger = logging.getLogger(__name__)


class OwnershipConflict(RuntimeError):
    pass


class ReservationCancelled(OwnershipConflict):
    """The waiting reserve's caller gave up: nothing reserved, nothing left held."""


def canonical_home(home: Path) -> Path:
    return Path(os.path.normcase(str(Path(home).expanduser().resolve())))


def tighten_lock_mode(fd: int) -> None:
    """Validate an open ``gateway.lock`` descriptor and narrow an upgraded lock to 0600.

    Main's ``acquire_gateway_runtime_lock`` created the inode with ``open(path, "a+")`` under the
    umask (0644 on most hosts), and lock inodes are never replaced, so an upgraded home keeps that
    mode forever and discovery refuses it. Only a regular, single-link inode this user owns is
    chmodded, through the already-open NOFOLLOW descriptor (no path race); anything else is refused.
    """
    info = os.fstat(fd)
    if os.name == 'nt':
        if not stat.S_ISREG(info.st_mode):
            raise PermissionError('unsafe gateway lock owner or type')
        return
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():  # windows-footgun: ok — POSIX branch
        raise PermissionError('unsafe gateway lock owner or type')
    if stat.S_IMODE(info.st_mode) & 0o077:
        if info.st_nlink != 1:
            raise PermissionError('unsafe gateway lock link count')
        os.fchmod(fd, 0o600)  # windows-footgun: ok — POSIX branch


_LOCK_FLAGS = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)


def _open_checked(path: Path, flags: int) -> int:
    fd = os.open(path, flags, 0o600)
    try:
        tighten_lock_mode(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _unlink_stale_lock(path: Path) -> bool:
    """Unlink a ``gateway.lock`` this user cannot write (another uid's) once a flock taken through a
    read-only descriptor proves it unheld. Symlinks, hardlinks, non-regular files, a home whose
    ``gateway.pid`` names a live process and anything not provably unheld are left in place (False).
    An inode this user cannot even open is refused with the recovery command: an unheld snapshot
    (``/proc/locks``) does not stop a holder locking it between the read and the unlink."""
    from gateway.status import _live_pid_from_record, _read_pid_record
    if os.name == 'nt':
        return False  # a sharing violation is a live holder there
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        return False
    if _live_pid_from_record(_read_pid_record(path.with_name('gateway.pid'))) is not None:
        return False
    try:
        probe = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK)
    except PermissionError:
        raise PermissionError(foreign_lock_recovery(path)) from None
    import fcntl
    try:
        if not _same_inode(os.fstat(probe), info):
            return False
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        # Unlinked while still locked: a late locker of the old inode never shares our lock.
        return _unlink_if_same(path, info)
    finally:
        os.close(probe)


def foreign_lock_recovery(path: Path) -> str:
    return (f'foreign_stale_lock: {path} is not readable by this user (left by a gateway run as root?) '
            f'and cannot be proven unheld. If `sudo lsof {path}` lists no process, run `sudo rm {path}`.')


def unreadable_lock_recovery(path: Path) -> str | None:
    """The recovery a client prints for a ``gateway.lock`` it cannot even open read-only."""
    try:
        os.close(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK))
    except PermissionError:
        return foreign_lock_recovery(path)
    except OSError:
        pass
    return None


def _unlink_if_same(path: Path, info) -> bool:
    if not _same_inode(os.lstat(path), info):
        return False
    os.unlink(path)
    return True


def _same_inode(left, right) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_lock(path: Path) -> int:
    """Descriptor for *path*, replacing a provably stale inode this user cannot use once."""
    try:
        return _open_checked(path, _LOCK_FLAGS)
    except PermissionError:
        if not _unlink_stale_lock(path):
            raise
    # A racer may create the fresh inode first; both then contend for its flock as usual.
    return _open_checked(path, _LOCK_FLAGS)


def _still_names(path: Path, fd: int) -> bool:
    """True when *path* is still the inode *fd* locked (a stale-lock replacement unlinks it)."""
    if os.name == 'nt':
        return True
    try:
        return _same_inode(os.lstat(path), os.fstat(fd))
    except FileNotFoundError:
        return False


class ProfileOwnership:
    def __init__(self):
        self._handles: dict[Path, object] = {}
        self._writers: list[threading.Thread] = []
        self._mutex = threading.RLock()

    @property
    def homes(self) -> tuple[Path, ...]:
        with self._mutex:
            return tuple(self._handles)

    def reserve(self, homes, *, purpose: str | None = None) -> None:
        """``purpose`` marks a maintenance reservation in the lock record, so a contender can tell
        a short maintenance holder from a gateway runtime owner (:func:`live_runtime_owner`)."""
        from gateway.status import _try_acquire_file_lock, _build_pid_record
        with self._mutex:
            added = []
            try:
                for home in sorted({canonical_home(p) for p in homes}):
                    if home in self._handles:
                        continue
                    make_user_dir(home, mode=0o700, parents=True, exist_ok=True)
                    path = home / 'gateway.lock'
                    fd = _open_lock(path)
                    handle = os.fdopen(fd, 'r+', encoding='utf-8')  # windows-footgun: ok — write-only
                    try:
                        if not _try_acquire_file_lock(handle) or not _still_names(path, handle.fileno()):
                            raise OwnershipConflict(f'Gateway runtime already owns profile {home}')
                        record = {**_build_pid_record(), 'hermes_home': str(home)}
                        if purpose:
                            record['purpose'] = purpose
                        handle.seek(0)
                        handle.truncate()
                        json.dump(record, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    except BaseException:
                        handle.close()
                        raise
                    self._handles[home] = handle
                    added.append(home)
            except BaseException:
                for home in reversed(added):
                    self.release(home)
                raise

    def owns(self, home: Path) -> bool:
        with self._mutex:
            return canonical_home(home) in self._handles

    def release(self, home: Path) -> None:
        from gateway.status import _release_file_lock
        with self._mutex:
            handle = self._handles.pop(canonical_home(home), None)
            if handle is not None:
                _release_file_lock(handle)
                handle.close()

    def start_writer(self, thread: threading.Thread) -> None:
        """Register before starting so partial startup cannot forget a live writer."""
        with self._mutex:
            self._writers.append(thread)
            thread.start()

    def close(self) -> None:
        with self._mutex:
            # A timed-out daemon can still write during finally/atexit. Keep the
            # handles alive; process exit releases them atomically in the OS.
            if any(thread.is_alive() for thread in self._writers):
                return
            self._writers.clear()
            for home in reversed(self.homes):
                self.release(home)


process_ownership = ProfileOwnership()
_maintenance = threading.local()


def maintenance_holder(home: Path, purpose: str) -> dict | None:
    """The live holder record of *home*'s ``gateway.lock`` when it is a ``purpose`` maintenance
    reservation (see :meth:`ProfileOwnership.reserve`), else None. Advisory only: it names the
    holder for a refusal or a short wait, it never grants the lock."""
    from gateway.status import _live_pid_from_record, _read_gateway_lock_record
    record = _read_gateway_lock_record(canonical_home(home) / 'gateway.lock')
    if not record or record.get('purpose') != purpose or _live_pid_from_record(record) is None:
        return None
    return record


def lock_record_unpublished(home: Path) -> bool:
    """True when *home*'s ``gateway.lock`` record names no live holder yet: empty, torn, or a dead
    previous holder's. :meth:`ProfileOwnership.reserve` takes the flock BEFORE it writes the record,
    so a contender that loses the flock in that window cannot tell maintenance from a gateway; it
    must retry (bounded) instead of reporting a runtime owner. A record naming the CALLER is the
    caller's own rolled-back attempt (an all-or-nothing reserve writes each home's record before a
    later home refuses), so whoever holds the flock now has not published either. Advisory only,
    never grants the lock."""
    from gateway.status import _live_pid_from_record, _read_gateway_lock_record
    record = _read_gateway_lock_record(canonical_home(home) / 'gateway.lock')
    return not record or _live_pid_from_record(record) in (None, os.getpid())


# A client's discovery probe (gateway.status._probe_lock_file) holds a gateway.lock flock for an
# instant; the next attempt after this pause normally finds it gone.
PROBE_RETRY_BACKOFF_S = 0.015
TRANSIENT_HOLDER_POLL_S = 0.1


def _conflicting_homes(homes):
    """The homes whose gateway.lock another holder has right now: each probed alone with its own open
    file description and released at once (the all-or-nothing reserve does not say which failed).
    A home freed meanwhile is simply not listed; the next reserve attempt takes it."""
    conflicting = []
    for home in homes:
        probe = ProfileOwnership()
        try:
            probe.reserve([home])
        except OwnershipConflict:
            conflicting.append(home)
        finally:
            probe.close()
    return conflicting


def reserve_waiting_out_transient_holders(ownership, homes, *, wait_s: float,
                                          cancel: threading.Event | None = None) -> None:
    """``ownership.reserve(homes)``, retried until *wait_s* passes while every contender is
    transient: a client's lock probe (gone by the re-check), a browse reconcile, or a holder that has
    not published its record yet (flock taken, record unwritten). A positively identified live owner
    on any home refuses at once. Every grant is a real ``reserve``: the flock is never bypassed.
    Shared by the gateway boot and the running multiplexer's hot-serve of a new profile.

    *cancel* set (a worker thread's caller gave up) ends the wait at once and releases a grant
    that landed meanwhile, raising :class:`ReservationCancelled`: the caller may already have run
    its ownership cleanup, so nobody would ever release a home taken after it."""
    from hermes_state_reconcile import BROWSE_RECONCILE_PURPOSE
    cancel = threading.Event() if cancel is None else cancel
    deadline = time.monotonic() + wait_s
    logged = False
    while not cancel.is_set():
        before = set(ownership.homes)
        try:
            ownership.reserve(homes)
        except OwnershipConflict:
            # Only the homes that actually conflict decide: a real owner on any of them refuses at
            # once, even while another home is mid-browse (multiplexer boot, all-or-nothing).
            conflicting = _conflicting_homes(homes)
            holders = [h for h in conflicting
                       if maintenance_holder(h, BROWSE_RECONCILE_PURPOSE) or lock_record_unpublished(h)]
            if len(holders) < len(conflicting) or time.monotonic() >= deadline:
                raise
            if not conflicting:
                cancel.wait(PROBE_RETRY_BACKOFF_S)  # the holder already let go: a lock probe, never an owner
                continue
            if not logged:
                logger.info("Profile %s is held by a short session-store schema catch-up; waiting up to %.0fs",
                            holders[0], wait_s)
                logged = True
            cancel.wait(TRANSIENT_HOLDER_POLL_S)
            continue
        if not cancel.is_set():
            return
        for home in {canonical_home(p) for p in homes} - before:
            ownership.release(home)
    raise ReservationCancelled(f'Reservation of {", ".join(map(str, homes))} cancelled')


def live_runtime_owner(home: Path) -> int | None:
    """PID of a live runtime owner of *home* (a gateway, or a reservation that is not marked as
    maintenance), else None. A purpose-marked maintenance holder is never an owner."""
    from gateway.status import _live_pid_from_record, _read_gateway_lock_record, get_running_pid
    home = canonical_home(home)
    record = _read_gateway_lock_record(home / 'gateway.lock')
    live = _live_pid_from_record(record) if record else None
    if live is not None and record.get('purpose'):
        return None  # the flock holder is maintenance, so no gateway can hold it too
    return get_running_pid(home / 'gateway.pid', cleanup_stale=False) or live


@contextmanager
def exclusive_maintenance(homes, *, purpose: str | None = None):
    """Reserve the authority's exact lock before touching managed state.

    This is not an owner-presence check: the reservation remains held through
    publication, excluding startup even before its PID/control socket exists.
    Only nested synchronous maintenance in this thread may reuse a reservation;
    being the runtime owner (even in this process) never grants maintenance.
    Lock inodes must not be restored, removed, or replaced by callers.
    """
    state = getattr(_maintenance, 'state', None)
    outermost = state is None or state[0] != os.getpid()
    owner = state[1] if state is not None and not outermost else ProfileOwnership()
    previous = set(owner.homes)
    try:
        owner.reserve(homes, purpose=purpose)
        if outermost:
            _maintenance.state = (os.getpid(), owner)
        yield
    except OwnershipConflict as exc:
        raise OwnershipConflict(
            f'Exclusive maintenance refused: {exc}. Drain and stop the gateway, then retry.'
        ) from exc
    finally:
        for home in reversed(owner.homes):
            if home not in previous:
                owner.release(home)
        if outermost:
            _maintenance.state = None
