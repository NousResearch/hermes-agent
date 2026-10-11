"""Nonblocking, all-or-nothing reservations of canonical profile homes.

A held lock inode is never removed: unlinking it creates a second owner. The one exception is a
stale inode this user cannot use (left by a gateway that ran as root, #42685) that is provably
unheld; it is replaced once, and every winner checks that the path still names the inode it locked.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import threading


class OwnershipConflict(RuntimeError):
    pass


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
    if not stat.S_ISREG(info.st_mode) or (os.name != 'nt' and info.st_uid != os.getuid()):  # windows-footgun: ok — POSIX only
        raise PermissionError('unsafe gateway lock owner or type')
    if info.st_nlink != 1:  # a second name (cp -al / rsync --link-dest clone) shares the record
        raise PermissionError('unsafe gateway lock link count')
    if os.name != 'nt' and stat.S_IMODE(info.st_mode) & 0o077:
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


def _inode_locked_per_proc(ino: int) -> bool | None:
    """Linux ``/proc/locks``: whether any lock names inode *ino* (on any device, so a btrfs/overlay
    st_dev mismatch can only refuse, never miss a holder). None where the table is unavailable."""
    try:
        table = Path('/proc/locks').read_text(encoding='ascii', errors='replace')
    except OSError:
        return None
    suffix = f':{ino}'
    return any(field.endswith(suffix) and field.count(':') == 2
               for line in table.splitlines() for field in line.split())


def _unlink_stale_lock(path: Path) -> bool:
    """Unlink a ``gateway.lock`` this user cannot use (unopenable, or another uid's) when no process
    holds it: a flock probe through a read-only descriptor, else Linux ``/proc/locks``. Symlinks,
    hardlinks, non-regular files, a home whose ``gateway.pid`` names a live process and anything not
    provably unheld are left in place (False)."""
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
        held = _inode_locked_per_proc(info.st_ino)
        if held is None:  # no /proc/locks (macOS/BSD): nothing can prove it unheld, so never guess
            raise PermissionError(foreign_lock_recovery(path)) from None
        return held is False and _unlink_if_same(path, info)
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

    def reserve(self, homes) -> None:
        from gateway.status import _try_acquire_file_lock, _build_pid_record
        with self._mutex:
            added = []
            try:
                for home in sorted({canonical_home(p) for p in homes}):
                    if home in self._handles:
                        continue
                    home.mkdir(mode=0o700, parents=True, exist_ok=True)
                    path = home / 'gateway.lock'
                    fd = _open_lock(path)
                    handle = os.fdopen(fd, 'r+', encoding='utf-8')  # windows-footgun: ok — write-only
                    try:
                        if not _try_acquire_file_lock(handle) or not _still_names(path, handle.fileno()):
                            raise OwnershipConflict(f'Gateway runtime already owns profile {home}')
                        record = {**_build_pid_record(), 'hermes_home': str(home)}
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


@contextmanager
def exclusive_maintenance(homes):
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
        owner.reserve(homes)
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
