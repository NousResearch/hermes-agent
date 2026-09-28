"""Nonblocking, all-or-nothing reservations of canonical profile homes.

Lock inodes are never removed: unlinking a locked inode creates a second owner.
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
                    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
                    fd = os.open(path, flags, 0o600)
                    handle = os.fdopen(fd, 'r+', encoding='utf-8')
                    try:
                        info = os.fstat(handle.fileno())
                        if not stat.S_ISREG(info.st_mode) or (os.name != 'nt' and info.st_uid != os.getuid()):  # windows-footgun: ok — guarded UID
                            raise PermissionError('unsafe gateway lock owner or type')
                        if not _try_acquire_file_lock(handle):
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
