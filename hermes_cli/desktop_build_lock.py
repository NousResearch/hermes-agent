"""Cross-process serialization for the mutable Desktop build preflight."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import IO


class DesktopBuildLock:
    """Advisory lock held while npm installs and packages Hermes Desktop.

    ``node_modules`` and ``apps/desktop/release`` are checkout-scoped even
    when two commands use different Hermes profiles.  The lock therefore
    lives under the profile-common Hermes root and is keyed by the resolved
    checkout path.  Keeping it out of the checkout preserves ``--skip-build``
    launches from read-only/prebuilt source trees.  The open handle owns the
    lock, so the OS releases it automatically if a builder crashes.

    The build also writes the checkout an update writes, so the lock first takes the
    checkout lock every updater holds (``update_lock``; review C5), then its own: one lock
    order (checkout, then desktop build) for a direct ``hermes desktop`` build and for the
    update's build (which already holds the checkout lock and only joins it again), and no
    gap in which an updater could take the checkout while npm still writes.  Holding the
    checkout lock also puts the build in the update custody (``update_custody``): its
    descendants hold the checkout until they exit.
    """

    def __init__(self, project_root: Path) -> None:
        from hermes_constants import get_default_hermes_root

        resolved = os.path.normcase(str(project_root.resolve(strict=False)))
        checkout_key = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:24]
        self.path = get_default_hermes_root() / "locks" / f"desktop-build-{checkout_key}.lock"
        self.project_root = project_root
        self._handle: IO[str] | None = None
        self._checkout = None

    def acquire(self, *, wait: bool = False) -> bool:
        """Acquire the lock; queue behind the current holder when ``wait``.

        Returns ``False`` only when another process owns the lock and ``wait``
        is unset.  Filesystem errors propagate so callers fail explicitly
        instead of silently falling back to the corrupting concurrent
        behavior.
        """
        if self._handle is not None:
            return True
        if not self._acquire_checkout(wait=wait):
            return False
        try:
            acquired = self._acquire_build_lock(wait=wait)
        except BaseException:
            self._release_checkout()
            raise
        if not acquired:
            self._release_checkout()
        return acquired

    def _acquire_checkout(self, *, wait: bool) -> bool:
        """Take (or, inside an update, join) the checkout lock before the build lock (C5)."""
        import time

        from hermes_cli import update_lock

        held, wanted = update_lock._HELD, update_lock.checkout_lock_path(self.project_root)
        if held is not None and held["path"] != str(wanted):
            # This process holds a checkout lock under another path. Another spelling of this
            # checkout's lock file (a symlinked install root) is this update's own custody: build
            # inside it, since a second acquire would wait on its own lock. Any other checkout's
            # lock proves nothing about this one (review Q2), and a process records one hold only.
            if not (wanted.exists() and os.path.samefile(held["path"], wanted)):
                raise OSError(f"this process holds the update lock of another checkout ({held['path']}), "
                              f"so it cannot also lock {self.project_root} for its build")
            return True
        lock = update_lock.UpdateLock(install_root=self.project_root, checkout_first=False)
        announced = False
        while not lock.acquire_checkout(self.project_root):
            if lock.holder is not None and lock.holder.reason:
                raise OSError(lock.holder.reason)  # never build unserialized; the caller says so
            if not wait:
                return False
            if not announced:
                print("→ Waiting for a running Hermes update to release this checkout...")
                announced = True
            time.sleep(1.0)
        self._checkout = lock
        return True

    def _release_checkout(self) -> None:
        lock, self._checkout = self._checkout, None
        if lock is not None:
            lock.release()

    def _acquire_build_lock(self, *, wait: bool) -> bool:
        from gateway.status import _try_acquire_file_lock

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        if _try_acquire_file_lock(handle):
            self._handle = handle
            return True
        if not wait:
            handle.close()
            return False

        # Blocking mode queues behind the holder; say so instead of appearing
        # to hang with no output (the update path would rather wait for the
        # in-flight build it depends on than fail the update).
        print("→ Waiting for another Hermes desktop dependency install or build to finish...")
        print(f"  Lock: {self.path}")
        from gateway.status import _release_file_lock

        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            # Windows has no fcntl; gateway.status's non-blocking try below is
            # the best-effort serialization there (msvcrt.locking), so poll it.
            import time

            while True:
                if _try_acquire_file_lock(handle):
                    self._handle = handle
                    return True
                _release_file_lock(handle)
                time.sleep(1.0)
        self._handle = handle
        return True

    def release(self) -> None:
        """Release the lock when held.  Safe to call more than once."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None

        try:
            from gateway.status import _release_file_lock

            _release_file_lock(handle)
        finally:
            try:
                handle.close()
            finally:
                self._release_checkout()  # the narrower lock first: the order is checkout, build

    def __enter__(self) -> DesktopBuildLock:
        if not self.acquire():
            raise RuntimeError("desktop build lock is already held")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
