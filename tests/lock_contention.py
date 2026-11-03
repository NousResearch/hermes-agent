"""Event-driven replacements for "give the other writer N seconds" in serialization tests.

A lost-update test parks writer A mid-cycle and lets writer B run. Unserialized, B finishes; serialized,
B blocks on the lock, so waiting for B with a fixed timeout always sits the timeout out on the passing
path. These helpers signal the moment B is turned away by the lock instead, so the test waits for
"B finished OR B is blocked" (``tests.waiting.wait_until``) and moves on immediately either way.
"""

from __future__ import annotations

import threading


def watch_file_lock_contention(monkeypatch) -> threading.Event:
    """An event set when any thread is refused by ``hermes_cli.auth._file_lock``.

    ``_file_lock`` retries a non-blocking ``_kernel_lock`` while another holder has the file, so the
    first contention error is the proof that a second writer is waiting."""
    from hermes_cli import auth

    contended = threading.Event()
    real_kernel_lock = auth._kernel_lock

    def kernel_lock(lock_file, acquire):
        try:
            return real_kernel_lock(lock_file, acquire)
        except OSError as exc:
            if acquire and auth._is_lock_contention(exc):
                contended.set()
            raise

    monkeypatch.setattr(auth, "_kernel_lock", kernel_lock)
    return contended


class ContentionObservedLock:
    """A ``threading.Lock`` stand-in that sets ``contended`` when a caller has to wait for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        self.contended.set()
        return self._lock.acquire(blocking, timeout) if blocking else False

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
