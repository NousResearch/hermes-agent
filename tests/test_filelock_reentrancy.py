"""_FileLock Windows mitigations: same-thread nested reentrancy and cross-thread bounded wait.

msvcrt.locking is per-FILE*-handle, not per-process, so nested acquisition through two
handles used to self-deadlock (errno 36 after LK_LOCK's ~10s retry). These tests pin the
two behaviors the fix promises: nested same-path acquisition succeeds, and a different
lock path in the same thread still takes a real lock (per-path keying), plus the bounded
cross-thread wait. POSIX flock is process-idempotent, so the reentrancy path is a no-op
there and the tests still hold.
"""

import threading
import time

import pytest

from hermes_cli.active_sessions import _FileLock


def test_nested_same_path_same_thread_succeeds(tmp_path):
    lock = tmp_path / "a.lock"
    with _FileLock(lock):
        with _FileLock(lock):
            with _FileLock(lock):
                pass
    # fully released: another acquisition must take the real lock again
    with _FileLock(lock):
        pass


def test_reentrancy_is_keyed_by_path(tmp_path):
    # A different lock file in the same thread must NOT be skipped by the
    # reentrancy bookkeeping — only the same resolved path is reentrant.
    a, b = tmp_path / "a.lock", tmp_path / "b.lock"
    holding = threading.Event()

    def hold_b():
        with _FileLock(b):
            holding.set()
            time.sleep(0.8)

    t = threading.Thread(target=hold_b)
    t.start()
    assert holding.wait(timeout=5.0)
    with _FileLock(a):
        # b is held by another thread; acquiring it from THIS thread (which holds
        # a) must still contend on the real lock instead of returning instantly
        # via the reentrancy path (instant return would mean the bookkeeping
        # leaked across paths). It succeeds only after the holder releases.
        start = time.monotonic()
        with _FileLock(b):
            elapsed = time.monotonic() - start
    t.join()
    assert elapsed >= 0.5, f"acquired a foreign lock too fast ({elapsed:.3f}s) — reentrancy leaked across paths"


def test_cross_thread_wait_is_bounded_and_releases(tmp_path):
    lock = tmp_path / "c.lock"
    hold = threading.Event()
    release = threading.Event()

    def hold_lock():
        with _FileLock(lock):
            hold.set()
            release.wait(timeout=5.0)

    t = threading.Thread(target=hold_lock)
    t.start()
    assert hold.wait(timeout=5.0)

    acquired = {}

    def try_acquire():
        start = time.monotonic()
        try:
            with _FileLock(lock):
                acquired["ok"] = True
        except RuntimeError:
            acquired["ok"] = False
        acquired["elapsed"] = time.monotonic() - start

    release.set()
    w = threading.Thread(target=try_acquire)
    w.start()
    w.join(timeout=15.0)
    t.join(timeout=5.0)
    assert acquired.get("ok") is True
    assert acquired["elapsed"] < 10.0
