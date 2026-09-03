"""Regression test for a Windows-only PermissionError under lock contention.

``_file_lock()`` (backing ``_auth_store_lock()``) ensures its lock file has
at least 1 byte of content before opening it for ``msvcrt.locking()`` (which
requires a non-empty file). That "ensure content" write was unguarded: under
real concurrency, one thread/process's ``msvcrt.locking()`` byte-range lock
can already be held on the file at the exact moment another thread runs the
same ensure-content check, and the write collides and raises
``PermissionError`` -- uncaught, since it happens before the retry loop the
rest of the function uses for exactly this kind of contention.

This was found by a stress test for the Anthropic OAuth credential-pool fix
(``tests/agent/test_anthropic_oauth_stress.py``), which reliably reproduces
it (16/20 concurrent refreshes hit PermissionError pre-fix, deterministically
across repeated runs) because the surrounding CredentialPool work widens the
race window enough for real OS-level thread interleaving to land on it. That
test is the authoritative regression guard for the exact failure mode. This
file is deliberately a minimal, isolated stress test against
``_auth_store_lock()`` directly -- best-effort coverage that does not always
land on the same narrow window in isolation, but exercises the SAME shared
primitive Codex/xAI/Nous already depend on for their single-use-refresh-token
protection, so this bug was never Anthropic-specific.
"""

from __future__ import annotations

import threading

import pytest

from hermes_cli.auth import _auth_store_lock

CONCURRENCY = 40


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.windows_only
def test_many_concurrent_lock_acquisitions_do_not_raise_permission_error(hermes_home):
    """CONCURRENCY threads race to acquire/release the same auth-store lock.

    None of them should ever see an uncaught PermissionError/OSError from
    the lock-file "ensure content" pre-check -- only the intended
    TimeoutError (never expected here, since each critical section is
    instant) should be a possible failure mode.
    """
    errors: dict[int, BaseException] = {}
    entered = 0
    entered_lock = threading.Lock()
    # Synchronize all threads to hit the lock-file "ensure content" pre-check
    # at (as close as the OS scheduler allows to) the exact same instant --
    # the race only exists on the very first acquisition against a fresh
    # lock file, so a tight barrier maximizes the odds of reproducing it
    # instead of relying on incidental thread-start jitter.
    barrier = threading.Barrier(CONCURRENCY)

    def _run(idx: int) -> None:
        nonlocal entered
        try:
            barrier.wait(timeout=10)
            with _auth_store_lock(timeout_seconds=10.0):
                with entered_lock:
                    entered += 1
        except BaseException as exc:  # pragma: no cover - failure diagnostics
            errors[idx] = exc

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(CONCURRENCY)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    still_alive = [t for t in threads if t.is_alive()]
    assert not still_alive, f"{len(still_alive)}/{CONCURRENCY} threads never finished"
    assert not errors, f"unexpected exceptions acquiring the lock concurrently: {errors!r}"
    assert entered == CONCURRENCY
