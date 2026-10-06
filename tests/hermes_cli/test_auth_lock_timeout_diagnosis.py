"""The auth-store lock timeout must name the lock file, and a holder only when one is real.

``_auth_store_lock`` serializes auth.json transactions across processes, but a holder can keep
the lock across a slow OAuth refresh POST, so a waiter (e.g. a Desktop assistant start) times out
after ~15s while every credential is perfectly fine (#124533). The TimeoutError names the lock
file, and names the holder only when the holder-pid sidecar carries a live foreign pid — never
as an unconditional literal for a timeout with no holder. The pid lives in a sidecar (never the
lock file: msvcrt.locking covers its byte 0, so a stamp there is unreadable while the holder is
live and only surfaces once it is dead) and the copy hedges that a pid is existence-checked at
best and may already be reused. Permanent lock failures (flock-unsupported filesystem,
ENOSYS/EOPNOTSUPP) must propagate immediately instead of burning the deadline telling the user
to kill a process that does not exist.
"""

from __future__ import annotations

import errno
import os

import pytest


def _busy_kernel_lock(*args, **kwargs):
    raise BlockingIOError()


def _holder_sidecar(auth_path):
    import hermes_cli.auth as auth

    lock = auth_path.with_suffix(".lock")
    return auth._lock_holder_sidecar_path(lock)


def test_lock_timeout_names_the_lock_file_and_a_live_holder(tmp_path, monkeypatch):
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "_kernel_lock", _busy_kernel_lock)
    holder_pid = os.getppid()  # the test runner's parent: a live foreign process by construction
    auth_path = tmp_path / "profiles" / "coder" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    _holder_sidecar(auth_path).write_text(f"{holder_pid}\n", encoding="utf-8")

    with pytest.raises(TimeoutError) as excinfo:
        with auth._auth_store_lock(timeout_seconds=0.01, target_path=auth_path):
            pass

    message = str(excinfo.value)
    assert "Timed out waiting for auth store lock" in message
    assert str(auth_path.with_suffix(".lock")) in message  # which file is contended
    assert f"pid {holder_pid}" in message  # and who is holding it — detected, not assumed
    assert "if it is still running" in message  # hedged: existence-checked, not identity-checked


def test_lock_timeout_without_a_holder_stays_silent_about_one(tmp_path, monkeypatch):
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "_kernel_lock", _busy_kernel_lock)
    auth_path = tmp_path / "profiles" / "coder" / "auth.json"

    with pytest.raises(TimeoutError) as excinfo:
        with auth._auth_store_lock(timeout_seconds=0.01, target_path=auth_path):
            pass

    message = str(excinfo.value)
    assert str(auth_path.with_suffix(".lock")) in message
    assert "another hermes process" not in message  # no holder detected: no one to blame


def test_holder_hint_reads_the_sidecar_never_the_byte_locked_lock_file(tmp_path):
    """A pid inside the lock file itself must not be reported as a holder.

    On Windows msvcrt.locking covers byte 0 of the lock file, so the waiter's read of that
    byte fails with EACCES exactly while a holder is live — and succeeds only once the holder
    died without releasing. Reading the sidecar instead is what makes the hint observable on
    both platforms (round 2 of #124533).
    """
    import hermes_cli.auth as auth

    auth_path = tmp_path / "profiles" / "coder" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.with_suffix(".lock").write_text(f"{os.getppid()}\n", encoding="utf-8")

    assert auth._lock_holder_hint(auth_path.with_suffix(".lock")) == ""


def test_holder_pid_is_stamped_in_the_sidecar_and_removed_on_release(tmp_path):
    """The live acquire/release path stamps the sidecar, never the lock file body."""
    import hermes_cli.auth as auth

    auth_path = tmp_path / "profiles" / "coder" / "auth.json"
    lock = auth_path.with_suffix(".lock")

    with auth._auth_store_lock(timeout_seconds=1.0, target_path=auth_path):
        assert _holder_sidecar(auth_path).read_text(encoding="utf-8-sig").strip() == str(os.getpid())
        assert lock.read_text(encoding="utf-8-sig").strip() != str(os.getpid())

    assert not _holder_sidecar(auth_path).exists()


def test_lock_timeout_ignores_a_stale_pid_from_a_dead_holder(tmp_path, monkeypatch):
    """A leftover sidecar pid from a dead holder is filtered cross-platform.

    The liveness probe is ``gateway.status._pid_exists`` (psutil / OpenProcess / os.kill sig 0),
    so a dead pid stays silent on Windows too — not only where os.kill(pid, 0) is safe.
    """
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "_kernel_lock", _busy_kernel_lock)
    stale_pid = 2 ** 22  # far beyond any pid namespace: the probe must report "no such process"
    auth_path = tmp_path / "profiles" / "coder" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    _holder_sidecar(auth_path).write_text(f"{stale_pid}\n", encoding="utf-8")

    with pytest.raises(TimeoutError) as excinfo:
        with auth._auth_store_lock(timeout_seconds=0.01, target_path=auth_path):
            pass

    assert f"pid {stale_pid}" not in str(excinfo.value)


def test_permanent_lock_failure_propagates_instead_of_burning_the_deadline(tmp_path, monkeypatch):
    import hermes_cli.auth as auth

    def _unsupported(*args, **kwargs):
        raise OSError(errno.ENOSYS, "flock not supported on this filesystem")

    monkeypatch.setattr(auth, "_kernel_lock", _unsupported)
    auth_path = tmp_path / "profiles" / "coder" / "auth.json"

    with pytest.raises(OSError) as excinfo:
        with auth._auth_store_lock(timeout_seconds=0.01, target_path=auth_path):
            pass

    assert excinfo.value.errno == errno.ENOSYS  # the original error, not a lock-holder blame


def test_lock_timeout_message_still_matches_the_documented_prefix():
    # Downstream diagnosis (tui_gateway.user_messages) matches on this prefix; keep it stable.
    import hermes_cli.auth as auth
    import inspect

    source = inspect.getsource(auth._auth_store_lock)
    assert "Timed out waiting for auth store lock" in source
