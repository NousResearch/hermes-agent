"""The auth-store lock timeout must name the lock file, and a holder only when one is real.

``_auth_store_lock`` serializes auth.json transactions across processes, but a holder can keep
the lock across a slow OAuth refresh POST, so a waiter (e.g. a Desktop assistant start) times out
after ~15s while every credential is perfectly fine (#124533). The TimeoutError names the lock
file, and names the holder only when the lock file carries a live foreign pid — never as an
unconditional literal for a timeout with no holder. Permanent lock failures (flock-unsupported
filesystem, ENOSYS/EOPNOTSUPP) must propagate immediately instead of burning the deadline
telling the user to kill a process that does not exist.
"""

from __future__ import annotations

import errno
import os

import pytest


def _busy_kernel_lock(*args, **kwargs):
    raise BlockingIOError()


def test_lock_timeout_names_the_lock_file_and_a_live_holder(tmp_path, monkeypatch):
    from hermes_cli import auth

    monkeypatch.setattr(auth, "_kernel_lock", _busy_kernel_lock)
    holder_pid = os.getppid()  # the test runner's parent: a live foreign process by construction
    auth_path = tmp_path / "profiles" / "coder" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.with_suffix(".lock").write_text(f"{holder_pid}\n", encoding="utf-8")

    with pytest.raises(TimeoutError) as excinfo:
        with auth._auth_store_lock(timeout_seconds=0.01, target_path=auth_path):
            pass

    message = str(excinfo.value)
    assert "Timed out waiting for auth store lock" in message
    assert str(auth_path.with_suffix(".lock")) in message  # which file is contended
    assert f"pid {holder_pid}" in message  # and who is holding it — detected, not assumed


def test_lock_timeout_without_a_holder_stays_silent_about_one(tmp_path, monkeypatch):
    from hermes_cli import auth

    monkeypatch.setattr(auth, "_kernel_lock", _busy_kernel_lock)
    auth_path = tmp_path / "profiles" / "coder" / "auth.json"

    with pytest.raises(TimeoutError) as excinfo:
        with auth._auth_store_lock(timeout_seconds=0.01, target_path=auth_path):
            pass

    message = str(excinfo.value)
    assert str(auth_path.with_suffix(".lock")) in message
    assert "another hermes process" not in message  # no holder detected: no one to blame


@pytest.mark.skipif(os.name != "posix", reason="holder liveness probe is POSIX-only (os.kill sig 0)")
def test_lock_timeout_ignores_a_stale_pid_from_a_dead_holder(tmp_path, monkeypatch):
    from hermes_cli import auth

    monkeypatch.setattr(auth, "_kernel_lock", _busy_kernel_lock)
    stale_pid = 2 ** 22  # far beyond any pid namespace: probe must report "no such process"
    real_kill = os.kill

    def _no_such_process(pid, sig):
        if pid == stale_pid:
            raise ProcessLookupError()
        return real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", _no_such_process)
    auth_path = tmp_path / "profiles" / "coder" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.with_suffix(".lock").write_text(f"{stale_pid}\n", encoding="utf-8")

    with pytest.raises(TimeoutError) as excinfo:
        with auth._auth_store_lock(timeout_seconds=0.01, target_path=auth_path):
            pass

    assert f"pid {stale_pid}" not in str(excinfo.value)


def test_permanent_lock_failure_propagates_instead_of_burning_the_deadline(tmp_path, monkeypatch):
    from hermes_cli import auth

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
    from hermes_cli import auth
    import inspect

    source = inspect.getsource(auth._auth_store_lock)
    assert "Timed out waiting for auth store lock" in source
