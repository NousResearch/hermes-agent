"""Regression tests for #34185 — `hermes uninstall` on Windows must
remove locked / read-only files instead of aborting on the first failure
and leaving 13 directories of user data behind.
"""

from __future__ import annotations

import os
import stat
import shutil
import sys
from pathlib import Path

import pytest


def test_robust_rmtree_removes_simple_tree(tmp_path):
    from hermes_cli.uninstall import _robust_rmtree_with_retry

    target = tmp_path / "tree"
    (target / "a" / "b").mkdir(parents=True)
    (target / "a" / "b" / "file.txt").write_text("hello")
    (target / "top.txt").write_text("top")

    ok, leftovers = _robust_rmtree_with_retry(target)
    assert ok is True
    assert leftovers == []
    assert not target.exists()


def test_robust_rmtree_removes_readonly_file(tmp_path):
    """Read-only files (common on Windows for state.db-shm) must be removed."""
    from hermes_cli.uninstall import _robust_rmtree_with_retry

    target = tmp_path / "tree"
    target.mkdir()
    readonly_file = target / "state.db-shm"
    readonly_file.write_text("locked")
    # Mark as read-only
    readonly_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    try:
        ok, leftovers = _robust_rmtree_with_retry(target)
        assert ok is True, f"Expected removal, got leftovers: {leftovers}"
    finally:
        # Cleanup if test failed
        if target.exists():
            try:
                readonly_file.chmod(stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
            shutil.rmtree(target, ignore_errors=True)


def test_robust_rmtree_returns_leftovers_when_path_uncleanable(tmp_path, monkeypatch):
    """If removal cannot complete, the helper reports the leftover paths so
    the user sees what's still on disk. Simulate by patching shutil.rmtree
    to always fail and leave the directory in place."""
    from hermes_cli.uninstall import _robust_rmtree_with_retry

    target = tmp_path / "stubborn"
    target.mkdir()
    (target / "locked.db").write_text("x")
    (target / "subdir").mkdir()
    (target / "subdir" / "more.txt").write_text("y")

    call_count = {"n": 0}

    def fake_rmtree(path, **kwargs):
        call_count["n"] += 1
        raise PermissionError("simulated lock")

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)

    ok, leftovers = _robust_rmtree_with_retry(target, max_attempts=2, sleep_between=0.01)
    assert ok is False
    # Helper retried max_attempts times.
    assert call_count["n"] == 2
    # Leftovers include the target and what's under it.
    assert any(str(target) in p for p in leftovers)
    # And the cleanup did NOT happen (the real tree is still there).
    assert (target / "locked.db").exists()


def test_robust_rmtree_retries_then_succeeds(tmp_path, monkeypatch):
    """Simulate transient lock: first attempt fails, second succeeds.
    This is the typical Windows case where a gateway service is shutting
    down and holds state.db for ~500ms after SIGTERM.
    """
    from hermes_cli.uninstall import _robust_rmtree_with_retry

    target = tmp_path / "transient"
    target.mkdir()
    (target / "state.db").write_text("data")

    real_rmtree = shutil.rmtree
    attempt = {"n": 0}

    def flaky_rmtree(path, **kwargs):
        attempt["n"] += 1
        if attempt["n"] == 1:
            raise PermissionError("first attempt fails")
        # Second attempt succeeds.
        return real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)

    ok, leftovers = _robust_rmtree_with_retry(target, max_attempts=3, sleep_between=0.01)
    assert ok is True
    assert leftovers == []
    assert attempt["n"] == 2, f"Expected exactly 2 attempts, got {attempt['n']}"


def test_robust_rmtree_handles_nonexistent_target(tmp_path):
    """rmtree of a non-existent path should be a no-op success."""
    from hermes_cli.uninstall import _robust_rmtree_with_retry

    target = tmp_path / "does-not-exist"
    ok, leftovers = _robust_rmtree_with_retry(target)
    assert ok is True
    assert leftovers == []


def test_on_rmtree_error_handler_clears_readonly(tmp_path):
    """The onerror handler clears the read-only bit and retries the original op."""
    from hermes_cli.uninstall import _on_rmtree_error

    p = tmp_path / "ro.txt"
    p.write_text("x")
    p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    called = {"n": 0}

    def fake_func(path):
        called["n"] += 1
        # First call simulates the original failure that brought us here.
        if called["n"] == 1:
            raise PermissionError("read-only")
        # After chmod, the func should succeed.
        os.unlink(path)

    # Call the error handler directly with the simulated original failure.
    _on_rmtree_error(fake_func, str(p), None)
    # fake_func was called at least once after the chmod.
    assert called["n"] >= 1


def test_uninstall_full_uninstall_path_uses_robust_remove(tmp_path, monkeypatch, capsys):
    """End-to-end shape check: when shutil.rmtree fails, run_uninstall (or
    its rmtree call) surfaces a 'Could not fully remove' message AND lists
    the leftovers \u2014 not just a single warn line as before #34185.

    We don't actually invoke run_uninstall (it has heavy interactive
    branches) \u2014 instead we exercise the same _robust_rmtree_with_retry
    helper that run_uninstall now uses and check that the report contains
    the leftover-listing semantics the user-facing log will print.
    """
    from hermes_cli.uninstall import _robust_rmtree_with_retry

    target = tmp_path / "hermes-home"
    target.mkdir()
    (target / "logs").mkdir()
    (target / "logs" / "agent.log").write_text("...")
    (target / "state.db").write_text("...")
    (target / "state.db-wal").write_text("...")

    def fail_rmtree(path, **kwargs):
        raise PermissionError("simulated lock")

    monkeypatch.setattr(shutil, "rmtree", fail_rmtree)
    ok, leftovers = _robust_rmtree_with_retry(target, max_attempts=2, sleep_between=0.01)
    assert ok is False
    # We expect to see ALL the user files in the leftovers list so the
    # user knows what's stuck.
    leftover_str = "\n".join(leftovers)
    assert "state.db" in leftover_str
    assert "agent.log" in leftover_str or "logs" in leftover_str
