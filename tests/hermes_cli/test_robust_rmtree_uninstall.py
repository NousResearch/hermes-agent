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


def test_run_uninstall_full_uses_robust_remove_and_reports_leftovers(
    tmp_path, monkeypatch, capsys
):
    """Caller-level regression for #34185.

    Drive the real ``run_uninstall`` full-uninstall path with a stubbed
    interactive prompt and all *other* destructive side effects mocked out,
    then assert:

    1. The production caller actually routes ~/.hermes removal through
       ``_robust_rmtree_with_retry`` (not a bare ``shutil.rmtree``).
    2. When removal cannot complete, the user-facing output lists the
       leftover paths AND the platform-specific manual-cleanup remediation —
       the behaviour that was missing before #34185 (a single warn line).
    """
    import hermes_cli.uninstall as un

    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "logs").mkdir(parents=True)
    (hermes_home / "logs" / "agent.log").write_text("...")
    (hermes_home / "state.db").write_text("...")
    project_root = tmp_path / "hermes-agent"
    project_root.mkdir()

    # Point the uninstaller at our temp dirs.
    monkeypatch.setattr(un, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(un, "get_project_root", lambda: project_root)

    # No named profiles; keep this a plain default-profile full uninstall.
    monkeypatch.setattr(un, "_is_default_hermes_home", lambda h: True)
    monkeypatch.setattr(un, "_discover_named_profiles", lambda: [])
    monkeypatch.setattr(un, "_is_windows", lambda: False)

    # Neutralize every other destructive / environment-touching step so the
    # test only exercises the ~/.hermes removal branch.
    monkeypatch.setattr(un, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(un, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(un, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(un, "remove_node_symlinks", lambda h: [])

    # Track that the robust helper is the thing the caller invokes, and force
    # it to report an unremovable tree so the leftover-listing path runs.
    calls = {"robust": 0}

    def fake_robust(target, *args, **kwargs):
        calls["robust"] += 1
        return False, [str(hermes_home / "state.db"), str(hermes_home / "logs" / "agent.log")]

    monkeypatch.setattr(un, "_robust_rmtree_with_retry", fake_robust)

    # A bare shutil.rmtree on hermes_home would be the pre-#34185 bug — make
    # it fail loudly if the caller regresses to using it for the data dir.
    real_rmtree = shutil.rmtree

    def guarded_rmtree(path, *args, **kwargs):
        assert Path(path) != hermes_home, (
            "run_uninstall must remove ~/.hermes via _robust_rmtree_with_retry, "
            "not a bare shutil.rmtree"
        )
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", guarded_rmtree)

    # Stub the interactive prompts: option 2 (full), then "yes" to confirm.
    answers = iter(["2", "yes"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))

    un.run_uninstall(args=None)

    assert calls["robust"] == 1, "run_uninstall did not use _robust_rmtree_with_retry"

    out = capsys.readouterr().out
    assert "Could not fully remove" in out
    assert "state.db" in out
    assert "agent.log" in out
    # POSIX remediation hint (we forced _is_windows() False).
    assert "rm -rf" in out
