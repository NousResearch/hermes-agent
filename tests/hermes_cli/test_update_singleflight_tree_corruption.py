"""Tests for issue #70211 — hermes update corrupts editable-install working tree.

Verifies the three protections added:

1. **Single-flight lock** — ``_UpdateSingleflightLock`` serialises concurrent
   ``hermes update`` runs so stash/pull/npm cycles cannot interleave.
2. **Mid-stash guard** — ``_update_node_dependencies`` re-verifies the root
   ``package.json`` is present before running npm, aborting rather than
   letting npm rewrite ``package-lock.json`` into a broken shape.
3. **Lockfile restoration** — ``_restore_package_lockfiles_from_git`` discards
   any lockfile dirtied by npm after the update, so even if npm ran against
   an inconsistent tree, the lockfile matches git.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


# ---------------------------------------------------------------------------
# _UpdateSingleflightLock
# ---------------------------------------------------------------------------


class TestUpdateSingleflightLock:
    """Tests for the single-flight lock that prevents concurrent updates."""

    def test_enter_succeeds_when_lock_free(self, tmp_path, monkeypatch):
        """The lock can be acquired when no other process holds it."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        lock = cli_main._UpdateSingleflightLock()
        lock.__enter__()
        assert lock._lock_fd is not None
        lock.__exit__(None, None, None)

    def test_second_attempt_exits_when_lock_held(self, tmp_path, monkeypatch):
        """A second concurrent lock attempt exits with SystemExit(1)."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        first = cli_main._UpdateSingleflightLock()
        first.__enter__()
        try:
            with pytest.raises(SystemExit) as exc_info:
                second = cli_main._UpdateSingleflightLock()
                second.__enter__()
            assert exc_info.value.code == 1
        finally:
            first.__exit__(None, None, None)

    def test_lock_released_after_exit(self, tmp_path, monkeypatch):
        """After __exit__, the same lock file can be re-acquired."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        first = cli_main._UpdateSingleflightLock()
        first.__enter__()
        first.__exit__(None, None, None)

        second = cli_main._UpdateSingleflightLock()
        second.__enter__()
        assert second._lock_fd is not None
        second.__exit__(None, None, None)

    def test_lock_skipped_on_non_posix(self, tmp_path, monkeypatch):
        """On non-POSIX (no fcntl), the lock is a no-op (_lock_fd is None)."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        with patch.dict("sys.modules", {"fcntl": None}):
            import builtins
            _real_import = builtins.__import__

            def _fake_import(name, *args, **kwargs):
                if name == "fcntl":
                    raise ImportError("No fcntl on this platform")
                return _real_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=_fake_import):
                lock = cli_main._UpdateSingleflightLock()
                lock.__enter__()
                assert lock._lock_fd is None
                lock.__exit__(None, None, None)

    def test_lock_contains_pid(self, tmp_path, monkeypatch):
        """The lock file contains the PID of the holder."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        lock = cli_main._UpdateSingleflightLock()
        lock.__enter__()
        try:
            lock_path = tmp_path / ".hermes-update.lock"
            content = lock_path.read_text()
            assert content.strip() == str(os.getpid())
        finally:
            lock.__exit__(None, None, None)

    def test_lock_best_effort_cleanup(self, tmp_path, monkeypatch):
        """After exit, the lock file is removed (best-effort)."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        lock = cli_main._UpdateSingleflightLock()
        lock.__enter__()
        lock_path = tmp_path / ".hermes-update.lock"
        assert lock_path.exists()
        lock.__exit__(None, None, None)
        assert not lock_path.exists()

    def test_lock_skips_when_project_root_sentinel(self, monkeypatch):
        """When PROJECT_ROOT is a sentinel that errors on /, skip the lock."""
        class _Sentinel:
            def __truediv__(self, other):
                raise RuntimeError("sentinel")

        monkeypatch.setattr(cli_main, "PROJECT_ROOT", _Sentinel())
        lock = cli_main._UpdateSingleflightLock()
        lock.__enter__()
        assert lock._lock_fd is None
        lock.__exit__(None, None, None)

    def test_cmd_update_impl_wraps_body_with_lock(self, tmp_path, monkeypatch):
        """_cmd_update_impl calls _cmd_update_body inside the lock."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        called = []

        def fake_body(*args, **kwargs):
            called.append(("body", args, kwargs))
            return "ok"

        with patch.object(cli_main, "_cmd_update_body", side_effect=fake_body):
            result = cli_main._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

        assert result == "ok"
        assert len(called) == 1


# ---------------------------------------------------------------------------
# _update_node_dependencies — mid-stash guard
# ---------------------------------------------------------------------------


class TestUpdateNodeDependenciesGuard:
    """Tests for the mid-stash guard that prevents npm install when
    root package.json is missing."""

    def test_skips_npm_when_root_package_json_missing(self, tmp_path, monkeypatch):
        """When root package.json is gone, npm install is skipped (early return)."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        result = cli_main._update_node_dependencies()
        assert result == []

    def test_mid_stash_guard_present_in_source(self):
        """The mid-stash guard code exists in the source file."""
        import inspect
        source = inspect.getsource(cli_main._update_node_dependencies)
        assert "package.json is missing" in source
        assert "mid-stash" in source


# ---------------------------------------------------------------------------
# _restore_package_lockfiles_from_git
# ---------------------------------------------------------------------------


class TestRestorePackageLockfiles:
    """Tests for restoring dirty package-lock.json files from git."""

    def test_restores_dirty_lockfiles(self, tmp_path, monkeypatch):
        """Dirty package-lock.json files are restored from git."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs.get("cwd")))
            joined = " ".join(str(p) for p in cmd)
            if "diff --name-only" in joined:
                return SimpleNamespace(
                    returncode=0,
                    stdout="package-lock.json\nweb/package-lock.json\n",
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
        cli_main._restore_package_lockfiles_from_git(["git"], tmp_path)

        checkout_calls = [
            c for c, w in calls if "checkout" in " ".join(str(p) for p in c)
        ]
        assert len(checkout_calls) == 1
        checkout_args = [str(p) for p in checkout_calls[0]]
        assert "package-lock.json" in checkout_args
        assert "web/package-lock.json" in checkout_args

    def test_noop_when_no_dirty_lockfiles(self, tmp_path, monkeypatch):
        """Nothing happens when no package-lock.json is dirty."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "diff --name-only" in " ".join(str(p) for p in cmd):
                return SimpleNamespace(
                    returncode=0,
                    stdout="some_other_file.py\n",
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
        cli_main._restore_package_lockfiles_from_git(["git"], tmp_path)

        checkout_calls = [
            c for c in calls if "checkout" in " ".join(str(p) for p in c)
        ]
        assert len(checkout_calls) == 0

    def test_noop_when_diff_fails(self, tmp_path, monkeypatch):
        """A failing diff does not crash — the function is best-effort."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="error")

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
        cli_main._restore_package_lockfiles_from_git(["git"], tmp_path)

    def test_includes_git_cmd_prefix(self, tmp_path, monkeypatch):
        """The git_cmd list is passed through to subprocess calls."""
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "diff" in str(cmd):
                return SimpleNamespace(
                    returncode=0,
                    stdout="package-lock.json\n",
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
        cli_main._restore_package_lockfiles_from_git(
            ["git", "-c", "windows.appendAtomically=false"], tmp_path
        )

        for cmd in calls:
            assert cmd[0] == "git"
            if len(cmd) > 2 and cmd[1] == "-c":
                assert cmd[2] == "windows.appendAtomically=false"
