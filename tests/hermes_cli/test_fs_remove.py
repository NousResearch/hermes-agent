"""Tests for hermes_cli.fs_remove, rmtree that survives read-only files.

The bug these guard: git writes ``.git/objects/**`` read-only, and on
Windows ``os.unlink`` refuses a read-only file with ``PermissionError:
[WinError 5]``. Any path that removes a git checkout hits it, which is
why ``hermes plugins remove`` fails outright on Windows.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from hermes_cli.fs_remove import rmtree_force


def _make_git_like_tree(root: Path) -> Path:
    """Build a directory whose layout mirrors a real plugin checkout.

    The read-only ``pack-*.idx`` under ``.git/objects/pack/`` is exactly
    what git produces and exactly what plain rmtree chokes on.
    """
    pack = root / ".git" / "objects" / "pack"
    pack.mkdir(parents=True)
    idx = pack / "pack-0123456789abcdef.idx"
    idx.write_bytes(b"idx")
    (pack / "pack-0123456789abcdef.pack").write_bytes(b"pack")
    (root / "plugin.yaml").write_text("name: demo\n", encoding="utf-8")

    for f in pack.iterdir():
        os.chmod(f, stat.S_IREAD)
    return idx


def _restore(root: Path) -> None:
    """Best-effort chmod +w sweep so a failed test can still clean up."""
    for p in root.rglob("*"):
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass


class TestRmtreeForce:
    def test_removes_tree_with_readonly_file(self, tmp_path):
        root = tmp_path / "demo-plugin"
        _make_git_like_tree(root)

        rmtree_force(root)

        assert not root.exists()

    def test_plain_rmtree_fails_on_windows(self, tmp_path):
        """Pin the underlying platform behaviour this module exists for.

        On POSIX the parent directory's write bit governs the unlink, so
        plain rmtree succeeds there and there is nothing to assert.
        """
        root = tmp_path / "demo-plugin"
        _make_git_like_tree(root)

        if sys.platform == "win32":
            with pytest.raises(PermissionError):
                shutil.rmtree(root)
            _restore(root)
            rmtree_force(root)
            assert not root.exists()
        else:
            shutil.rmtree(root)
            assert not root.exists()

    def test_readonly_directory(self, tmp_path):
        """A read-only *directory*, not just a read-only file."""
        root = tmp_path / "tree"
        inner = root / "inner"
        inner.mkdir(parents=True)
        (inner / "file.txt").write_text("x", encoding="utf-8")
        os.chmod(inner, stat.S_IREAD | stat.S_IEXEC)

        try:
            rmtree_force(root)
        finally:
            if root.exists():
                _restore(root)
                shutil.rmtree(root, ignore_errors=True)

        assert not root.exists()

    def test_missing_path_is_a_noop(self, tmp_path):
        rmtree_force(tmp_path / "does-not-exist")

    def test_missing_path_with_ignore_errors(self, tmp_path):
        rmtree_force(tmp_path / "does-not-exist", ignore_errors=True)

    def test_accepts_str_path(self, tmp_path):
        root = tmp_path / "demo-plugin"
        _make_git_like_tree(root)

        rmtree_force(str(root))

        assert not root.exists()

    def test_ignore_errors_swallows_failure(self, tmp_path, monkeypatch):
        root = tmp_path / "tree"
        root.mkdir()
        (root / "f.txt").write_text("x", encoding="utf-8")

        def boom(*a, **kw):
            raise OSError("device busy")

        monkeypatch.setattr(shutil, "rmtree", boom)

        # Without ignore_errors the error propagates...
        with pytest.raises(OSError):
            rmtree_force(root, attempts=1)

        # ...with it, the call is quiet.
        rmtree_force(root, attempts=1, ignore_errors=True)

    def test_non_permission_errors_are_re_raised(self):
        """The handler only rescues PermissionError.

        Anything else means a real problem (a busy device, a bad path) and
        must surface rather than be chmod-and-retried into silence.
        """
        from hermes_cli.fs_remove import _make_writable

        def should_not_run(_path):
            raise AssertionError("retry attempted for a non-permission error")

        with pytest.raises(IsADirectoryError):
            _make_writable(should_not_run, "/some/path", IsADirectoryError("nope"))

    def test_handler_accepts_the_311_exc_info_tuple(self):
        """3.11 passes onerror a sys.exc_info() tuple, 3.12+ passes onexc an
        exception instance. Both have to reach the same code path."""
        from hermes_cli.fs_remove import _make_writable

        with pytest.raises(IsADirectoryError):
            _make_writable(
                lambda _p: None,
                "/some/path",
                (IsADirectoryError, IsADirectoryError("nope"), None),
            )


class TestPluginRemoval:
    """The reported symptom: ``hermes plugins remove <name>`` on Windows."""

    def test_cmd_remove_deletes_git_checkout(self, tmp_path, monkeypatch):
        from hermes_cli import plugins_cmd

        plugins_dir = tmp_path / "plugins"
        target = plugins_dir / "demo-plugin"
        target.mkdir(parents=True)
        _make_git_like_tree(target)

        monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: plugins_dir)
        monkeypatch.setattr(
            plugins_cmd, "_require_installed_plugin", lambda *a, **kw: target
        )
        monkeypatch.setattr(plugins_cmd, "_display_removed", lambda *a, **kw: None)

        try:
            plugins_cmd.cmd_remove("demo-plugin")
        finally:
            if target.exists():
                _restore(target)
                shutil.rmtree(target, ignore_errors=True)

        assert not target.exists()

    def test_dashboard_remove_deletes_git_checkout(self, tmp_path, monkeypatch):
        from hermes_cli import plugins_cmd

        plugins_dir = tmp_path / "plugins"
        target = plugins_dir / "demo-plugin"
        target.mkdir(parents=True)
        _make_git_like_tree(target)

        monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: plugins_dir)
        monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: [])
        monkeypatch.setattr(
            plugins_cmd, "_user_installed_plugin_dir", lambda name: target
        )

        try:
            result = plugins_cmd.dashboard_remove_user_plugin("demo-plugin")
        finally:
            if target.exists():
                _restore(target)
                shutil.rmtree(target, ignore_errors=True)

        assert result == {"ok": True, "name": "demo-plugin"}
        assert not target.exists()
