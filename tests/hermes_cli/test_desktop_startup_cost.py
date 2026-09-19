"""Startup regressions exercised against real source trees on the current host."""

import argparse
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop as desktop


def _tree(tmp_path):
    root = tmp_path / "checkout"
    app = root / "apps" / "desktop"
    app.mkdir(parents=True)
    (root / ".gitignore").write_text("dist/\nrelease/\nnode_modules/\n")
    (root / "package.json").write_text("{}")
    (app / "package.json").write_text("{}")
    source = app / "app.ts"
    source.write_text("export const x = 1")
    return root, app, source


@pytest.mark.parametrize("tree", ["desktop", "shared"])
def test_desktop_hash_prunes_ignored_build_directories(tmp_path, tree):
    root, _, _ = _tree(tmp_path)
    app = root / "apps" / tree
    app.mkdir(exist_ok=True)
    ignored = app / "dist"
    ignored.mkdir()
    (ignored / "bundle.js").write_text("built output")
    expected = desktop._compute_desktop_content_hash(root)
    original_scandir = os.scandir

    def scan(directory):
        assert Path(directory) != ignored, "ignored build output must be pruned before traversal"
        return original_scandir(directory)

    (ignored / "bundle.js").write_text("different build output")
    with patch("os.scandir", side_effect=scan):
        assert desktop._compute_desktop_content_hash(root) == expected


def test_desktop_hash_invalidates_for_edits_even_with_restored_mtime(tmp_path):
    root, app, source = _tree(tmp_path)
    original = desktop._compute_desktop_content_hash(root)
    stamp = source.stat()
    source.write_text("export const x = 2")  # Same length; editors can preserve mtime.
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert desktop._compute_desktop_content_hash(root) != original
    source.write_text("export const x = 1")
    assert desktop._compute_desktop_content_hash(root) == original
    added = app / "new.ts"
    added.write_text("new module")
    assert desktop._compute_desktop_content_hash(root) != original
    added.unlink()
    assert desktop._compute_desktop_content_hash(root) == original
    (root / "package.json").write_text('{"changed": true}')
    assert desktop._compute_desktop_content_hash(root) != original


@pytest.mark.windows_only
def test_desktop_hash_detects_memory_mapped_edits(tmp_path):
    import mmap

    root, _, source = _tree(tmp_path)
    original = desktop._compute_desktop_content_hash(root)
    with source.open("r+b") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE) as view:
            view[-1:] = b"2"
            view.flush()
    assert source.read_bytes().endswith(b"2")
    assert desktop._compute_desktop_content_hash(root) != original


@pytest.mark.windows_only
def test_desktop_hash_recovers_after_a_temporary_read_lock(tmp_path):
    import msvcrt

    root, _, source = _tree(tmp_path)
    source.write_bytes(b"")
    original = desktop._compute_desktop_content_hash(root)
    payload = b"export const x = 2"
    source.write_bytes(payload)
    with source.open("r+b") as locked:
        msvcrt.locking(locked.fileno(), msvcrt.LK_NBLCK, len(payload))
        try:
            with pytest.raises(OSError):
                source.read_bytes()
            desktop._compute_desktop_content_hash(root)
        finally:
            locked.seek(0)
            msvcrt.locking(locked.fileno(), msvcrt.LK_UNLCK, len(payload))
    assert desktop._compute_desktop_content_hash(root) != original


def test_current_packaged_launch_does_not_require_npm(tmp_path, monkeypatch):
    root, app, _ = _tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    executable = app / "Hermes.exe"
    args = argparse.Namespace(source=False, skip_build=False, force_build=False, build_only=False)
    with patch.object(desktop, "_desktop_launch_env", return_value=({}, [])), \
         patch.object(desktop, "_desktop_packaged_executable", return_value=executable), \
         patch.object(desktop, "_desktop_build_needed", return_value=False), \
         patch.object(desktop, "_register_linux_desktop_entry"), \
         patch.object(desktop, "_packaged_desktop_launch_command", return_value=[str(executable)]), \
         patch("hermes_cli.main_install_repair._resolve_node_runtime_npm", return_value=None) as npm, \
         patch.object(desktop.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run, \
         pytest.raises(SystemExit) as exited:
        desktop.cmd_gui(args)
    assert exited.value.code == 0
    npm.assert_not_called()
    run.assert_called_once()
