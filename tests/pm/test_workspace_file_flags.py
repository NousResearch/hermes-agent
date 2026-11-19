"""Workspace snapshots copy build inputs without BSD file flags (#157508).

On macOS ``shutil.copy2`` and ``shutil.copytree`` replay ``st_flags`` through ``os.chflags``
(inside ``shutil.copystat``). A checkout locked with ``chflags -R schg`` then refused every
plugin enable with EPERM, for files and for directories alike.
"""
import errno
import os
import shutil
import stat

import pytest

from pm import workspace


def _refuse_flag_replay(monkeypatch):
    """Every stdlib metadata replay fails the way chflags(schg) does for a non-root user."""
    def copystat(src, dst, *, follow_symlinks=True):
        raise PermissionError(errno.EPERM, "Operation not permitted", os.fspath(dst))

    monkeypatch.setattr(shutil, "copystat", copystat)


def _core(tmp_path):
    source = tmp_path / "core"
    (source / "agent").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        '[project]\nname = "core"\nversion = "1"\n'
        '[tool.setuptools.packages.find]\ninclude = ["agent"]\n',
        encoding="utf-8",
    )
    (source / "agent" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    script = source / "agent" / "tool.py"
    script.write_text("print('x')\n", encoding="utf-8")
    script.chmod(0o755)
    os.utime(script, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    return source


def test_core_inputs_copy_without_replaying_flags(tmp_path, monkeypatch):
    source = _core(tmp_path)
    _refuse_flag_replay(monkeypatch)

    destination = tmp_path / "snapshot"
    destination.mkdir()
    workspace._copy_core_inputs(source, destination)

    copied = destination / "agent" / "tool.py"
    assert copied.read_text(encoding="utf-8") == "print('x')\n"
    assert (destination / "pyproject.toml").is_file()
    # Contents, mode and times still travel, as with copy2.
    assert stat.S_IMODE(copied.stat().st_mode) == stat.S_IMODE((source / "agent" / "tool.py").stat().st_mode)
    assert copied.stat().st_mtime_ns == 1_600_000_000_000_000_000


def test_member_tree_copy_keeps_links_and_ignores_without_replaying_flags(tmp_path, monkeypatch):
    plugin = tmp_path / "plugin"
    (plugin / "pkg" / "nested").mkdir(parents=True)
    (plugin / "pkg" / "nested" / "data.txt").write_text("data", encoding="utf-8")
    (plugin / "node_modules").mkdir()
    (plugin / "node_modules" / "junk.js").write_text("junk", encoding="utf-8")
    try:
        (plugin / "link").symlink_to("pkg/nested/data.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    _refuse_flag_replay(monkeypatch)

    member = tmp_path / "root" / "plugin-sources" / "plugin"
    workspace._copy_tree(plugin, member, symlinks=True, ignore=workspace._member_ignored)

    assert (member / "pkg" / "nested" / "data.txt").read_text(encoding="utf-8") == "data"
    assert (member / "link").is_symlink()
    assert os.readlink(member / "link") == "pkg/nested/data.txt"
    assert not (member / "node_modules").exists()


def test_tree_copy_fills_a_read_only_directory_before_applying_its_mode(tmp_path):
    source = tmp_path / "locked"
    source.mkdir()
    (source / "a.txt").write_text("a", encoding="utf-8")
    source.chmod(0o555)
    try:
        copied = workspace._copy_tree(source, tmp_path / "copy")
        assert (copied / "a.txt").read_text(encoding="utf-8") == "a"
        assert stat.S_IMODE(copied.stat().st_mode) == 0o555
    finally:
        source.chmod(0o755)
        if (tmp_path / "copy").exists():
            (tmp_path / "copy").chmod(0o755)


@pytest.mark.platforms("macos")
def test_flagged_checkout_snapshot_carries_no_flags(tmp_path):
    """The real macOS path: uchg (settable by the owner, unlike schg) on files and directories."""
    source = _core(tmp_path)
    locked = [source / "agent" / "tool.py", source / "agent"]
    for path in locked:
        os.chflags(path, stat.UF_IMMUTABLE)
    destination = tmp_path / "snapshot"
    destination.mkdir()
    try:
        workspace._copy_core_inputs(source, destination)
        for copied in (destination / "agent" / "tool.py", destination / "agent"):
            assert copied.stat().st_flags & stat.UF_IMMUTABLE == 0
        (destination / "agent" / "tool.py").write_text("writable", encoding="utf-8")
    finally:
        for path in locked:
            os.chflags(path, 0)
