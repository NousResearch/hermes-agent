"""``_remove_entry`` retires a Windows junction instead of aborting the update.

Regression: the store keeps ``.previous-<entry>`` restore points, and PM publishes
junction entries -- a bundled tool linked to a host install is exactly the shape
``pm.build_operations._copy_links`` permits. With only ``is_symlink()`` in the
guard a junction took the ``shutil.rmtree`` branch, which refuses it with
"Cannot call rmtree on a symbolic link"; ``pm.ensure("git")`` then turned that
into "Source update completion failed" and ``hermes update`` exited 1 with the
desktop bundle never rebuilt.
"""
import os
import subprocess
from pathlib import Path

import pytest

from pm.filesystem import is_junction
from pm.install import _remove_entry
from pm.store import Store


def _mklink(link: Path, target: Path) -> None:
    """Create a real NTFS junction, or skip where that is not available."""
    command = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe")
    link.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [command, "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
    )
    if result.returncode != 0:
        pytest.skip("cannot create a junction: " + (result.stdout + result.stderr).strip())


@pytest.mark.platforms("windows")
def test_remove_entry_retires_junction_without_touching_target(tmp_path, monkeypatch):
    # The helper is also reached before a pre-3.12 interpreter is replaced, so it
    # must not lean on Path.is_junction for the predicate.
    monkeypatch.delattr(Path, "is_junction", raising=False)
    target = tmp_path / "host-tool"
    target.mkdir()
    (target / "tool.exe").write_bytes(b"MZ-real-tool")

    store = Store(tmp_path / "store")
    store.root.mkdir()
    entry = store.entry(".previous-tool-1.0-win32-x64")
    _mklink(entry, target)
    assert entry.is_symlink() is False  # the blind spot this regression pins
    assert entry.is_dir() is True       # ... which sent the old guard to rmtree
    try:
        _remove_entry(store, entry.name)
        assert not os.path.lexists(entry)
        assert (target / "tool.exe").read_bytes() == b"MZ-real-tool"
    finally:
        if os.path.lexists(entry):
            entry.rmdir()


@pytest.mark.platforms("windows")
def test_remove_entry_retires_dangling_junction(tmp_path):
    store = Store(tmp_path / "store")
    store.root.mkdir()
    entry = store.entry(".previous-tool-gone")
    _mklink(entry, tmp_path / "missing-target")
    try:
        _remove_entry(store, entry.name)
        assert not os.path.lexists(entry)
    finally:
        if os.path.lexists(entry):
            entry.rmdir()


def test_remove_entry_still_recurses_a_plain_directory(tmp_path):
    store = Store(tmp_path / "store")
    store.root.mkdir()
    entry = store.entry("fetch-deadbeef")
    (entry / "nested").mkdir(parents=True)
    (entry / "nested" / "payload").write_bytes(b"x" * 32)
    _remove_entry(store, entry.name)
    assert not entry.exists()


def test_remove_entry_missing_entry_is_a_noop(tmp_path):
    store = Store(tmp_path / "store")
    store.root.mkdir()
    _remove_entry(store, "fetch-absent")
    assert not store.entry("fetch-absent").exists()


def test_is_junction_is_total_for_a_missing_path(tmp_path):
    """A removal path runs after the entry may already be gone.

    ``is_junction`` reads the reparse tag through ``lstat``, which raises for a path that is
    not there -- use it as a plain predicate and the whole rollback aborts (regression:
    ``agent.curator_backup._remove_entry`` on an entry a partial restore had already dropped).
    """
    assert is_junction(tmp_path / "absent") is False


@pytest.mark.platforms("windows")
def test_is_junction_reports_a_dangling_junction(tmp_path):
    """``Path.exists()`` is False for a dangling junction, so guarding with it misses one."""
    target = tmp_path / "host-tool"
    target.mkdir()
    link = tmp_path / "dangling"
    _mklink(link, target)
    target.rmdir()
    try:
        assert (link / "nothing").exists() is False
        assert is_junction(link) is True
    finally:
        if os.path.lexists(link):
            link.rmdir()
