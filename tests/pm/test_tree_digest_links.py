"""Realized digests bind link text without reading the linked directory."""
import logging
import os
from pathlib import Path

import pytest

from pm.install import _entry_verified
from pm.store import Store, tree_digest


def test_digest_without_pathlib_junction_api(tmp_path, monkeypatch):
    # Source updates reach the store before replacing a pre-3.12 interpreter.
    monkeypatch.delattr(Path, "is_junction", raising=False)
    root = tmp_path / "tree"
    nested = root / "lib"
    nested.mkdir(parents=True)
    payload = nested / "payload"
    payload.write_bytes(b"before")
    first = tree_digest(root)
    assert tree_digest(root) == first
    payload.write_bytes(b"after")
    assert tree_digest(root) != first


@pytest.mark.platforms("windows", "posix")
def test_directory_links_bind_only_their_target_text(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "payload").write_bytes(b"unchanged")
    left, right = tmp_path / "left", tmp_path / "right"
    for target in (left, right):
        target.mkdir()
        (target / "content").write_bytes(b"same bytes")
    plain = tree_digest(root)
    link = root / "directory"
    try:
        link.symlink_to(Path("..") / "left", target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("directory symlinks require Windows Developer Mode or privilege")
        raise
    first = tree_digest(root)
    assert first != plain
    (left / "content").write_bytes(b"changed outside the tree")
    assert tree_digest(root) == first
    link.unlink()
    link.symlink_to(Path("..") / "right", target_is_directory=True)
    assert tree_digest(root) != first
    link.unlink()
    link.symlink_to(".", target_is_directory=True)
    cycle = tree_digest(root)
    assert tree_digest(root) == cycle
    assert cycle not in (plain, first)


@pytest.mark.platforms("windows")
def test_junctions_bind_target_text_without_walking_outside(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.delattr(Path, "is_junction", raising=False)
    root = tmp_path / "tree"
    root.mkdir()
    left, right = tmp_path / "left", tmp_path / "right"
    for target in (left, right):
        target.mkdir()
        (target / "content").write_bytes(b"same bytes")
    junction = root / "junction"
    command = str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe")

    def point_at(target):
        result = subprocess.run(
            [command, "/d", "/c", "mklink", "/J", str(junction), str(target)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    point_at(left)
    try:
        first = tree_digest(root)
        (left / "content").write_bytes(b"changed outside the tree")
        assert tree_digest(root) == first
        junction.rmdir()
        point_at(right)
        assert tree_digest(root) != first
    finally:
        junction.rmdir()
    assert (left / "content").read_bytes() == b"changed outside the tree"
    assert (right / "content").read_bytes() == b"same bytes"


def _raise(error: OSError) -> None:
    raise error


class _VerifiedPackage:
    name = "pkg"

    def verify(self, entry, target):
        return None


@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores mode bits")
def test_unreadable_subtree_digests_as_absent_by_default_and_raises_on_demand(tmp_path):
    root = tmp_path / "tree"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (locked / "payload").write_bytes(b"secret")
    (root / "unlocked").write_bytes(b"plain")
    locked.chmod(0o000)
    try:
        without = tmp_path / "without"
        without.mkdir()
        (without / "unlocked").write_bytes(b"plain")
        # Historic default: the unreadable subtree contributes nothing.
        assert tree_digest(root) == tree_digest(without)
        with pytest.raises(PermissionError):
            tree_digest(root, onerror=_raise)
    finally:
        locked.chmod(0o755)


@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores mode bits")
def test_unreadable_entry_stays_installed_instead_of_reinstalling(tmp_path, caplog):
    entry = tmp_path / "store" / "pkg-entry"
    (entry / "locked").mkdir(parents=True)
    (entry / "locked" / "payload").write_bytes(b"secret")
    fact = {"entry": "pkg-entry", "digest": tree_digest(entry)}
    store = Store(tmp_path / "store")
    (entry / "locked").chmod(0o000)
    try:
        with caplog.at_level(logging.WARNING):
            assert _entry_verified(_VerifiedPackage(), fact, store, "x86_64-unknown-linux")
        assert "instead of reinstalling" in caplog.text
        # Readable bytes still verify normally: drift keeps triggering a repair.
        (entry / "locked").chmod(0o755)
        (entry / "locked" / "payload").write_bytes(b"tampered")
        assert not _entry_verified(_VerifiedPackage(), fact, store, "x86_64-unknown-linux")
    finally:
        (entry / "locked").chmod(0o755)
