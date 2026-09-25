"""Realized digests bind link text without reading the linked directory."""
import os
from pathlib import Path

import pytest

from pm.store import tree_digest


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
