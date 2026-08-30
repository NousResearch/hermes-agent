"""Tests for the shared tar.gz writer (``hermes_cli.archive_safe.make_targz``).

``make_targz`` backs both ``hermes profile export`` and ``hermes kanban
export``. The contract pinned down here: a failure partway through writing
the archive (disk full, permission loss, interruption) must never destroy a
pre-existing file at the destination path — the same failure-atomicity
guarantee the desktop gateway-download path already has.
"""

from __future__ import annotations
import io
import os
import stat
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli.archive_safe import (
    archive_root_dirs,
    copy_regular_files,
    make_targz,
    normalize_archive_parts,
    safe_extract_targz,
)


def _stage_source(tmp_path: Path) -> None:
    payload = tmp_path / "src" / "inner"
    payload.mkdir(parents=True)
    (payload / "file.txt").write_text("hello")


def test_make_targz_preserves_existing_file_on_mid_write_failure(tmp_path, monkeypatch):
    _stage_source(tmp_path)
    base = str(tmp_path / "out")
    archive_path = Path(f"{base}.tar.gz")
    sentinel = b"PRE-EXISTING ARCHIVE THAT MUST SURVIVE A FAILED RE-EXPORT"
    archive_path.write_bytes(sentinel)

    def _boom(self, *a, **k):
        raise RuntimeError("simulated failure mid-add (disk full / permission loss)")

    monkeypatch.setattr(tarfile.TarFile, "add", _boom)

    with pytest.raises(RuntimeError):
        make_targz(base, str(tmp_path), "src")

    assert archive_path.read_bytes() == sentinel, (
        "a mid-write failure must not truncate/replace a pre-existing archive"
    )
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".archive_")]
    assert leftovers == [], f"temp file was not cleaned up on failure: {leftovers}"


def test_make_targz_round_trips_content(tmp_path):
    _stage_source(tmp_path)
    base = str(tmp_path / "out")

    result = make_targz(base, str(tmp_path), "src")

    assert result == f"{base}.tar.gz"
    with tarfile.open(result, "r:gz") as tf:
        names = sorted(m.name for m in tf.getmembers())
        assert "src/inner/file.txt" in names
        member = tf.extractfile("src/inner/file.txt")
        assert member is not None
        assert member.read() == b"hello"


def test_make_targz_overwrites_existing_file_on_success(tmp_path):
    _stage_source(tmp_path)
    base = str(tmp_path / "out")
    archive_path = Path(f"{base}.tar.gz")
    archive_path.write_bytes(b"stale archive from a previous export")

    make_targz(base, str(tmp_path), "src")

    with tarfile.open(archive_path, "r:gz") as tf:
        assert "src/inner/file.txt" in {m.name for m in tf.getmembers()}


class TestNormalizeArchiveParts:
    def test_simple_file(self):
        assert normalize_archive_parts("foo/bar.txt") == ["foo", "bar.txt"]

    def test_single_component(self):
        assert normalize_archive_parts("README.md") == ["README.md"]

    def test_strips_leading_dot_slash(self):
        assert normalize_archive_parts("./data/file.txt") == ["data", "file.txt"]

    def test_backslash_folded_to_slash(self):
        # Windows-authored archive: backslash must not smuggle a separator
        assert normalize_archive_parts("foo\\bar\\baz.txt") == ["foo", "bar", "baz.txt"]

    def test_rejects_absolute_posix(self):
        with pytest.raises(ValueError, match="Unsafe"):
            normalize_archive_parts("/etc/passwd")

    def test_rejects_dotdot_component(self):
        with pytest.raises(ValueError, match="Unsafe"):
            normalize_archive_parts("foo/../../../etc/passwd")

    def test_rejects_windows_absolute_drive_letter(self):
        with pytest.raises(ValueError, match="Unsafe"):
            normalize_archive_parts("C:\\Windows\\System32\\evil.exe")

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="Unsafe"):
            normalize_archive_parts("")

    def test_rejects_bare_dotdot(self):
        with pytest.raises(ValueError, match="Unsafe"):
            normalize_archive_parts("..")

    def test_rejects_dotdot_at_start(self):
        with pytest.raises(ValueError, match="Unsafe"):
            normalize_archive_parts("../escape")

    def test_deep_nested_path(self):
        parts = normalize_archive_parts("a/b/c/d/e.json")
        assert parts == ["a", "b", "c", "d", "e.json"]

    def test_mixed_backslash_and_slash(self):
        parts = normalize_archive_parts("foo\\bar/baz.txt")
        assert parts == ["foo", "bar", "baz.txt"]




def _make_archive(tmp_path: Path, members: list[tuple[str, bytes]]) -> Path:
    """Build a tar.gz with the given (name, content) pairs."""
    arc = tmp_path / "test.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return arc




class TestSafeExtractTargz:
    def test_extracts_normal_files(self, tmp_path):
        arc = _make_archive(tmp_path, [
            ("mydir/hello.txt", b"hello"),
            ("mydir/nested/world.txt", b"world"),
        ])
        dest = tmp_path / "out"
        dest.mkdir()
        safe_extract_targz(arc, dest)
        assert (dest / "mydir" / "hello.txt").read_bytes() == b"hello"
        assert (dest / "mydir" / "nested" / "world.txt").read_bytes() == b"world"

    def test_rejects_path_traversal_member(self, tmp_path):
        arc = tmp_path / "evil.tar.gz"
        with tarfile.open(arc, "w:gz") as tf:
            info = tarfile.TarInfo(name="../escape.txt")
            info.size = 4
            tf.addfile(info, io.BytesIO(b"evil"))
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(ValueError, match="Unsafe"):
            safe_extract_targz(arc, dest)

    def test_rejects_symlink_member(self, tmp_path):
        arc = tmp_path / "symlink.tar.gz"
        with tarfile.open(arc, "w:gz") as tf:
            info = tarfile.TarInfo(name="link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(ValueError, match="Unsupported"):
            safe_extract_targz(arc, dest)

    @pytest.mark.platforms("posix")
    def test_preserves_file_mode(self, tmp_path):
        arc = tmp_path / "mode.tar.gz"
        with tarfile.open(arc, "w:gz") as tf:
            info = tarfile.TarInfo(name="dir/script.sh")
            info.size = 7
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(b"#!/bin/"))
        dest = tmp_path / "out"
        dest.mkdir()
        safe_extract_targz(arc, dest)
        extracted = dest / "dir" / "script.sh"
        assert extracted.exists()
        # mode bits set (mask to low 9 bits)
        assert (extracted.stat().st_mode & 0o777) == 0o755




class TestArchiveRootDirs:
    def test_single_root_directory(self, tmp_path):
        arc = _make_archive(tmp_path, [
            ("myprofile/config.yaml", b""),
            ("myprofile/sessions/db.sqlite", b""),
        ])
        roots = archive_root_dirs(arc)
        assert roots == {"myprofile"}

    def test_multiple_root_directories(self, tmp_path):
        arc = _make_archive(tmp_path, [
            ("a/file.txt", b""),
            ("b/file.txt", b""),
        ])
        roots = archive_root_dirs(arc)
        assert roots == {"a", "b"}

    def test_single_file_no_parent(self, tmp_path):
        # A lone top-level file has no root directory in len(parts) > 1 sense
        arc = _make_archive(tmp_path, [("orphan.txt", b"")])
        # orphan.txt has parts=["orphan.txt"], len == 1 and not a dir → not in set
        roots = archive_root_dirs(arc)
        assert "orphan.txt" not in roots




class TestCopyRegularFiles:
    def test_copies_files_recursively(self, tmp_path):
        src = tmp_path / "src"
        (src / "sub").mkdir(parents=True)
        (src / "a.txt").write_bytes(b"aaa")
        (src / "sub" / "b.txt").write_bytes(b"bbb")
        dst = tmp_path / "dst"
        dst.mkdir()
        count = copy_regular_files(src, dst)
        assert count == 2
        assert (dst / "a.txt").read_bytes() == b"aaa"
        assert (dst / "sub" / "b.txt").read_bytes() == b"bbb"

    def test_skips_symlinks(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        real = src / "real.txt"
        real.write_bytes(b"real")
        link = src / "link.txt"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        dst = tmp_path / "dst"
        dst.mkdir()
        count = copy_regular_files(src, dst)
        assert count == 1  # only real.txt
        assert not (dst / "link.txt").exists()

    def test_missing_src_returns_zero(self, tmp_path):
        dst = tmp_path / "dst"
        dst.mkdir()
        assert copy_regular_files(tmp_path / "nonexistent", dst) == 0

    def test_empty_src_returns_zero(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        assert copy_regular_files(src, dst) == 0




def test_make_targz_round_trip(tmp_path):
    src_dir = tmp_path / "mydata"
    src_dir.mkdir()
    (src_dir / "file.txt").write_bytes(b"content")
    arc_base = str(tmp_path / "archive")
    arc_path = make_targz(arc_base, str(tmp_path), "mydata")
    assert arc_path.endswith(".tar.gz")
    assert Path(arc_path).is_file()
    # Must be a valid tar.gz that our extractor accepts
    dest = tmp_path / "extracted"
    dest.mkdir()
    safe_extract_targz(Path(arc_path), dest)
    assert (dest / "mydata" / "file.txt").read_bytes() == b"content"
