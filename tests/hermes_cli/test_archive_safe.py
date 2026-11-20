"""Tests for the shared tar.gz writer (``hermes_cli.archive_safe.make_targz``).

``make_targz`` backs both ``hermes profile export`` and ``hermes kanban
export``. The contract pinned down here: a failure partway through writing
the archive (disk full, permission loss, interruption) must never destroy a
pre-existing file at the destination path — the same failure-atomicity
guarantee the desktop gateway-download path already has.
"""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli.archive_safe import make_targz


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


@pytest.mark.platforms("posix")  # directory fsync is a no-op on Windows
def test_make_targz_syncs_complete_archive_before_publishing(tmp_path, monkeypatch):
    """A crash right after the rename must not leave a zero-tailed export under the final name."""
    _stage_source(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    events: list[tuple] = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd):
        st = os.fstat(fd)
        events.append(("fsync", st.st_ino, st.st_size))
        return real_fsync(fd)

    def spy_replace(src, dst):
        events.append(("replace", os.stat(src).st_ino))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    archive = Path(make_targz(str(out_dir / "export"), str(tmp_path), "src"))

    st = archive.stat()
    replace_at = events.index(("replace", st.st_ino))
    assert ("fsync", st.st_ino, st.st_size) in events[:replace_at]
    dir_ino = out_dir.stat().st_ino
    assert any(e[0] == "fsync" and e[1] == dir_ino for e in events[replace_at:])
    parent_ino = tmp_path.stat().st_ino  # out/ may be new: its own entry must be durable too
    assert any(e[0] == "fsync" and e[1] == parent_ino for e in events[replace_at:])


def test_make_targz_keeps_previous_archive_when_sync_fails(tmp_path, monkeypatch):
    _stage_source(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    archive = out_dir / "export.tar.gz"
    archive.write_bytes(b"previous export")

    def failing_fsync(fd):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        make_targz(str(out_dir / "export"), str(tmp_path), "src")

    assert archive.read_bytes() == b"previous export"
    assert list(out_dir.iterdir()) == [archive]
