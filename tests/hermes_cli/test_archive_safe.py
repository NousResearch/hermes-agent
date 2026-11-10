"""Tests for the shared tar.gz writer (``hermes_cli.archive_safe.make_targz``).

``make_targz`` backs both ``hermes profile export`` and ``hermes kanban
export``. The contract pinned down here: a failure partway through writing
the archive (disk full, permission loss, interruption) must never destroy a
pre-existing file at the destination path — the same failure-atomicity
guarantee the desktop gateway-download path already has.
"""

from __future__ import annotations

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
