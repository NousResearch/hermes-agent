"""Unit tests for atomic_io / atomic_io_replace."""

import json
import os
import stat
from pathlib import Path
import pytest

from atomic_io import cross_platform_atomic_writer
from atomic_io_replace import capture_target_metadata, replace_with_retry


def test_cross_platform_atomic_writer_text(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"

    def writer(fp) -> None:
        fp.write("hello")

    cross_platform_atomic_writer(target, writer, binary=False)
    assert target.read_text(encoding="utf-8") == "hello"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits differ on Windows")
def test_preserves_mode_after_replace(tmp_path: Path) -> None:
    target = tmp_path / "mode.txt"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, 0o640)

    def writer(fp) -> None:
        fp.write("new")

    cross_platform_atomic_writer(target, writer, binary=False)
    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="POSIX metadata")
def test_capture_metadata_roundtrip_uid_gid(tmp_path: Path) -> None:
    target = tmp_path / "x"
    target.write_text("z")
    meta = capture_target_metadata(target)
    assert meta.mode is not None
    assert meta.uid is not None and meta.gid is not None


def test_replace_with_retry_succeeds_first_try(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    replace_with_retry(str(a), b)
    assert not a.exists()
    assert b.read_bytes() == b"1"


def test_win_reserved_name_raises_on_windows(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows-only reserved names")
    bad = tmp_path / "CON.txt"

    def writer(fp) -> None:
        fp.write("x")

    with pytest.raises(OSError):
        cross_platform_atomic_writer(bad, writer, binary=False)
