"""PM's shared Node unpacker needs no host xz executable (#11197).

The installer/runtime shell selectors were retired. The exact pinned archive
is extracted by Python's tarfile/lzma instead of selecting unpinned gzip bytes.
"""
from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path

import pytest

from pm.packages import Nodejs
from pm.store import current_target


@pytest.mark.platforms("linux", "macos")
def test_node_archive_unpacks_without_host_xz(tmp_path: Path, monkeypatch) -> None:
    package = Nodejs()
    target = current_target()
    version = "26.7.0"
    archive = tmp_path / Path(package.fetch_url(version, target)).name
    payload = b"node fixture bytes\n"
    with tarfile.open(archive, "w:xz") as writer:
        member = tarfile.TarInfo(f"node-v{version}/{package.binary_rel['posix']}")
        member.size = len(payload)
        member.mode = 0o755
        writer.addfile(member, io.BytesIO(payload))

    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    assert shutil.which("xz") is None
    staged = tmp_path / "staged"
    package.unpack(archive, staged, target)
    assert (staged / f"node-v{version}" / "bin/node").read_bytes() == payload