"""What a dependency generation copies of each plugin member (regression for #154442).

A profile clone copies ``plugins/``, so one plugin sits at one path per profile. Each path was a
member of its own, and adding a profile rebuilt the environment with one more full copy. A member
symlinked from a development checkout carried the checkout's git-ignored data into every
generation, and on APFS each copy wrote new blocks.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import pm.workspace as ws
import pm.workspace_members as ws_members


def _plugin(path: Path, dependency: str = "left-pad-py>=1,<2") -> Path:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text(
        f'[project]\nname = "camofox"\nversion = "1.0"\ndependencies = ["{dependency}"]\n', encoding="utf-8")
    (path / "camofox.py").write_text("VALUE = 1\n", encoding="utf-8")
    return path


def _core(path: Path) -> Path:
    path.mkdir()
    (path / "pyproject.toml").write_text('[project]\nname = "core"\nversion = "1"\nrequires-python = ">=3.11"\n',
                                         encoding="utf-8")
    return path


def _members(root: Path) -> list[str]:
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return document["tool"]["uv"]["workspace"]["members"]


def test_byte_identical_copies_in_two_homes_are_one_member(tmp_path):
    first = _plugin(tmp_path / "home" / "plugins" / "camofox")
    clone = _plugin(tmp_path / "home" / "profiles" / "bot" / "plugins" / "camofox")

    assert ws.members_stamp([first, clone]) == ws.members_stamp([first])
    root = tmp_path / "generation"
    ws._generate_pyproject([first, clone], root, source=_core(tmp_path / "core"), python=Path(sys.executable))
    assert len(_members(root)) == 1

    (clone / "pyproject.toml").write_text((clone / "pyproject.toml").read_text().replace(">=1,<2", ">=1.1,<2"))
    assert ws.members_stamp([first, clone]) != ws.members_stamp([first])


def test_checkout_member_leaves_out_what_git_ignores(tmp_path):
    plugin = _plugin(tmp_path / "checkout")
    (plugin / ".gitignore").write_text("media/\n", encoding="utf-8")
    (plugin / "media").mkdir()
    (plugin / "media" / "render.bin").write_bytes(b"render")
    (plugin / "NOTES.md").write_text("untracked, not ignored\n", encoding="utf-8")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "-q"], cwd=plugin, env=env, check=True, timeout=60)
    subprocess.run(["git", "add", "pyproject.toml", "camofox.py", ".gitignore"], cwd=plugin, env=env, check=True,
                   timeout=60)
    stamp = ws.members_stamp([plugin])

    (plugin / "media" / "render.bin").write_bytes(b"another render")
    assert ws.members_stamp([plugin]) == stamp
    member = ws._workspace_member(plugin, tmp_path / "generation", identity=plugin)
    assert (member / "camofox.py").is_file() and (member / "NOTES.md").is_file()
    assert not (member / "media").exists()
    (plugin / "camofox.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert ws.members_stamp([plugin]) != stamp


def _device_offset(path: Path) -> int:
    """Where the first byte of *path* lies on the device (``fcntl(F_LOG2PHYS_EXT)``)."""
    import fcntl

    descriptor = os.open(path, os.O_RDONLY)
    try:
        answer = fcntl.fcntl(descriptor, 65, struct.pack("=Iqq", 0, 1 << 20, 0))  # F_LOG2PHYS_EXT
    finally:
        os.close(descriptor)
    return struct.unpack("=Iqq", answer)[2]


_FS_IOC_FIEMAP = 0xC020660B  # _IOWR('f', 11, struct fiemap), linux/fs.h
_FIEMAP_FLAG_SYNC = 0x1
_FIEMAP_EXTENT_SHARED = 0x2000


def _first_extent(path: Path) -> tuple[int, int]:
    """The physical offset and the flags of the first extent of *path* (``ioctl(FS_IOC_FIEMAP)``)."""
    import fcntl

    header = struct.pack("=QQIIII", 0, 1 << 20, _FIEMAP_FLAG_SYNC, 0, 1, 0)  # room for one extent
    with open(path, "rb") as reader:
        answer = fcntl.ioctl(reader.fileno(), _FS_IOC_FIEMAP, header + bytes(56))
    _, physical, _, _, _, flags = struct.unpack_from("=QQQQQI", answer, 32)
    return physical, flags


@pytest.mark.platforms("linux")
def test_member_snapshot_shares_its_blocks_on_btrfs_and_xfs(tmp_path):
    """Run with ``--basetemp`` on a Btrfs or an XFS (reflink=1) mount; ext4 and tmpfs get a copy."""
    plugin = _plugin(tmp_path / "plugin")
    (plugin / "model.bin").write_bytes(os.urandom(1 << 20))

    member = ws._workspace_member(plugin, tmp_path / "generation", identity=plugin)
    assert (member / "model.bin").read_bytes() == (plugin / "model.bin").read_bytes()
    probe = tmp_path / "probe"
    if not ws_members._ficlone(plugin / "model.bin", probe):
        pytest.skip("the filesystem of tmp_path has no reflinks; the snapshot is a copy")
    source, snapshot = _first_extent(plugin / "model.bin"), _first_extent(member / "model.bin")
    assert snapshot[0] == source[0]
    assert snapshot[1] & _FIEMAP_EXTENT_SHARED


@pytest.mark.platforms("macos")
def test_member_snapshot_shares_its_blocks_on_apfs(tmp_path):
    plugin = _plugin(tmp_path / "plugin")
    (plugin / "model.bin").write_bytes(os.urandom(1 << 20))

    member = ws._workspace_member(plugin, tmp_path / "generation", identity=plugin)
    assert (member / "model.bin").read_bytes() == (plugin / "model.bin").read_bytes()
    assert _device_offset(member / "model.bin") == _device_offset(plugin / "model.bin")
