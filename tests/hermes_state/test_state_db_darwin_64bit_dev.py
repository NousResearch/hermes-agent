"""Regression for #126785: 64-bit st_dev in Darwin holder detection.

On macOS ``os.stat().st_dev`` is a 64-bit ``dev_t``; APFS synthetic device
numbers can exceed 2**32. ``_iter_darwin_fd_targets`` used to unpack
``vst_dev`` with ``'<I'`` (32-bit), truncating the device number so
``count_db_holders`` never matched and falsely returned 0.

These tests fake libproc (so they run on every platform) and cover:

- offset alignment: the 8-byte dev field at ``_DARWIN_FD_DEV_OFFSET`` abuts
  ``vst_ino`` without overlap;
- record parsing preserves full 64-bit dev values;
- ``count_db_holders`` matches a holder whose dev exceeds 2**32;
- devices sharing only the low 32 bits are still distinguished (guards
  against a masking "fix" that truncates both sides).
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys

import pytest

import hermes_state_dbfile as dbfile
from hermes_state_dbfile import (
    _DARWIN_FD_DEV_OFFSET,
    _DARWIN_FD_INO_OFFSET,
    _DARWIN_FD_PATH_OFFSET,
    _DARWIN_FD_RECORD_SIZE,
    _iter_darwin_fd_targets,
    count_db_holders,
)

# Representative 64-bit device numbers: just over the 32-bit boundary, an
# APFS-synthetic-like value, and a large high-bit value.
LARGE_DEVS = [
    (1 << 32) + 1,
    (1 << 32) + 0x12345,
    0x100000001,
    (1 << 33) + 7,
    (1 << 63) - 1,
]


def test_darwin_dev_field_is_64bit_aligned():
    """The dev slot must fit a full 64-bit dev_t without overlapping vst_ino."""
    assert _DARWIN_FD_DEV_OFFSET + 8 == _DARWIN_FD_INO_OFFSET
    assert _DARWIN_FD_INO_OFFSET + 8 <= _DARWIN_FD_PATH_OFFSET
    assert _DARWIN_FD_PATH_OFFSET < _DARWIN_FD_RECORD_SIZE


class _FakeLibproc:
    """Minimal libproc double serving fixed (pid, fd) -> (dev, ino, path)."""

    def __init__(self, entries):
        # entries: dict[int, list[tuple[int, int, int, str]]]
        self._entries = entries

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        fds = self._entries.get(pid, [])
        payload = b"".join(struct.pack("<iI", fd, 1) for fd, _dev, _ino, _path in fds)
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        for entry_fd, dev, ino, path in self._entries.get(pid, []):
            if entry_fd != fd:
                continue
            blob = bytearray(_DARWIN_FD_RECORD_SIZE)
            struct.pack_into("<Q", blob, _DARWIN_FD_DEV_OFFSET, dev)
            struct.pack_into("<Q", blob, _DARWIN_FD_INO_OFFSET, ino)
            encoded = path.encode("utf-8")
            blob[_DARWIN_FD_PATH_OFFSET:_DARWIN_FD_PATH_OFFSET + len(encoded)] = encoded
            ctypes.memmove(record, bytes(blob), len(blob))
            return 1
        return 0


def _install_libproc(monkeypatch, entries):
    lib = _FakeLibproc(entries)
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: list(entries))
    return lib


@pytest.mark.parametrize("dev", LARGE_DEVS)
def test_iter_darwin_fd_targets_preserves_64bit_dev(monkeypatch, dev):
    """The yielded identity must carry the full dev, not the low 32 bits."""
    ino = 0xBEEF1234
    _install_libproc(monkeypatch, {4242: [(7, dev, ino, "/tmp/state.db")]})
    seen = list(_iter_darwin_fd_targets())
    assert seen == [(4242, 7, "/tmp/state.db", (dev, ino))]
    assert seen[0][3][0] != (dev & 0xFFFFFFFF) or dev < (1 << 32)


@pytest.mark.parametrize("dev", LARGE_DEVS)
def test_count_db_holders_matches_64bit_dev(monkeypatch, tmp_path, dev):
    """End-to-end: a holder on a >2**32 device must be counted."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    target = os.path.realpath(str(db_path))
    real_stat = os.stat(target)
    ino = real_stat.st_ino
    fake_stat = os.stat_result(
        (real_stat.st_mode, ino, dev, real_stat.st_nlink, real_stat.st_uid,
         real_stat.st_gid, real_stat.st_size, real_stat.st_atime,
         real_stat.st_mtime, real_stat.st_ctime)
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(dbfile.sys, "platform", "darwin", raising=False)
    real_os_stat = os.stat

    def _stat(path, *args, **kwargs):
        if os.path.realpath(str(path)) == target:
            return fake_stat
        return real_os_stat(path, *args, **kwargs)

    monkeypatch.setattr(dbfile.os, "stat", _stat)
    _install_libproc(monkeypatch, {4242: [(7, dev, ino, "/irrelevant/path.db")]})

    # _iter_darwin_fd_targets does not report the pathname for this leg's
    # verdict in count_db_holders (identity-only), so any path matches as
    # long as the identity is exact.
    assert count_db_holders(db_path) == 1


def test_count_db_holders_distinguishes_low32_collision(monkeypatch, tmp_path):
    """Devices sharing the low 32 bits must NOT be conflated."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    target = os.path.realpath(str(db_path))
    real_stat = os.stat(target)
    ino = real_stat.st_ino
    stat_dev = (1 << 32) + 0xABCDEF
    fd_dev = (2 << 32) + 0xABCDEF  # same low 32, different device
    assert (stat_dev & 0xFFFFFFFF) == (fd_dev & 0xFFFFFFFF)
    assert stat_dev != fd_dev
    fake_stat = os.stat_result(
        (real_stat.st_mode, ino, stat_dev, real_stat.st_nlink, real_stat.st_uid,
         real_stat.st_gid, real_stat.st_size, real_stat.st_atime,
         real_stat.st_mtime, real_stat.st_ctime)
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(dbfile.sys, "platform", "darwin", raising=False)
    real_os_stat = os.stat

    def _stat(path, *args, **kwargs):
        if os.path.realpath(str(path)) == target:
            return fake_stat
        return real_os_stat(path, *args, **kwargs)

    monkeypatch.setattr(dbfile.os, "stat", _stat)
    _install_libproc(monkeypatch, {4242: [(7, fd_dev, ino, "/irrelevant/path.db")]})
    assert count_db_holders(db_path) == 0


def test_count_db_holders_counts_distinct_pids_large_dev(monkeypatch, tmp_path):
    """Distinct PIDs on the same large-dev inode count once each; others ignored."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    target = os.path.realpath(str(db_path))
    real_stat = os.stat(target)
    ino = real_stat.st_ino
    dev = (1 << 32) + 0x9999
    fake_stat = os.stat_result(
        (real_stat.st_mode, ino, dev, real_stat.st_nlink, real_stat.st_uid,
         real_stat.st_gid, real_stat.st_size, real_stat.st_atime,
         real_stat.st_mtime, real_stat.st_ctime)
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(dbfile.sys, "platform", "darwin", raising=False)
    real_os_stat = os.stat

    def _stat(path, *args, **kwargs):
        if os.path.realpath(str(path)) == target:
            return fake_stat
        return real_os_stat(path, *args, **kwargs)

    monkeypatch.setattr(dbfile.os, "stat", _stat)
    _install_libproc(monkeypatch, {
        111: [(3, dev, ino, "/a.db"), (4, dev, ino, "/a.db")],  # same pid twice
        222: [(5, dev, ino, "/a.db")],
        333: [(6, dev, ino + 1, "/a.db")],  # different inode: ignored
    })
    assert count_db_holders(db_path) == 2
