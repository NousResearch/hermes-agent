"""A zero-length WAL/SHM sidecar has no stable identity on some filesystems.

macOS FSKit exFAT (and other cluster-addressed FAT drivers) report a provisional inode number near
2**64 for a zero-length file and replace it once the first cluster is allocated. Recording that
provisional identity at open made the first write after a reopen raise DeletedWalGenerationError
although no generation was lost.
"""

import os
from pathlib import Path

import pytest

import hermes_state_dbfile
from hermes_state import SessionDB
from tests.hermes_state._wal_generation_harness import make_db, pin_wal, require_wal

PROVISIONAL_INO = (1 << 64) - 4099
_EXTRA_STAT_FIELDS = (
    "st_atime", "st_mtime", "st_ctime", "st_atime_ns", "st_mtime_ns", "st_ctime_ns",
    "st_blocks", "st_blksize", "st_rdev", "st_flags", "st_birthtime",
)


@pytest.fixture
def force_wal(monkeypatch):
    pin_wal(monkeypatch)


def _with_ino(st: os.stat_result, ino: int) -> os.stat_result:
    seq = list(st)
    seq[1] = ino
    extra = {name: getattr(st, name) for name in _EXTRA_STAT_FIELDS if hasattr(st, name)}
    return os.stat_result(seq, extra)


def _fskit_like_stat(real_stat):
    """Zero-length sidecars report a provisional inode, as macOS FSKit exFAT does."""
    def fake_stat(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        if os.fspath(path).endswith(("-wal", "-shm")) and st.st_size == 0:
            return _with_ino(st, PROVISIONAL_INO)
        return st
    return fake_stat


def test_zero_length_sidecar_has_no_identity(tmp_path):
    wal = tmp_path / "state.db-wal"
    wal.write_bytes(b"")
    assert hermes_state_dbfile._stable_sidecar_identity(wal) is None
    wal.write_bytes(b"frame")
    st = os.stat(wal)
    assert hermes_state_dbfile._stable_sidecar_identity(wal) == (st.st_dev, st.st_ino)


def test_provisional_inode_is_not_an_identity(tmp_path, monkeypatch):
    wal = tmp_path / "state.db-wal"
    wal.write_bytes(b"frame")
    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        return _with_ino(st, PROVISIONAL_INO) if Path(path) == wal else st

    monkeypatch.setattr(os, "stat", fake_stat)
    assert hermes_state_dbfile._stable_sidecar_identity(wal) is None


def test_reopen_with_empty_wal_survives_first_write_with_provisional_inodes(tmp_path, force_wal, monkeypatch):
    path = tmp_path / "state.db"
    db = make_db(path, "s1", "hello")
    require_wal(db)
    db.close()
    Path(os.fspath(path) + "-wal").write_bytes(b"")  # what a TRUNCATE checkpoint leaves behind
    monkeypatch.setattr(os, "stat", _fskit_like_stat(os.stat))

    reopened = SessionDB(db_path=path)
    try:
        reopened.append_message("s1", role="user", content="after-reopen")
        assert any(m["content"] == "after-reopen" for m in reopened.get_messages("s1"))
    finally:
        reopened.close()
