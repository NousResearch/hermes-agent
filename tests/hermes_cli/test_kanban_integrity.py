"""Regression tests for the corrupt-DB guards extracted to
``hermes_cli/kanban_integrity.py`` (wave-1 godfile decomposition, s1 c13).

The moved functions are exercised through the new module directly AND through
the ``hermes_cli.kanban_db`` re-export surface (``kb.*``), which is what all
existing callers (connect/repair paths, tests) use.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_integrity as ki

_SQLITE_HEADER_BYTES = b"SQLite format 3\x00"


@pytest.fixture
def clean_cache(tmp_path):
    """Ensure the moved guard's _INITIALIZED_PATHS check sees a fresh path."""
    yield tmp_path
    kb._INITIALIZED_PATHS.clear()


# ---------------------------------------------------------------------------
# Re-export surface
# ---------------------------------------------------------------------------


def test_moved_functions_are_re_exported_from_kanban_db():
    for name in (
        "_looks_like_tls_record_at",
        "_validate_sqlite_header",
        "_prune_corrupt_backups",
        "_backup_corrupt_db",
        "_integrity_messages_ok",
        "_run_integrity_check",
        "_repairable_index_names",
        "_attempt_index_reindex_repair",
        "_guard_existing_db_is_healthy",
    ):
        assert getattr(kb, name) is getattr(ki, name), name


# ---------------------------------------------------------------------------
# TLS-record sniffing (pure)
# ---------------------------------------------------------------------------


def test_looks_like_tls_record_at():
    handshake = bytes([0x16, 0x03, 0x03, 0x00, 0x05]) + b"\x01\x00\x00\x00\x00"
    assert ki._looks_like_tls_record_at(handshake, 0) is True
    assert ki._looks_like_tls_record_at(b"SQLit" + handshake, 5) is True
    assert ki._looks_like_tls_record_at(b"not tls here", 0) is False
    assert ki._looks_like_tls_record_at(b"\x16\x03\x03\x00\x05", 1) is False  # too short
    assert ki._looks_like_tls_record_at(b"\x99\x03\x03\x00\x05\x00", 0) is False  # bad type


# ---------------------------------------------------------------------------
# Header validation (pure, file-based)
# ---------------------------------------------------------------------------


def test_validate_sqlite_header_missing_file_is_fine(tmp_path):
    ki._validate_sqlite_header(tmp_path / "does-not-exist.db")  # no raise


def test_validate_sqlite_header_zero_byte_is_fine(tmp_path):
    p = tmp_path / "empty.db"
    p.write_bytes(b"")
    ki._validate_sqlite_header(p)  # no raise


def test_validate_sqlite_header_accepts_real_header(tmp_path):
    p = tmp_path / "valid.db"
    p.write_bytes(_SQLITE_HEADER_BYTES + b"\x00" * 4096)
    ki._validate_sqlite_header(p)  # no raise


def test_validate_sqlite_header_rejects_garbage(tmp_path):
    p = tmp_path / "junk.db"
    p.write_bytes(b"this is not a sqlite database at all")
    with pytest.raises(sqlite3.DatabaseError):
        ki._validate_sqlite_header(p)


# ---------------------------------------------------------------------------
# Integrity-check message helpers (pure)
# ---------------------------------------------------------------------------


def test_integrity_messages_ok():
    assert ki._integrity_messages_ok(["ok"]) is True
    assert ki._integrity_messages_ok(["OK"]) is True
    assert ki._integrity_messages_ok(["ok", "extra"]) is False
    assert ki._integrity_messages_ok([]) is False


def test_repairable_index_names():
    msgs = [
        "wrong # of entries in index idx_tasks_status",
        "row 42 missing from index idx_tasks_status",
    ]
    assert ki._repairable_index_names(msgs) == ["idx_tasks_status"]
    # Order of first appearance preserved.
    msgs2 = ["row 1 missing from index b_idx", "wrong # of entries in index a_idx"]
    assert ki._repairable_index_names(msgs2) == ["b_idx", "a_idx"]
    # A non-index error poisons the whole batch -> fail closed.
    assert ki._repairable_index_names(["database disk image is malformed"]) is None
    assert ki._repairable_index_names(["wrong # of entries in index x", "page 5 is corrupt"]) is None
    assert ki._repairable_index_names([]) is None
    assert ki._repairable_index_names([""]) is None


def test_run_integrity_check(tmp_path):
    db = tmp_path / "real.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    conn2 = sqlite3.connect(db)
    try:
        assert ki._run_integrity_check(conn2) == ["ok"]
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# Quarantine backups (pure, file-based)
# ---------------------------------------------------------------------------


def test_prune_corrupt_backups_retention_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(ki, "_CORRUPT_BACKUP_RETENTION", 3)
    base = tmp_path / "kanban.db"
    for i in range(5):
        (tmp_path / f"kanban.db.corrupt.aaaa{i}.bak").write_bytes(b"x")
    ki._prune_corrupt_backups(tmp_path, "kanban.db")
    remaining = sorted(tmp_path.glob("kanban.db.corrupt.*.bak"))
    assert len(remaining) == 3


def test_prune_corrupt_backups_keeps_just_minted(tmp_path, monkeypatch):
    monkeypatch.setattr(ki, "_CORRUPT_BACKUP_RETENTION", 3)
    for i in range(5):
        (tmp_path / f"kanban.db.corrupt.aaaa{i}.bak").write_bytes(b"x")
    keep = tmp_path / "kanban.db.corrupt.zzzz.bak"
    keep.write_bytes(b"x")
    ki._prune_corrupt_backups(tmp_path, "kanban.db", keep=keep)
    remaining = sorted(tmp_path.glob("kanban.db.corrupt.*.bak"))
    assert len(remaining) == 3
    assert keep in remaining


def test_backup_corrupt_db_content_addressed(tmp_path):
    db = tmp_path / "kanban.db"
    db.write_bytes(_SQLITE_HEADER_BYTES + b"corrupt page data here")
    backup = ki._backup_corrupt_db(db)
    assert backup is not None
    assert backup.name.startswith("kanban.db.corrupt.")
    assert backup.name.endswith(".bak")
    assert backup.exists()
    assert backup.read_bytes() == db.read_bytes()
    # Same corrupt bytes -> same content-addressed name (no duplicate).
    again = ki._backup_corrupt_db(db)
    assert again == backup


# ---------------------------------------------------------------------------
# End-to-end guard through the re-exported surface (matches repair test style)
# ---------------------------------------------------------------------------


def test_guard_existing_db_is_healthy_raises_on_corrupt(tmp_path, clean_cache):
    db = tmp_path / "kanban.db"
    db.write_bytes(b"not a database at all, just plain text bytes here")
    kb._INITIALIZED_PATHS.discard(str(db.resolve()))
    with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
        ki._guard_existing_db_is_healthy(db)
    assert excinfo.value.backup_path is not None
    assert excinfo.value.backup_path.exists()
