"""Tests for kanban DB corruption repair, backup retention, WAL checkpointing,
and the ``hermes kanban repair`` CLI verb."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _build_board_db(db_path: Path, tasks: int = 12) -> None:
    """Create a real board DB with data so indexes have entries."""
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(db_path=db_path)
    with kb.connect(db_path=db_path) as conn:
        for i in range(tasks):
            kb.create_task(conn, title=f"task-{i}")
    conn.close()
    # Force the next connect() to re-run the health guard.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))


def _corrupt_index(db_path: Path, index_name: str) -> None:
    """Make ``index_name`` disagree with its table → 'wrong # of entries'.

    writable_schema approach: temporarily rewrite the index's schema SQL to
    a partial index matching no rows, REINDEX under that lie (emptying the
    index b-tree), then restore the original SQL. integrity_check now sees a
    non-partial index whose b-tree is missing every row — exactly the
    index-scoped corruption class ('wrong # of entries in index <name>' +
    'row N missing from index <name>') with intact table b-trees.
    """
    conn = sqlite3.connect(db_path, isolation_level=None)
    original_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (index_name,)
    ).fetchone()[0]
    lie = original_sql + " WHERE 0"
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "UPDATE sqlite_master SET sql = ? WHERE name = ?", (lie, index_name)
    )
    conn.execute("PRAGMA writable_schema=OFF")
    conn.close()
    # New connection so the rewritten schema is what REINDEX parses.
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute(f'REINDEX "{index_name}"')
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "UPDATE sqlite_master SET sql = ? WHERE name = ?",
        (original_sql, index_name),
    )
    conn.execute("PRAGMA writable_schema=OFF")
    conn.close()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))


def _write_page_corrupt_db(path: Path) -> bytes:
    """Valid SQLite header, garbage pages — NON-index corruption class."""
    header = b"SQLite format 3\x00" + b"\x10\x00\x02\x02\x00\x40\x20\x20"
    header += b"\x00\x00\x00\x0c\x00\x00\x23\x46\x00\x00\x00\x00"
    header = header.ljust(100, b"\x00")
    blob = header + b"definitely not a valid sqlite page \x00\x01\x02\x03" * 64
    path.write_bytes(blob)
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return blob


def _integrity_messages(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Index-error parsing (generic, no hardcoded index names)
# ---------------------------------------------------------------------------

def test_repairable_index_names_parses_generically():
    messages = [
        "wrong # of entries in index idx_anything_at_all",
        "row 3 missing from index idx_anything_at_all",
        "wrong # of entries in index some_other_index",
    ]
    assert kb._repairable_index_names(messages) == [
        "idx_anything_at_all", "some_other_index",
    ]


@pytest.mark.parametrize("messages", [
    [],
    ["ok"],
    ["database disk image is malformed"],
    ["*** in database main ***", "Page 5: btreeInitPage() returns error code 11"],
    # Mixed: one repairable line + one non-index line → NOT repairable.
    ["wrong # of entries in index idx_tasks_status",
     "database disk image is malformed"],
])
def test_repairable_index_names_rejects_non_index_classes(messages):
    assert kb._repairable_index_names(messages) is None


# ---------------------------------------------------------------------------
# Narrow auto-repair in the connect-time guard
# ---------------------------------------------------------------------------

def test_connect_auto_repairs_index_only_corruption(tmp_path, caplog):
    """Index-only integrity errors are REINDEXed and connect proceeds."""
    import logging

    db_path = tmp_path / "kanban.db"
    _build_board_db(db_path)
    _corrupt_index(db_path, "idx_tasks_status")

    # Precondition: the fixture really produced the index-scoped class.
    messages = _integrity_messages(db_path)
    assert any(m.startswith("wrong # of entries in index") for m in messages)
    assert kb._repairable_index_names(messages) == ["idx_tasks_status"]

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        conn = kb.connect(db_path=db_path)
    try:
        # DB is clean again and data survived.
        row = conn.execute("PRAGMA integrity_check").fetchone()
        assert row[0] == "ok"
        titles = {t.title for t in kb.list_tasks(conn)}
        assert "task-0" in titles and "task-11" in titles
    finally:
        conn.close()
    assert "auto-repaired via REINDEX" in caplog.text

    # The corrupt bytes were quarantined BEFORE the repair mutated the file.
    backups = list(tmp_path.glob("kanban.db.corrupt.*.bak"))
    assert len(backups) == 1
    backup_messages_db = backups[0]
    # The backup still exhibits the pre-repair corruption.
    pre = _integrity_messages(backup_messages_db)
    assert any(m.startswith("wrong # of entries in index") for m in pre)


def test_connect_still_fails_closed_on_page_corruption(tmp_path):
    """Non-index corruption keeps the exact fail-closed contract."""
    db_path = tmp_path / "kanban.db"
    original = _write_page_corrupt_db(db_path)

    with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
        kb.connect(db_path=db_path)

    err = excinfo.value
    assert err.backup_path is not None and err.backup_path.exists()
    # No repair was attempted: original bytes untouched on the live path.
    assert db_path.read_bytes() == original


def test_guard_fails_closed_when_reindex_does_not_clean(tmp_path, monkeypatch):
    """If the post-REINDEX re-check is not clean, raise exactly as today."""
    db_path = tmp_path / "kanban.db"
    _build_board_db(db_path)
    _corrupt_index(db_path, "idx_tasks_status")

    monkeypatch.setattr(
        kb, "_attempt_index_reindex_repair",
        lambda path, names: (False, ["wrong # of entries in index idx_tasks_status"]),
    )
    with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
        kb.connect(db_path=db_path)
    assert "REINDEX auto-repair attempted" in str(excinfo.value)
    assert excinfo.value.backup_path is not None
    assert excinfo.value.backup_path.exists()


def test_repaired_db_connects_normally_afterwards(tmp_path):
    """After one auto-repair, subsequent connects are ordinary fast-path."""
    db_path = tmp_path / "kanban.db"
    _build_board_db(db_path)
    _corrupt_index(db_path, "idx_tasks_status")

    conn = kb.connect(db_path=db_path)
    conn.close()
    # Second connect: healthy cache path, no new backups minted.
    before = set(tmp_path.glob("kanban.db.corrupt.*.bak"))
    conn = kb.connect(db_path=db_path)
    try:
        kb.create_task(conn, title="post-repair")
        assert "post-repair" in {t.title for t in kb.list_tasks(conn)}
    finally:
        conn.close()
    assert set(tmp_path.glob("kanban.db.corrupt.*.bak")) == before


# ---------------------------------------------------------------------------
# Corrupt-backup retention cap
# ---------------------------------------------------------------------------

def test_corrupt_backup_retention_cap_prunes_oldest(tmp_path, monkeypatch):
    """Mutating corruption can't accumulate quarantine files forever.

    Regression for the field report of 124 ``.corrupt.*.bak`` files: each
    distinct corrupt byte-state mints a new content-addressed backup, so a
    board whose file keeps changing between failures grows one backup per
    mutation. The cap keeps only the newest ``_CORRUPT_BACKUP_RETENTION``.
    """
    monkeypatch.setattr(kb, "_CORRUPT_BACKUP_RETENTION", 3)
    db_path = tmp_path / "kanban.db"
    _write_page_corrupt_db(db_path)

    minted: list[Path] = []
    for i in range(8):
        # Mutate the corrupt bytes → new sha → new backup each round.
        with db_path.open("r+b") as fh:
            fh.seek(200)
            fh.write(bytes([i]) * 16)
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
            kb.connect(db_path=db_path)
        assert excinfo.value.backup_path is not None
        minted.append(excinfo.value.backup_path)
        # The just-created backup always survives its own prune pass.
        assert excinfo.value.backup_path.exists()

    remaining = sorted(tmp_path.glob("kanban.db.corrupt.*.bak"))
    assert len(remaining) == 3, (
        f"expected retention cap of 3, found {len(remaining)}: {remaining}"
    )
    # The newest backup (this round's) is among the survivors.
    assert minted[-1] in remaining


def test_corrupt_backup_retention_prunes_sidecar_copies(tmp_path, monkeypatch):
    """Pruned backups take their copied -wal/-shm sidecars with them."""
    monkeypatch.setattr(kb, "_CORRUPT_BACKUP_RETENTION", 1)
    db_path = tmp_path / "kanban.db"
    _write_page_corrupt_db(db_path)

    # Fabricate an old backup + sidecars that the next prune should remove.
    import os
    stale = tmp_path / "kanban.db.corrupt.deadbeef00000000.bak"
    stale.write_bytes(b"old corrupt bytes")
    (tmp_path / (stale.name + "-wal")).write_bytes(b"wal")
    (tmp_path / (stale.name + "-shm")).write_bytes(b"shm")
    past = 1_000_000_000
    os.utime(stale, (past, past))

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
        kb.connect(db_path=db_path)
    fresh = excinfo.value.backup_path
    assert fresh is not None and fresh.exists()

    assert not stale.exists()
    assert not (tmp_path / (stale.name + "-wal")).exists()
    assert not (tmp_path / (stale.name + "-shm")).exists()


def test_identical_corrupt_bytes_still_reuse_one_backup(tmp_path):
    """The retention cap must not break content-addressed dedupe."""
    db_path = tmp_path / "kanban.db"
    _write_page_corrupt_db(db_path)

    backups: set[Path] = set()
    for _ in range(5):
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
            kb.connect(db_path=db_path)
        assert excinfo.value.backup_path is not None
        backups.add(excinfo.value.backup_path)
    assert len(backups) == 1
    assert len(list(tmp_path.glob("kanban.db.corrupt.*.bak"))) == 1
