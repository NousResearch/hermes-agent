"""P5 idempotency atomicity (audit 2026-10-04, create_task double-insert race).

``create_task`` checked the idempotency key BEFORE ``write_txn``, with the
comment itself admitting "a concurrent-create race may insert twice". The race
window is closed by re-checking the key inside the transaction and by a partial
UNIQUE index (``idx_tasks_idempotency_uniq``, ``WHERE idempotency_key IS NOT
NULL AND status != 'archived'``) — the database itself is the last line of
defence against two ``INSERT``s (two concurrent creators) between the lookup
and the write. Dedup semantics are preserved:

- duplicate key with an existing non-archived task → returns the SAME id
  (no second row, no error);
- reopen: creating again with the same key after the card was archived still
  creates a NEW row — the partial index does not block it and never
  resurrects an archived card.

Covered here:
  1. Two threads calling ``create_task`` with the same NEW key on the same
     board produce exactly ONE task row and both calls return that id.
  2. The same key on two DIFFERENT boards stays independent (uniqueness is
     per board DB, not global).
  3. Schema-ensure (``init_db``/``connect``) creates the partial UNIQUE
     index (unique=1 with the documented partial WHERE) on a board missing
     it — i.e. every legacy board on its next open.
  4. The legacy non-unique ``idx_tasks_idempotency`` keeps existing next to
     the new unique one (P5 adds; it does not drop).
  5. Raising an ARCHIVED card to a non-archived status while another
     non-archived card carries the same key is refused loudly — the guarded
     ``create_task`` UPDATE path (``REPLACE`` semantics) raises
     :class:`IdempotencyStateConflictError` with both ids and the key in the
     message, and the archived card stays archived (no silent double-row of
     the key, no raw IntegrityError).
  6. Status-transition UPDATEs never NULL the indexed column (canary
     protecting the partial index from future writer drift).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

IDX = "idx_tasks_idempotency_uniq"


def _raw_conn(path) -> sqlite3.Connection:
    raw = sqlite3.connect(str(path))
    raw.row_factory = sqlite3.Row
    return raw


@pytest.fixture
def board_setup(tmp_path, monkeypatch, board="default"):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    db_path = kb.kanban_db_path(board=board)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return db_path


@pytest.fixture
def conn(board_setup):
    with kbc.connect() as c:
        yield c


# (1) ---------------------------------------------------------------------------
def test_same_key_two_threads_exactly_one_row_same_id(conn):
    """The pre-txn lookup could be doubled by two concurrent creators; the
    in-txn re-check plus the partial UNIQUE index must admit exactly one
    INSERT, and both callers must observe the same task id."""
    key = "race-key-1"
    results: list[dict] = []
    barrier = threading.Barrier(2)

    def creator(dup: bool) -> None:
        try:
            with kbc.connect() as c:
                barrier.wait()  # maximise overlap of the two pre-txn lookups
                tid = kb.create_task(
                    c, title="race card", idempotency_key=key,
                    body="from-A" if dup else "from-B",
                )
        except Exception as exc:  # the failure mode under test
            results.append({"error": repr(exc)})
        else:
            results.append({"id": tid})

    t1 = threading.Thread(target=creator, args=(True,))
    t2 = threading.Thread(target=creator, args=(False,))
    t1.start()
    t2.start()
    t1.join(20)
    t2.join(20)

    assert not any("error" in r for r in results), results
    ids = {r["id"] for r in results if "id" in r}
    assert len(ids) == 1, f"two creators settled on different ids: {results}"
    raw = _raw_conn(kb.kanban_db_path(board="default"))
    try:
        n = raw.execute(
            "SELECT count(*) FROM tasks WHERE idempotency_key = ?", (key,)
        ).fetchone()[0]
    finally:
        raw.close()
    assert n == 1, f"expected exactly one row for key {key!r}"


# (2) ---------------------------------------------------------------------------
def test_same_key_different_boards_independent(tmp_path, monkeypatch):
    """Uniqueness is per board DB: the index lives inside each board's file."""
    home = tmp_path / ".hermes"
    home.mkdir()
    for board in ("b1", "b2"):
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        with kb.scoped_current_board(board):
            # ``board=`` explicitly: scoped_current_board alone is not enough —
            # get_current_board() only honors the override for boards that
            # already exist, so a not-yet-created b1 falls back to default.
            db_path = kb.kanban_db_path(board=board)
            kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
            kb.init_db()
            with kbc.connect(db_path) as c:
                kb.create_task(c, title=f"card on {board}", idempotency_key="shared-key")
    for board in ("b1", "b2"):
        with kb.scoped_current_board(board):
            db_path = kb.kanban_db_path(board=board)
        raw = _raw_conn(db_path)
        try:
            n = raw.execute(
                "SELECT count(*) FROM tasks WHERE idempotency_key = 'shared-key'"
            ).fetchone()[0]
        finally:
            raw.close()
        assert n == 1


# (3) ---------------------------------------------------------------------------
def test_schema_ensure_unique_index(tmp_path, monkeypatch):
    """A legacy board missing the index gains it on the next open, with the
    exact partial WHERE and unique=1; reopening is idempotent."""
    db_path = tmp_path / "legacy-kanban.db"
    raw = sqlite3.connect(str(db_path))
    raw.executescript(kb.SCHEMA_SQL)
    raw.execute("DROP INDEX IF EXISTS " + IDX)
    raw.commit()
    raw.close()

    with kbc.connect(db_path) as c:
        entries = [(r[1], r[2]) for r in c.execute("PRAGMA index_list(tasks)").fetchall()]
        assert (IDX, 1) in entries, entries
        sql_row = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (IDX,)
        ).fetchone()
        norm = " ".join(sql_row[0].split()).lower()
        assert "unique" in norm
        assert "on tasks(idempotency_key)" in norm
        assert "where idempotency_key is not null and status != 'archived'" in norm

    # Idempotent: reopening the migrated board neither raises nor duplicates.
    raw = _raw_conn(db_path)
    try:
        n = raw.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='index' AND name=?",
            (IDX,),
        ).fetchone()[0]
    finally:
        raw.close()
    assert n == 1


# (4) ---------------------------------------------------------------------------
def test_legacy_index_not_dropped(conn):
    """The original non-unique lookup index stays — schema-ensure only adds."""
    names = {r[1] for r in conn.execute("PRAGMA index_list(tasks)").fetchall()}
    assert "idx_tasks_idempotency" in names
    assert IDX in names


# (5) ---------------------------------------------------------------------------
def test_archived_raise_conflict(board_setup):
    """Guarded create_task UPDATE path (REPLACE semantics) refuses the
    archive -> non-archived hop when another non-archived card holds the
    same key: typed IdempotencyStateConflictError with both ids and the key
    in the message; the archived card stays archived."""
    tid_a = None
    with kbc.connect() as c:
        tid_a = kb.create_task(c, title="first", idempotency_key="clash-key")
        assert kb.archive_task(c, tid_a)
        tid_b = kb.create_task(c, title="replacement", idempotency_key="clash-key")
        assert tid_b != tid_a
        with pytest.raises(kb.IdempotencyStateConflictError) as ei:
            kb.create_task(c, title="resurrected", idempotency_key="clash-key",
                           replace_state=(tid_a, "archived"))
        msg = str(ei.value)
        assert "clash-key" in msg and tid_a in msg and tid_b in msg
        st = c.execute("SELECT status FROM tasks WHERE id = ?", (tid_a,)).fetchone()[0]
        assert st == "archived"


# (6) ---------------------------------------------------------------------------
def test_no_status_edit_sets_null_key(conn):
    """Canary: status-transition UPDATEs never NULL the indexed column."""
    tid = kb.create_task(conn, title="canary", idempotency_key="canary-key")
    kb.complete_task(conn, tid, force=True, summary="ok", result="ok")
    assert kb._task_status(conn, tid) == "done"
    kb.archive_task(conn, tid)
    key = conn.execute(
        "SELECT idempotency_key FROM tasks WHERE id = ?", (tid,)
    ).fetchone()[0]
    assert key == "canary-key"
