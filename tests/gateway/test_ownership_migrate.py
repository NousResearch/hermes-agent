"""#47 Phase 2 — sandbox tests for ``ownership_migrate.py``.

Every test runs on a private ``tmp_path`` DB; the production DB is never
touched. The tool must: create the two ledger tables behind a backup, verify
rowcounts + integrity, be idempotent, refuse a non-WAL DB, and roll back
ledger-only by default (dropping only the two ledger tables, so post-apply
writes to every other table survive). ``--full`` is the explicit emergency
path that restores the whole backup instead.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from ownership_migrate import (
    BACKUP_SUFFIX,
    LEDGER_TABLES,
    Abort,
    _user_tables,
    cmd_apply,
    cmd_backfill,
    cmd_rollback,
    cmd_status,
    main,
)


# --- helpers ----------------------------------------------------------------

def _wal_db(path: Path, *, rows: int = 1) -> Path:
    """A minimal board-shaped DB in WAL mode with real rows to preserve."""
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, kind TEXT)")
        for i in range(rows):
            conn.execute("INSERT INTO tasks VALUES (?, ?)", (f"t{i}", f"card {i}"))
            conn.execute("INSERT INTO task_events (task_id, kind) VALUES (?, 'created')", (f"t{i}",))
        conn.commit()
    finally:
        conn.close()
    return path


def _delete_db(path: Path) -> Path:
    """A rollback-journal DB (journal_mode must NOT be wal)."""
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO tasks VALUES ('t1')")
        conn.commit()
    finally:
        conn.close()
    return path


def _tables(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return _user_tables(conn)
    finally:
        conn.close()


def _rowcount(path: Path, table: str) -> int:
    conn = sqlite3.connect(str(path))
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    finally:
        conn.close()


def _journal(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}{BACKUP_SUFFIX}*.bak"))


# --- status -----------------------------------------------------------------

def test_status_reports_missing_ledger_on_wal_db(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    res = cmd_status(db)
    assert res["wal"] is True
    assert res["journal_mode"] == "wal"
    assert res["ledger_tables_present"] == []
    assert res["schema_ok"] is None          # nothing to validate yet
    assert res["schema_problems"] == []
    assert res["integrity"] == ["ok"]
    assert res["backup_count"] == 0


def test_status_reports_ready_after_apply(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    cmd_apply(db)
    res = cmd_status(db)
    assert res["ledger_tables_present"] == list(LEDGER_TABLES)
    assert res["schema_ok"] is True
    assert res["schema_problems"] == []
    assert res["ledger_rowcounts"] == {t: 0 for t in LEDGER_TABLES}
    assert res["backup_count"] == 1


# --- apply ------------------------------------------------------------------

def test_apply_creates_ledger_tables_with_backup_and_verifies(tmp_path):
    db = _wal_db(tmp_path / "kanban.db", rows=2)
    res = cmd_apply(db)
    assert res["already_applied"] is False
    assert res["created"] == list(LEDGER_TABLES)
    assert res["rowcounts_unchanged"] is True
    assert res["integrity"] == ["ok"]
    backup = Path(res["backup"])
    assert backup.exists()
    # The board data survived untouched.
    assert _rowcount(db, "tasks") == 2
    assert _rowcount(db, "task_events") == 2
    for table in LEDGER_TABLES:
        assert _rowcount(db, table) == 0
    # The backup is a pre-apply snapshot: it must NOT carry the ledger tables.
    assert set(LEDGER_TABLES).isdisjoint(_tables(backup))
    assert _rowcount(backup, "tasks") == 2


def test_apply_is_idempotent(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    first = cmd_apply(db)
    second = cmd_apply(db)
    assert first["already_applied"] is False
    assert second["already_applied"] is True
    assert second["created"] == []
    assert second["backup"] is None
    assert len(_backups(db)) == 1          # no second backup on a no-op re-run
    assert _rowcount(db, "tasks") == 1
    assert set(LEDGER_TABLES) <= set(_tables(db))


def test_apply_dry_run_writes_nothing(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    res = cmd_apply(db, dry_run=True)
    assert res["dry_run"] is True
    assert res["would_create"] == list(LEDGER_TABLES)
    assert set(LEDGER_TABLES).isdisjoint(_tables(db))
    assert _backups(db) == []
    assert _rowcount(db, "tasks") == 1


def test_apply_aborts_when_not_wal(tmp_path):
    db = _delete_db(tmp_path / "kanban.db")
    with pytest.raises(Abort, match="not 'wal'"):
        cmd_apply(db)
    assert set(LEDGER_TABLES).isdisjoint(_tables(db))
    assert _backups(db) == []
    assert main(["--db", str(db), "apply"]) == 2


def test_apply_aborts_on_incompatible_existing_schema(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE task_ownership (scope_key TEXT)")   # wrong shape
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(Abort, match="incompatible ledger schema"):
        cmd_apply(db)
    assert _backups(db) == []              # aborted before any backup


def test_apply_preserves_existing_rowcounts(tmp_path):
    db = _wal_db(tmp_path / "kanban.db", rows=5)
    before = {t: _rowcount(db, t) for t in _tables(db)}
    cmd_apply(db)
    after = {t: _rowcount(db, t) for t in _tables(db) if t not in LEDGER_TABLES}
    assert before == after


# --- rollback ---------------------------------------------------------------

def test_rollback_is_ledger_only_and_preserves_post_apply_writes(tmp_path):
    db = _wal_db(tmp_path / "kanban.db", rows=3)
    cmd_apply(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("INSERT INTO tasks VALUES ('tx', 'transient')")
        conn.execute("INSERT INTO task_events (task_id, kind) VALUES ('tx', 'created')")
        conn.execute("CREATE TABLE task_comments (id INTEGER PRIMARY KEY, task_id TEXT)")
        conn.commit()
    finally:
        conn.close()

    rolled = cmd_rollback(db)

    assert rolled["mode"] == "ledger"
    assert rolled["backup"] is None
    assert rolled["dropped"] == list(LEDGER_TABLES)
    assert set(LEDGER_TABLES).isdisjoint(_tables(db))
    # Every non-ledger table and row written after apply survives untouched.
    assert _rowcount(db, "tasks") == 4
    assert _rowcount(db, "task_events") == 4
    assert "task_comments" in _tables(db)
    assert rolled["non_ledger_rowcounts_unchanged"] is True
    assert _journal(db) == "wal"
    assert rolled["integrity"] == ["ok"]


def test_rollback_full_with_explicit_backup_discards_post_apply_writes(tmp_path):
    db = _wal_db(tmp_path / "kanban.db", rows=2)
    backup = Path(cmd_apply(db)["backup"])
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("INSERT INTO tasks VALUES ('tx', 'transient')")
        conn.commit()
    finally:
        conn.close()
    assert _rowcount(db, "tasks") == 3

    rolled = cmd_rollback(db, backup=backup, full=True)

    assert rolled["mode"] == "full"
    assert rolled["matches_backup"] is True
    assert set(LEDGER_TABLES).isdisjoint(_tables(db))
    assert _rowcount(db, "tasks") == 2     # the post-apply row is gone
    assert backup.exists()                 # the backup is kept for a second restore


def test_rollback_ledger_only_needs_no_backup_but_full_aborts(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    rolled = cmd_rollback(db)              # no ledger tables, no backup: no-op
    assert rolled["mode"] == "ledger"
    assert rolled["dropped"] == []
    assert _rowcount(db, "tasks") == 1
    assert main(["--db", str(db), "rollback"]) == 0

    with pytest.raises(Abort, match="no backup"):
        cmd_rollback(db, full=True)
    assert main(["--db", str(db), "rollback", "--full"]) == 2


def test_rollback_aborts_when_not_wal(tmp_path):
    db = _delete_db(tmp_path / "kanban.db")
    with pytest.raises(Abort, match="not 'wal'"):
        cmd_rollback(db)
    with pytest.raises(Abort, match="not 'wal'"):
        cmd_rollback(db, full=True)


def test_rollback_full_aborts_when_backup_is_missing_file(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    with pytest.raises(Abort, match="no backup"):
        cmd_rollback(db, backup=tmp_path / "does-not-exist.bak", full=True)


# --- CLI exit codes ---------------------------------------------------------

def test_main_apply_status_exit_zero_and_json(tmp_path, capsys):
    db = _wal_db(tmp_path / "kanban.db")
    assert main(["--db", str(db), "apply"]) == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out
    assert '"already_applied": false' in out
    assert main(["--db", str(db), "status"]) == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out
    assert '"schema_ok": true' in out


def test_main_apply_missing_db_aborts(tmp_path):
    db = tmp_path / "absent.db"
    assert main(["--db", str(db), "apply"]) == 2
    assert main(["--db", str(db), "status"]) == 2


# --- backfill (C5c) ---------------------------------------------------------
#
# A card that was already ``running`` when the gate was activated never went
# through the fresh-admission gate, so the ledger has no row for it. The sweep
# must fence it for its recorded owner (newest owner_admitted marker for the
# current run, else tasks.session_id), skip it when no owner is provable, and
# never steal a live foreign owner.

BOARD = "backfill"
BODY = "source_repo: Andrew288369/ANDREW-GRP\nsource_issue: 47"
SCOPE = "gh:andrew288369/andrew-grp#47"


def _board_db(tmp_path, monkeypatch, *, bodies=None):
    """A real WAL board with one RUNNING card per body; returns the DB path."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env",
                        lambda name: str(tmp_path) if name == "default" else str(tmp_path / name))
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    kb.create_board(BOARD)
    db = kb.kanban_db_path(board=BOARD)
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        for i, body in enumerate(bodies or [BODY]):
            tid = kb.create_task(conn, title=f"card {i}", assignee="default",
                                 workspace_kind="dir", workspace_path=str(ws), body=body)
            kb.recompute_ready(conn)
            kb.claim_task(conn, tid)
    return db


def _running_cards(db):
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT id, current_run_id FROM tasks ORDER BY id")]


def _seed_marker(db, task_id, run_id, session_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        kb._append_event(conn, task_id, "owner_admitted",
                         {"session_id": session_id, "request_id": "kanban:probe"},
                         run_id=run_id)


def _set_session(db, session_id):
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.execute("UPDATE tasks SET session_id=?", (session_id,))


def _owners(db):
    conn = sqlite3.connect(str(db))
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM task_ownership ORDER BY scope_key")]
    finally:
        conn.close()


def test_backfill_claims_running_cards_from_owner_markers(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch, bodies=[
        "source_repo: Andrew288369/ANDREW-GRP\nsource_issue: 47",
        "source_repo: Andrew288369/ANDREW-GRP\nsource_issue: 48",
    ])
    for card, session in zip(_running_cards(db), ("sess-a", "sess-b")):
        _seed_marker(db, card["id"], card["current_run_id"], session)
    cmd_apply(db)
    res = cmd_backfill(db)
    assert res["scanned"] == 2 and res["skipped"] == [] and res["conflicts"] == []
    assert {c["owner_session"] for c in res["claimed"]} == {"sess-a", "sess-b"}
    assert {c["owner_source"] for c in res["claimed"]} == {"owner_admitted"}
    owners = _owners(db)
    assert len(owners) == 2
    assert {o["owner_session"] for o in owners} == {"sess-a", "sess-b"}
    assert all(o["state"] == "ACTIVE" for o in owners)


def test_backfill_falls_back_to_tasks_session_id(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    _set_session(db, "sess-card")
    cmd_apply(db)
    res = cmd_backfill(db)
    assert [c["owner_session"] for c in res["claimed"]] == ["sess-card"]
    assert res["claimed"][0]["owner_source"] == "tasks.session_id"
    assert [o["owner_session"] for o in _owners(db)] == ["sess-card"]


def test_backfill_ignores_marker_from_a_previous_run(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    card = _running_cards(db)[0]
    _seed_marker(db, card["id"], card["current_run_id"] + 100, "sess-stale")
    _set_session(db, "sess-current")
    cmd_apply(db)
    res = cmd_backfill(db)
    assert [c["owner_session"] for c in res["claimed"]] == ["sess-current"]
    assert res["claimed"][0]["owner_source"] == "tasks.session_id"


def test_backfill_skips_running_card_without_owner_session(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    cmd_apply(db)
    res = cmd_backfill(db)
    assert res["scanned"] == 1 and res["claimed"] == []
    assert len(res["skipped"]) == 1
    assert res["skipped"][0]["reason"] == "no_owner_session"
    assert _owners(db) == []          # never a fabricated session


def test_backfill_skips_malformed_owner_session(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    _set_session(db, "not a session!")
    cmd_apply(db)
    res = cmd_backfill(db)
    assert res["claimed"] == []
    assert res["skipped"][0]["reason"] == "invalid_owner_session"
    assert _owners(db) == []


def test_backfill_skips_underivable_scope(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    card = _running_cards(db)[0]
    _seed_marker(db, card["id"], card["current_run_id"], "sess-a")
    monkeypatch.setattr("cross_session_ownership.canonical_key_from_kanban",
                        lambda task, **k: None)
    cmd_apply(db)
    res = cmd_backfill(db)
    assert res["claimed"] == []
    assert res["skipped"][0]["reason"] == "no_identity"
    assert _owners(db) == []


def test_backfill_dry_run_writes_nothing(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    card = _running_cards(db)[0]
    _seed_marker(db, card["id"], card["current_run_id"], "sess-a")
    cmd_apply(db)
    res = cmd_backfill(db, dry_run=True)
    assert res["dry_run"] is True
    assert [c["owner_session"] for c in res["claimed"]] == ["sess-a"]
    assert _owners(db) == [] and _rowcount(db, "task_ownership_events") == 0
    real = cmd_backfill(db)
    assert [c["owner_session"] for c in real["claimed"]] == ["sess-a"]
    assert len(_owners(db)) == 1


def test_backfill_is_idempotent(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    card = _running_cards(db)[0]
    _seed_marker(db, card["id"], card["current_run_id"], "sess-a")
    cmd_apply(db)
    first = cmd_backfill(db)
    second = cmd_backfill(db)
    assert [c["reason"] for c in first["claimed"]] == ["claimed"]
    assert [c["reason"] for c in second["claimed"]] == ["already_owner"]
    assert len(_owners(db)) == 1


def test_backfill_never_steals_a_live_foreign_owner(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    card = _running_cards(db)[0]
    _seed_marker(db, card["id"], card["current_run_id"], "sess-owner")
    cmd_apply(db)
    from cross_session_ownership import claim_ownership
    conn = sqlite3.connect(str(db))
    try:
        assert claim_ownership(conn, SCOPE, "sess-foreign", ttl_seconds=900)["ok"] is True
    finally:
        conn.close()
    res = cmd_backfill(db)
    assert res["claimed"] == [] and len(res["conflicts"]) == 1
    assert res["conflicts"][0]["reason"].startswith("owned_by_other_session")
    owners = _owners(db)
    assert len(owners) == 1 and owners[0]["owner_session"] == "sess-foreign"
    assert owners[0]["state"] == "ACTIVE"


def test_backfill_takes_over_a_stale_owner(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    card = _running_cards(db)[0]
    _seed_marker(db, card["id"], card["current_run_id"], "sess-owner")
    cmd_apply(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO task_ownership (scope_key, owner_session, claim_lock,"
            " claim_expires, last_heartbeat, state, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (SCOPE, "sess-dead", "lock", 1, 1, "ACTIVE", 1, 1))
        conn.commit()
    finally:
        conn.close()
    res = cmd_backfill(db)
    assert [c["reason"] for c in res["claimed"]] == ["taken_over_from:sess-dead"]
    owners = _owners(db)
    assert len(owners) == 1 and owners[0]["owner_session"] == "sess-owner"


def test_backfill_ignores_cards_that_are_not_running(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.execute("UPDATE tasks SET status='done'")
    cmd_apply(db)
    res = cmd_backfill(db)
    assert res["scanned"] == 0 and res["claimed"] == []


def test_backfill_aborts_before_apply(tmp_path, monkeypatch):
    db = _board_db(tmp_path, monkeypatch)
    with pytest.raises(Abort, match="apply"):
        cmd_backfill(db)
    assert set(LEDGER_TABLES).isdisjoint(_tables(db))


def test_main_backfill_exit_zero_and_json(tmp_path, monkeypatch, capsys):
    db = _board_db(tmp_path, monkeypatch)
    card = _running_cards(db)[0]
    _seed_marker(db, card["id"], card["current_run_id"], "sess-a")
    assert main(["--db", str(db), "apply"]) == 0
    assert main(["--db", str(db), "backfill", "--dry-run"]) == 0
    assert '"scanned": 1' in capsys.readouterr().out
    assert main(["--db", str(db), "backfill"]) == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out and '"dry_run": false' in out
