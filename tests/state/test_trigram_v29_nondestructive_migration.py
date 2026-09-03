"""Non-destructive v29→v30 ordinary open under sessions.trigram_fts=false (P1 Tension B).

A schema-v29 database with historical trigram storage (table + view + triggers
+ rows) opened writable with the knob false must be quarantined — triggers
dropped, stale marker written — while the historical table, view and rows are
preserved byte-for-byte. Physical retirement happens only via explicit
``optimize-storage``. Schema still advances to 30 so the migration does not
re-run forever.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import FTS_SQL, SCHEMA_SQL, SCHEMA_VERSION, SessionDB
from hermes_state_common import FTS_TRIGRAM_STALE_KEY

_FTS_TRIGRAM_TRIGGERS = (
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)

# Historical v29 trigram DDL: v2 column set (content, tool_name), NO cron or
# subagent exclusion — the pre-#102032 boundary the migration upgrades.
_V29_TRIGRAM_DDL = """
DROP TRIGGER IF EXISTS messages_fts_trigram_insert;
DROP TRIGGER IF EXISTS messages_fts_trigram_delete;
DROP TRIGGER IF EXISTS messages_fts_trigram_update;
DROP TABLE IF EXISTS messages_fts_trigram;
DROP VIEW IF EXISTS messages_fts_trigram_src;

CREATE VIEW messages_fts_trigram_src AS
    SELECT id, role, content, tool_name FROM messages WHERE role <> 'tool';

CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
    content, tool_name,
    content='messages_fts_trigram_src',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
BEGIN
    INSERT INTO messages_fts_trigram(rowid, content, tool_name)
    VALUES (new.id, new.content, new.tool_name);
END;

CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name)
    VALUES ('delete', old.id, old.content, old.tool_name);
END;

CREATE TRIGGER messages_fts_trigram_update
AFTER UPDATE OF content, tool_name, role ON messages
WHEN (old.content IS NOT new.content
   OR old.tool_name IS NOT new.tool_name
   OR old.role IS NOT new.role)
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name)
    SELECT 'delete', old.id, old.content, old.tool_name WHERE old.role <> 'tool';
    INSERT INTO messages_fts_trigram(rowid, content, tool_name)
    SELECT new.id, new.content, new.tool_name WHERE new.role <> 'tool';
END;
"""


def _trigger_names(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }


def _exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()
        is not None
    )


def _marker(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM state_meta WHERE key = ? LIMIT 1",
            (FTS_TRIGRAM_STALE_KEY,),
        ).fetchone()
        is not None
    )


def _trigram_rowids(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT id FROM messages_fts_trigram_docsize"
        ).fetchall()
    }


def _schema_ddl(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    return (row[0] or "") if row else ""


def _build_v29_db(path: Path, *, knob: bool = False) -> dict:
    """Realistic v29 DB: schema, base FTS, historical v29 trigram + rows."""
    if knob:
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "config.yaml").write_text(
            "sessions:\n  trigram_fts: false\n", encoding="utf-8"
        )
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA_SQL)
        # SCHEMA_SQL creates an EMPTY schema_version table — _init_schema
        # inserts the row on fresh DBs, so a hand-built historical fixture
        # must insert the v29 row explicitly or the version-gated migration
        # chain never runs (row None → treated as fresh).
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (29)"
        )
        conn.executescript(FTS_SQL)
        conn.executescript(_V29_TRIGRAM_DDL)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("root", "cli", 1.0),
        )
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("cronjob", "cron", 2.0),
        )
        conn.execute(
            "INSERT INTO messages (session_id, timestamp, role, content, tool_name) "
            "VALUES (?, ?, ?, ?, ?)",
            ("root", 1.5, "user", "v29历史行 root-legacy-token", None),
        )
        conn.execute(
            "INSERT INTO messages (session_id, timestamp, role, content, tool_name) "
            "VALUES (?, ?, ?, ?, ?)",
            ("cronjob", 2.5, "user", "v29定时任务 cron-legacy-token", None),
        )
        conn.commit()
        historical = sorted(_trigram_rowids(conn))
        assert historical, "v29 fixture must contain historical trigram rows"
        return {"historical_rowids": historical}
    finally:
        conn.close()


def _fts5_with_trigram(path: Path) -> bool:
    try:
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE t_probe USING fts5(x, tokenize='trigram')"
            )
        finally:
            conn.close()
        return True
    except sqlite3.OperationalError:
        return False


# ── 1. Ordinary false open preserves historical storage ─────────────────


def test_v29_false_ordinary_open_preserves_historical_storage(tmp_path):
    path = tmp_path / "state.db"
    fixture = _build_v29_db(path, knob=True)
    if not _fts5_with_trigram(tmp_path / "probe.db"):
        pytest.skip("trigram tokenizer unavailable in this SQLite build")

    db = SessionDB(db_path=path)
    try:
        conn = db._conn
        # Table + view + rows preserved exactly.
        assert _exists(conn, "messages_fts_trigram")
        view_before = _schema_ddl(conn, "messages_fts_trigram_src")
        assert view_before, "historical view must survive the ordinary open"
        assert sorted(_trigram_rowids(conn)) == fixture["historical_rowids"]
        # Triggers dropped, marker written, availability false.
        names = _trigger_names(conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names
        assert _marker(conn)
        assert db._trigram_available is False
        assert db._trigram_enabled is False
        # Schema advanced to 30 (migration does not re-run forever).
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == SCHEMA_VERSION == 30
        # Canonical messages + standard FTS untouched.
        assert (
            conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        )
        assert _exists(conn, "messages_fts")
        assert db.search_messages("root-legacy-token")
    finally:
        db.close()


# ── 2. Repeated false open: idempotent, no further mutation ─────────────


def test_v29_false_repeated_open_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    fixture = _build_v29_db(path, knob=True)
    if not _fts5_with_trigram(tmp_path / "probe.db"):
        pytest.skip("trigram tokenizer unavailable in this SQLite build")

    db1 = SessionDB(db_path=path)
    db1.close()
    snap = _snapshot(path)

    db2 = SessionDB(db_path=path)
    try:
        conn = db2._conn
        assert sorted(_trigram_rowids(conn)) == fixture["historical_rowids"]
        names = _trigger_names(conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names
        assert _marker(conn)
        assert db2._trigram_available is False
        assert db2.search_messages("cron-legacy-token")
    finally:
        db2.close()
    assert _snapshot(path) == snap, "second false open must not mutate storage"


def _snapshot(path: Path) -> dict:
    conn = sqlite3.connect(str(path))
    try:
        return {
            "rowids": sorted(_trigram_rowids(conn)),
            "view_sql": _schema_ddl(conn, "messages_fts_trigram_src"),
            "table_exists": _exists(conn, "messages_fts_trigram"),
            "view_exists": _exists(conn, "messages_fts_trigram_src"),
            "marker": _marker(conn),
            "triggers": sorted(
                t
                for t in _trigger_names(conn)
                if t.startswith("messages_fts_trigram")
            ),
            "schema_version": conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()[0],
        }
    finally:
        conn.close()


# ── 3. Read-only false open serves nothing from trigram ─────────────────


def test_v29_false_read_only_open_no_stale_trigram_serving(tmp_path):
    path = tmp_path / "state.db"
    _build_v29_db(path, knob=True)
    if not _fts5_with_trigram(tmp_path / "probe.db"):
        pytest.skip("trigram tokenizer unavailable in this SQLite build")

    writer = SessionDB(db_path=path)
    writer.close()  # writable false open quarantines first

    ro = SessionDB(db_path=path, read_only=True)
    try:
        assert ro._trigram_enabled is False
        assert ro._trigram_available is False
        assert ro.search_messages("root-legacy-token")
    finally:
        ro.close()


# ── 4. Explicit optimize-storage retires the preserved storage ──────────


def test_v29_false_optimize_storage_retires_preserved_storage(tmp_path):
    path = tmp_path / "state.db"
    _build_v29_db(path, knob=True)
    if not _fts5_with_trigram(tmp_path / "probe.db"):
        pytest.skip("trigram tokenizer unavailable in this SQLite build")

    db = SessionDB(db_path=path)
    try:
        assert _exists(db._conn, "messages_fts_trigram")
        assert db.fts_optimize_available() is True
        result = db.optimize_fts_storage(vacuum=False)
        assert result["ok"] is True
        assert not _exists(db._conn, "messages_fts_trigram")
        assert not _exists(db._conn, "messages_fts_trigram_src")
        assert not _marker(db._conn)
        assert db.search_messages("root-legacy-token")
    finally:
        db.close()


# ── 5. Re-enable after quarantine: controlled rebuild under v30 predicate ─


def test_v29_false_then_reenable_rebuilds_v30_predicate(tmp_path):
    path = tmp_path / "state.db"
    fixture = _build_v29_db(path, knob=True)
    if not _fts5_with_trigram(tmp_path / "probe.db"):
        pytest.skip("trigram tokenizer unavailable in this SQLite build")

    db = SessionDB(db_path=path)
    db.close()

    # Re-enable: remove knob → default true; next writable open rebuilds.
    (path.parent / "config.yaml").unlink()
    db2 = SessionDB(db_path=path)
    try:
        assert db2._trigram_available is True
        assert not _marker(db2._conn)
        names = _trigger_names(db2._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig in names
        # v30 predicate excludes the cron session's rows.
        rowids = set(_trigram_rowids(db2._conn))
        cron_rowid = db2._conn.execute(
            "SELECT id FROM messages WHERE content LIKE '%cron-legacy-token%'"
        ).fetchone()[0]
        assert cron_rowid not in rowids
        # Interactive row re-indexed.
        root_rowid = db2._conn.execute(
            "SELECT id FROM messages WHERE content LIKE '%root-legacy-token%'"
        ).fetchone()[0]
        assert root_rowid in rowids
        # Historical rows that belonged to excluded sources are gone.
        assert rowids != set(fixture["historical_rowids"])
        assert db2.search_messages("root-legacy-token")
    finally:
        db2.close()


# ── 6. Ordering: marker precedes trigger drop; partial failure safe ─────


def test_v29_false_marker_precedes_trigger_drop(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    _build_v29_db(path, knob=True)
    if not _fts5_with_trigram(tmp_path / "probe.db"):
        pytest.skip("trigram tokenizer unavailable in this SQLite build")

    import hermes_state_schema

    events: list[tuple[str, bool]] = []
    orig_set_meta = SessionDB.set_meta

    def spy_set_meta(self, key, value, *, cursor=None):
        if key == FTS_TRIGRAM_STALE_KEY and cursor is not None:
            # At the moment the marker is written, triggers may still exist,
            # but nothing has been DROPPED yet — the marker is first.
            events.append(("marker", True))
        return orig_set_meta(self, key, value, cursor=cursor)

    orig_drop = hermes_state_schema.SessionSchemaMixin._quarantine_trigram_schema

    def spy_quarantine(self, cursor):
        names = _trigger_names(cursor)
        dropped = [t for t in _FTS_TRIGRAM_TRIGGERS if t not in names]
        events.append(("quarantine_start", not dropped))
        return orig_drop(self, cursor)

    monkeypatch.setattr(SessionDB, "set_meta", spy_set_meta)
    monkeypatch.setattr(
        hermes_state_schema.SessionSchemaMixin,
        "_quarantine_trigram_schema",
        spy_quarantine,
    )

    db = SessionDB(db_path=path)
    try:
        assert db._trigram_available is False
        assert _marker(db._conn)
    finally:
        db.close()
    marker_events = [e for e in events if e[0] == "marker"]
    quarantine_events = [e for e in events if e[0] == "quarantine_start"]
    assert marker_events, "stale marker must be written during false open"
    assert quarantine_events, "quarantine helper must run"
    assert all(no_dropped for (_, no_dropped) in quarantine_events), (
        "quarantine must start with no trigram triggers already dropped"
    )


def test_v29_false_partial_failure_leaves_no_live_trigram(tmp_path, monkeypatch):
    """If the trigger drop fails midway, trigram must still not be served:
    the marker was already written, so availability stays false."""
    path = tmp_path / "state.db"
    _build_v29_db(path, knob=True)
    if not _fts5_with_trigram(tmp_path / "probe.db"):
        pytest.skip("trigram tokenizer unavailable in this SQLite build")

    import hermes_state_schema

    orig = hermes_state_schema.SessionSchemaMixin._quarantine_trigram_schema
    calls = {"n": 0}

    def flaky(self, cursor):
        calls["n"] += 1
        if calls["n"] == 1:
            # Drop one trigger, then fail — partial teardown.
            cursor.execute("DROP TRIGGER IF EXISTS messages_fts_trigram_insert")
            raise sqlite3.OperationalError("injected partial quarantine failure")
        return orig(self, cursor)

    monkeypatch.setattr(
        hermes_state_schema.SessionSchemaMixin,
        "_quarantine_trigram_schema",
        flaky,
    )
    with pytest.raises(Exception):
        SessionDB(db_path=path)

    db2 = SessionDB(db_path=path)
    try:
        assert db2._trigram_available is False
        assert db2._trigram_enabled is False
        assert _marker(db2._conn)
        # Serving gate: with the stale marker set, search answers via the
        # standard FTS/LIKE fallback — trigram availability stays false.
        assert db2.search_messages("root-legacy-token")
    finally:
        db2.close()
