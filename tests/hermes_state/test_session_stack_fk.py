"""session_stack foreign keys must cascade on session delete."""

from hermes_state import SessionDB


def _install_legacy_session_stack(conn):
    conn.executescript(
        """
        DROP TABLE IF EXISTS session_stack;
        CREATE TABLE session_stack (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            parent_session_id TEXT NOT NULL REFERENCES sessions(id),
            side_session_id TEXT NOT NULL REFERENCES sessions(id),
            title TEXT,
            pushed_at REAL NOT NULL,
            popped_at REAL,
            status TEXT NOT NULL DEFAULT 'active'
        );
        """
    )


def test_legacy_session_stack_fk_is_rebuilt_with_cascade(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="cli")
    db.create_session("side", source="cli")

    conn = db._conn
    _install_legacy_session_stack(conn)
    conn.execute(
        "INSERT INTO session_stack (source, parent_session_id, side_session_id, pushed_at, status) "
        "VALUES ('cli', 'parent', 'side', 1.0, 'active')"
    )
    conn.commit()

    legacy = conn.execute("PRAGMA foreign_key_list('session_stack')").fetchall()
    assert legacy and all((str(row[6]) or "").upper() != "CASCADE" for row in legacy)

    db._init_schema()

    migrated = conn.execute("PRAGMA foreign_key_list('session_stack')").fetchall()
    assert len(migrated) == 2
    assert all((str(row[6]) or "").upper() == "CASCADE" for row in migrated)
    assert conn.execute("SELECT COUNT(*) FROM session_stack").fetchone()[0] == 1


def test_deleting_referenced_session_cascades_stack_row(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="cli")
    db.create_session("side", source="cli")
    db.push_side_session("cli", "parent", "side", title="topic")

    conn = db._conn
    assert conn.execute("SELECT COUNT(*) FROM session_stack").fetchone()[0] == 1

    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DELETE FROM sessions WHERE id = 'parent'")
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM session_stack").fetchone()[0] == 0


def test_reconcile_recovers_from_leftover_session_stack_new(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="cli")
    db.create_session("side", source="cli")

    conn = db._conn
    _install_legacy_session_stack(conn)
    conn.execute(
        "INSERT INTO session_stack (source, parent_session_id, side_session_id, pushed_at, status) "
        "VALUES ('cli', 'parent', 'side', 1.0, 'active')"
    )
    conn.execute("CREATE TABLE session_stack_new (bogus INTEGER)")
    conn.commit()

    legacy = conn.execute("PRAGMA foreign_key_list('session_stack')").fetchall()
    assert legacy and all((str(row[6]) or "").upper() != "CASCADE" for row in legacy)

    db._init_schema()

    migrated = conn.execute("PRAGMA foreign_key_list('session_stack')").fetchall()
    assert len(migrated) == 2
    assert all((str(row[6]) or "").upper() == "CASCADE" for row in migrated)
    assert conn.execute("SELECT COUNT(*) FROM session_stack").fetchone()[0] == 1
    leftover = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='session_stack_new'"
    ).fetchall()
    assert leftover == []
