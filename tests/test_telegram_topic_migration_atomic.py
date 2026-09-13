"""Topic upgrades must publish both tables and their version marker atomically."""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture(params=[1, 2], ids=["v1", "v2"])
def legacy_db(tmp_path, request):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="topic-session", source="telegram", user_id="u1")
    cascade = "ON DELETE CASCADE" if request.param == 2 else ""

    def seed(conn):
        conn.execute("""
            CREATE TABLE telegram_dm_topic_mode (
                chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                activated_at REAL NOT NULL, updated_at REAL NOT NULL,
                has_topics_enabled INTEGER, allows_users_to_create_topics INTEGER,
                capability_checked_at REAL, intro_message_id TEXT, pinned_message_id TEXT
            )
        """)
        conn.execute("""
            INSERT INTO telegram_dm_topic_mode
                (chat_id, user_id, activated_at, updated_at) VALUES ('c1', 'u1', 1, 1)
        """)
        conn.execute(f"""
            CREATE TABLE telegram_dm_topic_bindings (
                chat_id TEXT NOT NULL, thread_id TEXT NOT NULL, user_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id) {cascade},
                managed_mode TEXT NOT NULL DEFAULT 'auto',
                linked_at REAL NOT NULL, updated_at REAL NOT NULL,
                PRIMARY KEY (chat_id, thread_id)
            )
        """)
        conn.execute("""
            INSERT INTO telegram_dm_topic_bindings
            VALUES ('c1', 't1', 'u1', 'k1', 'topic-session', 'auto', 1, 1)
        """)
        conn.execute("""
            CREATE INDEX idx_telegram_dm_topic_bindings_user
            ON telegram_dm_topic_bindings(user_id, chat_id)
        """)
        conn.execute(
            "INSERT INTO state_meta VALUES ('telegram_dm_topic_schema_version', ?)",
            (str(request.param),),
        )

    db._execute_write(seed)
    try:
        yield db
    finally:
        db.close()


def topic_snapshot(conn):
    schema = conn.execute("""
        SELECT type, name, sql FROM sqlite_master
        WHERE tbl_name LIKE 'telegram_dm_topic_%' ORDER BY type, name
    """).fetchall()
    rows = {
        name: [tuple(row) for row in conn.execute(f'SELECT * FROM "{name}"')]
        for kind, name, _ in schema if kind == "table"
    }
    version = conn.execute("""
        SELECT value FROM state_meta WHERE key = 'telegram_dm_topic_schema_version'
    """).fetchone()
    return [tuple(row) for row in schema], rows, tuple(version)


@pytest.mark.parametrize("action,target", [
    (sqlite3.SQLITE_ALTER_TABLE, "telegram_dm_topic_mode_new"),
    (sqlite3.SQLITE_ALTER_TABLE, "telegram_dm_topic_bindings_new"),
    (sqlite3.SQLITE_CREATE_INDEX, "idx_telegram_dm_topic_bindings_session"),
    (sqlite3.SQLITE_INSERT, "state_meta"),
])
def test_failed_migration_restores_legacy_state_and_can_retry(legacy_db, action, target):
    db = legacy_db
    before = topic_snapshot(db._conn)
    denied = []

    def deny_statement(code, arg1, arg2, *_):
        if code == action and target in (arg1, arg2):
            denied.append(target)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    # SQLite's own authorizer faults the real DDL, including executescript on
    # the unfixed path; a proxy intercepting execute() would miss that path.
    db._conn.set_authorizer(deny_statement)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            db.apply_telegram_topic_migration()
    finally:
        db._conn.set_authorizer(None)

    assert denied
    assert topic_snapshot(db._conn) == before
    db.apply_telegram_topic_migration()
    assert db.is_telegram_topic_mode_enabled(chat_id="c1", user_id="u1")
    assert db.get_telegram_topic_binding(chat_id="c1", thread_id="t1")["session_id"] == "topic-session"


def test_migration_keeps_other_writers_out_until_commit(legacy_db):
    db = legacy_db
    contender = SessionDB(db_path=db.db_path)
    contender._conn.execute("PRAGMA busy_timeout=0")
    write_attempts = []

    def compete_at_swap(code, *_):
        if code == sqlite3.SQLITE_ALTER_TABLE:
            try:
                contender._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                write_attempts.append(str(exc))
            else:
                contender._conn.rollback()
                write_attempts.append("writer entered during migration")
        return sqlite3.SQLITE_OK

    db._conn.set_authorizer(compete_at_swap)
    try:
        db.apply_telegram_topic_migration()
        assert write_attempts
        assert all("locked" in error for error in write_attempts)
        # After commit the second connection can both read the complete schema
        # and acquire the write lock, on WAL and rollback-journal databases.
        assert topic_snapshot(contender._conn) == topic_snapshot(db._conn)
        contender._conn.execute("BEGIN IMMEDIATE")
        contender._conn.rollback()
    finally:
        db._conn.set_authorizer(None)
        contender.close()

    assert db.get_meta("telegram_dm_topic_schema_version") == "3"
    assert db.is_telegram_topic_mode_enabled(chat_id="c1", user_id="u1", profile_name="default")
    assert db.get_telegram_topic_binding(chat_id="c1", thread_id="t1", profile_name="default")["session_id"] == "topic-session"
    assert db.get_telegram_topic_binding(chat_id="c1", thread_id="t1", profile_name="other") is None
    after = topic_snapshot(db._conn)
    db.apply_telegram_topic_migration()
    assert topic_snapshot(db._conn) == after
