"""Config-backed disable/reenable behavior for the trigram FTS index."""

import sqlite3

import pytest

from hermes_state import SessionDB


def _sqlite_supports_trigram() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(content, tokenize='trigram')"
        )
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


pytestmark = pytest.mark.skipif(
    not _sqlite_supports_trigram(),
    reason="SQLite FTS5 trigram tokenizer is not available",
)


@pytest.fixture(autouse=True)
def _clear_disable_env(monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_FTS_TRIGRAM", raising=False)


def _write_config(home, *, disabled: bool) -> None:
    home.mkdir(parents=True, exist_ok=True)
    home.joinpath("config.yaml").write_text(
        "sessions:\n"
        f"  disable_fts_trigram: {'true' if disabled else 'false'}\n",
        encoding="utf-8",
    )


def _objects(conn):
    rows = conn.execute(
        "SELECT type, name FROM sqlite_master "
        "WHERE name = 'messages_fts_trigram' "
        "OR name IN ("
        "'messages_fts_trigram_insert',"
        "'messages_fts_trigram_delete',"
        "'messages_fts_trigram_update'"
        ")"
    ).fetchall()
    return {(row[0], row[1]) for row in rows}


def _seed_message(db: SessionDB, content: str) -> None:
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="user", content=content)


class _DropTrigramFailsCursor(sqlite3.Cursor):
    def execute(self, sql, parameters=()):
        if sql.strip() == "DROP TABLE IF EXISTS messages_fts_trigram":
            raise sqlite3.OperationalError("no such module: fts5")
        return super().execute(sql, parameters)


class _DropTrigramFailsConnection(sqlite3.Connection):
    def cursor(self, factory=None):
        return super().cursor(factory or _DropTrigramFailsCursor)


def test_default_config_creates_trigram_and_triggers_write(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._fts_trigram_disabled is False
        assert db._trigram_available is True
        _seed_message(db, "hello unicode 大别山项目")

        assert (
            db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            == 1
        )
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_trigram"
            ).fetchone()[0]
            == 1
        )
        assert ("table", "messages_fts_trigram") in _objects(db._conn)
        assert {
            ("trigger", "messages_fts_trigram_insert"),
            ("trigger", "messages_fts_trigram_delete"),
            ("trigger", "messages_fts_trigram_update"),
        }.issubset(_objects(db._conn))
    finally:
        db.close()


def test_disabled_fresh_v23_db_keeps_compact_trigram_and_search_works(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home, disabled=True)

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._fts_enabled is True
        assert db._fts_trigram_disabled is True
        assert db._trigram_available is True
        assert ("table", "messages_fts_trigram") in _objects(db._conn)

        _seed_message(db, "hello unicode 大别山项目")
        assert len(db.search_messages("hello")) == 1
        assert len(db.search_messages("大别山项目")) == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts_trigram"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_explicit_profile_v23_db_keeps_compact_trigram_despite_legacy_flag(
    tmp_path, monkeypatch
):
    process_home = tmp_path / "profiles" / "gpt"
    target_home = tmp_path / "profiles" / "coding"
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    _write_config(process_home, disabled=False)
    _write_config(target_home, disabled=True)

    db = SessionDB(db_path=target_home / "state.db")
    try:
        assert db._fts_trigram_disabled is True
        assert db._trigram_available is True
        assert ("table", "messages_fts_trigram") in _objects(db._conn)
    finally:
        db.close()


def test_disabled_existing_v23_trigram_is_not_dropped_or_backfilled(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    seeded = SessionDB(db_path=db_path)
    try:
        _seed_message(seeded, "before disable 大别山项目")
        assert ("table", "messages_fts_trigram") in _objects(seeded._conn)
    finally:
        seeded.close()

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home, disabled=True)

    traced_sql = []
    real_connect = sqlite3.connect

    def connect_with_trace(*args, **kwargs):
        conn = real_connect(*args, **kwargs)

        def trace(sql):
            text = " ".join(str(sql).split())
            if "messages_fts_trigram" in text:
                traced_sql.append(text)

        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr("hermes_state.sqlite3.connect", connect_with_trace)

    db = SessionDB(db_path=db_path)
    try:
        assert db._fts_trigram_disabled is True
        assert db._trigram_available is True
        assert ("table", "messages_fts_trigram") in _objects(db._conn)
        assert not any(
            "messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')" in sql
            for sql in traced_sql
        )

        db.append_message("s1", role="assistant", content="after disable")
        assert len(db.search_messages("after")) == 1
    finally:
        db.close()


def test_disabled_existing_v23_trigram_is_preserved_when_fts5_probe_fails(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    seeded = SessionDB(db_path=db_path)
    try:
        _seed_message(seeded, "before no fts5 runtime")
        assert ("table", "messages_fts_trigram") in _objects(seeded._conn)
    finally:
        seeded.close()

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home, disabled=True)
    monkeypatch.setattr(
        SessionDB,
        "_sqlite_supports_fts5",
        lambda self, cursor: False,
    )

    db = SessionDB(db_path=db_path)
    try:
        assert db._fts_enabled is False
        assert _objects(db._conn) == {("table", "messages_fts_trigram")}
        db.append_message("s1", role="assistant", content="writes still work")
    finally:
        db.close()


def test_disabled_existing_v23_trigram_never_attempts_legacy_drop(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    seeded = SessionDB(db_path=db_path)
    try:
        _seed_message(seeded, "before drop failure")
        assert ("table", "messages_fts_trigram") in _objects(seeded._conn)
    finally:
        seeded.close()

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home, disabled=True)
    monkeypatch.setattr(
        SessionDB,
        "_sqlite_supports_fts5",
        lambda self, cursor: False,
    )
    real_connect = sqlite3.connect

    def connect_with_drop_failure(*args, **kwargs):
        kwargs["factory"] = _DropTrigramFailsConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("hermes_state.sqlite3.connect", connect_with_drop_failure)

    db = SessionDB(db_path=db_path)
    try:
        assert ("table", "messages_fts_trigram") in _objects(db._conn)
        remaining_shadow_rows = db._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE name LIKE 'messages_fts_trigram_%'"
        ).fetchone()[0]
        assert remaining_shadow_rows > 0
    finally:
        db.close()


def test_legacy_disable_flag_does_not_remove_v23_trigram(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    db_path = tmp_path / "state.db"

    _write_config(home, disabled=False)
    seeded = SessionDB(db_path=db_path)
    try:
        _seed_message(seeded, "reenable needle 大别山项目")
    finally:
        seeded.close()

    _write_config(home, disabled=True)
    disabled = SessionDB(db_path=db_path)
    try:
        assert ("table", "messages_fts_trigram") in _objects(disabled._conn)
    finally:
        disabled.close()

    _write_config(home, disabled=False)
    restored = SessionDB(db_path=db_path)
    try:
        assert restored._trigram_available is True
        assert ("table", "messages_fts_trigram") in _objects(restored._conn)
        assert (
            restored._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_trigram"
            ).fetchone()[0]
            == 1
        )
        assert len(restored.search_messages("大别山项目")) == 1
    finally:
        restored.close()
