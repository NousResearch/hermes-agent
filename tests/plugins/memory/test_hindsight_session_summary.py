import sqlite3

from plugins.memory.hindsight.session_summary import (
    SESSION_SUMMARY_SCHEMA_VERSION,
    SessionSummaryStore,
    SessionSummaryWrite,
)


def _write(**overrides):
    values = {
        "summary_key": "bank/session",
        "identity_scope": "bank",
        "summary_json": {"topics": ["setup"]},
        "summary_text": "setup summary",
        "turn": 4,
        "turn_hash": "turn-hash",
        "last_input_hash": "input-hash-1",
        "parent_summary_key": None,
        "status": "ready",
    }
    values.update(overrides)
    return SessionSummaryWrite(**values)


def test_creates_schema_and_enables_wal_busy_timeout(tmp_path):
    db_path = tmp_path / "summary.sqlite"
    store = SessionSummaryStore(db_path, busy_timeout_ms=1234)
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
    store.close()

    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SESSION_SUMMARY_SCHEMA_VERSION
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(session_summaries)").fetchall()
    }
    assert {
        "summary_json",
        "summary_text",
        "schema_version",
        "version",
        "turn",
        "turn_hash",
        "identity_scope",
        "parent_summary_key",
        "status",
        "last_error",
    } <= columns
    conn.close()


def test_inserts_and_reads_summary_without_raw_history_fields(tmp_path):
    store = SessionSummaryStore(tmp_path / "summary.sqlite")
    result = store.upsert(_write())

    assert result.inserted is True
    assert result.record is not None
    assert result.record.summary_key == "bank/session"
    assert result.record.identity_scope == "bank"
    assert result.record.summary_json == {"topics": ["setup"]}
    assert result.record.summary_text == "setup summary"
    assert result.record.schema_version == SESSION_SUMMARY_SCHEMA_VERSION
    assert result.record.version == 1
    assert result.record.turn == 4
    assert result.record.turn_hash == "turn-hash"
    assert result.record.last_input_hash == "input-hash-1"
    assert "raw" not in result.record.__dataclass_fields__
    store.close()


def test_stale_cas_write_is_dropped_without_overwrite(tmp_path):
    store = SessionSummaryStore(tmp_path / "summary.sqlite")
    store.upsert(_write(last_input_hash="hash-1", summary_text="one"))
    current = store.upsert(
        _write(expected_version=1, last_input_hash="hash-2", summary_text="two")
    )

    stale = store.upsert(
        _write(expected_version=1, last_input_hash="hash-3", summary_text="stale")
    )

    assert current.record is not None
    assert current.record.version == 2
    assert stale.stale is True
    assert stale.updated is False
    record = store.get("bank/session")
    assert record is not None
    assert record.summary_text == "two"
    assert record.version == 2
    store.close()


def test_same_input_hash_is_idempotent(tmp_path):
    store = SessionSummaryStore(tmp_path / "summary.sqlite")
    store.upsert(_write(last_input_hash="same", summary_text="first"))
    again = store.upsert(_write(last_input_hash="same", summary_text="second"))

    assert again.idempotent is True
    assert again.updated is False
    assert again.record is not None
    assert again.record.summary_text == "first"
    assert again.record.version == 1
    store.close()


def test_corrupt_database_is_renamed_and_recreated(tmp_path):
    db_path = tmp_path / "summary.sqlite"
    db_path.write_text("not a sqlite database")

    store = SessionSummaryStore(db_path)
    result = store.upsert(_write())

    assert result.inserted is True
    assert any(path.name.startswith("summary.sqlite.corrupt.") for path in tmp_path.iterdir())
    assert store.get("bank/session") is not None
    store.close()
