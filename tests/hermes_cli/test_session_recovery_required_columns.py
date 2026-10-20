"""Required-field recovery and explicit partial-loss accounting contracts.

Current salvage reports destination constraint failures as rejected rows, rather
than prefiltering them into excluded_rows. All databases here are synthetic.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli.session_recovery import _copy_table_salvage, recover_session_database
from hermes_state import SessionDB


@pytest.mark.parametrize("chunk_size", [1, 1000])
def test_nullable_routing_requires_partial_recovery(
    tmp_path: Path, chunk_size: int,
) -> None:
    source = tmp_path / "nullable-routing.db"
    with SessionDB(db_path=source) as db:
        db.create_session("required-fields", "cli")
        db.set_session_title("required-fields", "Synthetic recovery transcript")
    # Seed canonical rows directly: recovery, not live-turn processing, is
    # under test (append_message also bootstraps the agent dependency env).
    with closing(sqlite3.connect(source, isolation_level=None)) as conn:
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            [
                ("required-fields", "user", "Keep this question", 1_700_000_000.0),
                ("required-fields", "assistant", "Keep this answer", 1_700_000_001.0),
            ],
        )

    routing_rows = [
        ("gateway", "valid-before", "{}", 1.0),
        ("gateway", "invalid-null", None, 2.0),
        ("gateway", "valid-after", '{"retained": true}', 3.0),
    ]
    with closing(sqlite3.connect(source, isolation_level=None)) as conn:
        # Model a legacy source accepting NULL where today's destination does not.
        conn.execute("DROP TABLE IF EXISTS gateway_routing")
        conn.execute(
            "CREATE TABLE gateway_routing ("
            "scope TEXT NOT NULL DEFAULT '', session_key TEXT NOT NULL, "
            "entry_json TEXT, updated_at REAL NOT NULL, "
            "PRIMARY KEY (scope, session_key))"
        )
        conn.executemany("INSERT INTO gateway_routing VALUES (?, ?, ?, ?)", routing_rows)
        sessions = conn.execute("SELECT id, title FROM sessions ORDER BY id").fetchall()
        messages = conn.execute(
            "SELECT session_id, role, content FROM messages ORDER BY id"
        ).fetchall()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")

    source_bytes = source.read_bytes()
    strict = recover_session_database(
        source, tmp_path / "strict.db", work_dir=tmp_path, chunk_size=chunk_size,
    )
    # Strict mode returns a failed verification, not a verified lossy recovery.
    assert strict["verified"] is False
    assert strict["complete"] is False
    assert strict["source_unchanged"] is True
    assert strict["copy"]["gateway_routing"]["status"] in {"failed", "partial"}
    assert "NOT NULL" in strict["copy"]["gateway_routing"]["error"]
    assert strict["verification"]["errors"]
    assert source.read_bytes() == source_bytes

    output = tmp_path / "partial.db"
    report = recover_session_database(
        source, output, work_dir=tmp_path, chunk_size=chunk_size, allow_partial=True,
    )
    assert report["verified"] is True
    assert report["complete"] is False
    assert report["partial"] is True
    assert report["source_unchanged"] is True
    assert report["verification"]["errors"] == []
    assert report["verification"]["warnings"]
    routing = report["copy"]["gateway_routing"]
    assert routing["status"] == "partial"
    assert routing["source_rows"] == 3
    assert routing["copied_rows"] == 2
    assert routing["excluded_rows"] == 0
    assert routing["destination_rejected_rows"] == 1
    assert routing["skipped_rowid_span"] == 1
    assert [(item["low"], item["high"]) for item in routing["skipped_rowid_ranges"]] == [(2, 2)]
    assert "destination constraint rejected row" in routing["skipped_rowid_ranges"][0]["error"]

    with closing(sqlite3.connect(output)) as conn:
        assert conn.execute("SELECT id, title FROM sessions ORDER BY id").fetchall() == sessions
        assert conn.execute(
            "SELECT session_id, role, content FROM messages ORDER BY id"
        ).fetchall() == messages
        assert conn.execute(
            "SELECT scope, session_key, entry_json, updated_at "
            "FROM gateway_routing ORDER BY updated_at"
        ).fetchall() == [routing_rows[0], routing_rows[2]]
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert source.read_bytes() == source_bytes
    assert not Path(f"{source}-wal").exists()
    assert not Path(f"{source}-shm").exists()


@pytest.mark.parametrize("has_default", [False, True], ids=["no-default", "with-default"])
def test_salvage_missing_required_column_uses_only_destination_default(
    has_default: bool,
) -> None:
    with (
        closing(sqlite3.connect(":memory:", isolation_level=None)) as source,
        closing(sqlite3.connect(":memory:", isolation_level=None)) as destination,
    ):
        source.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, payload TEXT)")
        rows = [(1, "first"), (2, "second")]
        source.executemany("INSERT INTO records VALUES (?, ?)", rows)
        default_sql = " DEFAULT 'fallback'" if has_default else ""
        destination.execute(
            "CREATE TABLE records (id INTEGER PRIMARY KEY, payload TEXT, "
            f"required_value TEXT NOT NULL{default_sql})"
        )

        report = _copy_table_salvage(
            source, destination, "records", chunk_size=1000,
            progress_cb=None, source_rows=len(rows),
        )

        assert report["source_rows"] == len(rows)
        assert report["excluded_rows"] == 0
        assert source.execute("SELECT * FROM records ORDER BY id").fetchall() == rows
        if has_default:
            assert report["status"] == "complete"
            assert report["copied_rows"] == len(rows)
            assert report["destination_rejected_rows"] == 0
            assert report["skipped_rowid_ranges"] == []
            assert destination.execute("SELECT * FROM records ORDER BY id").fetchall() == [
                (1, "first", "fallback"), (2, "second", "fallback"),
            ]
        else:
            assert report["status"] == "failed"
            assert report["copied_rows"] == 0
            assert report["destination_rejected_rows"] == len(rows)
            assert report["skipped_rowid_span"] == len(rows)
            assert [(item["low"], item["high"]) for item in report["skipped_rowid_ranges"]] == [(1, 2)]
            assert "NOT NULL" in report["skipped_rowid_ranges"][0]["error"]
            assert destination.execute("SELECT * FROM records").fetchall() == []
