"""Behavior gates for the restored local exact-accounting reader."""
import sqlite3
import time

from agent.usage_window import aggregate_model_usage, aggregate_window_usage, get_usage_window_coverage, get_window_usage_rows
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION, USAGE_EVENTS_COVERAGE_START_KEY
from hermes_cli.session_recovery import recover_session_database


def test_legacy_fallback_and_exact_events_do_not_overlap(tmp_path):
    db = SessionDB(db_path=tmp_path / "usage.db")
    try:
        db.create_session("legacy", "cli", model="legacy-model")
        db._conn.execute("UPDATE sessions SET input_tokens=100, api_call_count=1 WHERE id='legacy'")
        db.create_session("exact", "telegram", model="exact-model")
        db.update_token_counts("exact", input_tokens=20, api_call_count=1)
        db.record_auxiliary_usage("exact", "vision", model="vision-model", input_tokens=5)
        cutoff = time.time() - 100
        rows = get_window_usage_rows(db._conn, cutoff)
        _, _, totals = aggregate_window_usage(rows)
        assert totals["total_input"] == 125
        assert totals["total_sessions"] == 2
        assert get_usage_window_coverage(db._conn, cutoff, usage_rows=rows)["legacy_fallback_used"] is True
        assert {row["source"] for row in rows} == {"cli", "telegram"}
        scoped = get_window_usage_rows(db._conn, cutoff, "telegram")
        assert {row["session_id"] for row in scoped} == {"exact"}
        assert {row["source"] for row in scoped} == {"telegram"}
        assert sum(row["input_tokens"] for row in scoped) == 25
    finally:
        db.close()


def test_model_aggregation_preserves_auxiliary_task_dimension():
    rows = [
        {
            "session_id": "s1", "model": "shared-model", "billing_provider": "provider", "task": "",
            "input_tokens": 10, "output_tokens": 2, "recorded_at": 1,
        },
        {
            "session_id": "s1", "model": "shared-model", "billing_provider": "provider", "task": "compression",
            "input_tokens": 3, "output_tokens": 1, "recorded_at": 2,
        },
    ]

    result = aggregate_model_usage(rows, [{"model": "shared-model", "billing_provider": "provider", "tool_calls": 4}])
    by_task = {row["task"]: row for row in result}
    assert set(by_task) == {"", "compression"}
    assert by_task[""]["tool_calls"] == 4
    assert by_task["compression"]["tool_calls"] == 0
    assert by_task["compression"]["input_tokens"] == 3
    assert by_task["compression"]["avg_tokens_per_session"] == 0


def test_preledger_database_is_readable_without_migration(tmp_path):
    path = tmp_path / "legacy.db"
    db = SessionDB(db_path=path)
    db.create_session("legacy", "cli", model="legacy-model")
    db._conn.execute("UPDATE sessions SET input_tokens=7 WHERE id='legacy'")
    db._conn.execute("DROP TABLE session_model_usage_events")
    db._conn.commit()
    db.close()
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = get_window_usage_rows(conn, time.time() - 100)
        assert sum(row["input_tokens"] for row in rows) == 7
        assert get_usage_window_coverage(conn, time.time() - 100, usage_rows=rows)["exact_events_available"] is False
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_model_usage_events'").fetchone() is None
    finally:
        conn.close()


def test_recovery_invalidates_coverage_when_event_table_is_missing(tmp_path):
    source = tmp_path / "source-missing-events.db"
    output = tmp_path / "recovered-missing-events.db"
    db = SessionDB(db_path=source)
    try:
        db.create_session("legacy", "cli")
        db.update_token_counts("legacy", input_tokens=3)
        db._conn.execute(
            "UPDATE state_meta SET value='1234567890' WHERE key=?",
            (USAGE_EVENTS_COVERAGE_START_KEY,),
        )
        db._conn.execute("DROP TABLE session_model_usage_events")
        db._conn.commit()
    finally:
        db.close()

    result = recover_session_database(source, output, work_dir=tmp_path)
    assert result["verified"] is True
    historical_cutoff = 1234567890.0
    conn = sqlite3.connect(output)
    try:
        coverage = get_usage_window_coverage(conn, historical_cutoff)
        marker = conn.execute(
            "SELECT value FROM state_meta WHERE key=?",
            (USAGE_EVENTS_COVERAGE_START_KEY,),
        ).fetchone()
    finally:
        conn.close()
    assert marker is not None
    assert coverage["coverage_start"] is not None
    assert coverage["coverage_start"] > historical_cutoff
    assert coverage["window_complete"] is False
    for _ in range(2):
        reopened = SessionDB(db_path=output)
        try:
            after = get_usage_window_coverage(reopened._conn, historical_cutoff)
        finally:
            reopened.close()
        assert after["coverage_start"] == coverage["coverage_start"]
        assert after["window_complete"] is False

    post = SessionDB(db_path=output)
    try:
        post.create_session("post-recovery", "cli")
        post.update_token_counts("post-recovery", input_tokens=2, api_call_count=1)
        fresh = get_usage_window_coverage(post._conn, coverage["coverage_start"])
    finally:
        post.close()
    assert fresh["window_complete"] is True
    assert fresh["coverage_start"] == coverage["coverage_start"]


def test_recovery_preserves_coverage_and_official_schema_version(tmp_path):
    source, output = tmp_path / "source.db", tmp_path / "recovered.db"
    db = SessionDB(db_path=source)
    db.create_session("accounted", "cli")
    db.update_token_counts("accounted", input_tokens=3)
    db._conn.execute("UPDATE state_meta SET value='1234567890' WHERE key=?", (USAGE_EVENTS_COVERAGE_START_KEY,))
    db._conn.commit()
    db.close()
    result = recover_session_database(source, output, work_dir=tmp_path)
    assert result["verified"]
    conn = sqlite3.connect(output)
    try:
        assert conn.execute("SELECT value FROM state_meta WHERE key=?", (USAGE_EVENTS_COVERAGE_START_KEY,)).fetchone()[0] == "1234567890"
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        assert conn.execute("SELECT SUM(input_tokens) FROM session_model_usage_events").fetchone()[0] == 3
    finally:
        conn.close()
    for _ in range(2):
        reopened = SessionDB(db_path=output)
        try:
            preserved = reopened._conn.execute(
                "SELECT value FROM state_meta WHERE key=?",
                (USAGE_EVENTS_COVERAGE_START_KEY,),
            ).fetchone()
        finally:
            reopened.close()
        assert preserved[0] == "1234567890"
