"""One corrupt timestamp row must degrade to one '?' cell, never kill a whole listing/export/report.

Real SQLite fixtures: SQLite dynamic typing lets a TEXT cell or a garbage double sit in a REAL
timestamp column (#102399, #102352, #99959). Never monkeypatch the coercion helper.
"""

import argparse
import logging
import sqlite3

import pytest

from agent.insights import InsightsEngine
from hermes_cli.session_export import iter_user_prompt_records
from hermes_cli.session_export_html import generate_multi_session_html_export
from hermes_cli.session_export_md import _iso_timestamp
from hermes_cli.sessions_cmd import _cmd_list
from hermes_state import SessionDB


@pytest.fixture
def corrupt_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    for sid in ("good", "bad-text", "bad-huge"):
        db.create_session(sid, "cli")
        db.append_message(sid, "user", f"hello from {sid}")

    def _corrupt(conn):
        conn.execute("UPDATE sessions SET started_at='not-a-timestamp', last_activity_at='not-a-timestamp' "
                     "WHERE id='bad-text'")
        conn.execute("UPDATE messages SET timestamp='not-a-timestamp' WHERE session_id='bad-text'")
        conn.execute("UPDATE sessions SET started_at=8.4e252 WHERE id='bad-huge'")
        conn.execute("UPDATE messages SET timestamp=1e30 WHERE session_id='bad-huge'")

    db._execute_write(_corrupt)
    yield db
    db.close()


def test_list_export_and_insights_survive_corrupt_timestamp_rows(corrupt_db, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_cli.timefmt"):
        _cmd_list(corrupt_db, argparse.Namespace(limit=20, source=None, all=False, workspace=None))
        listing = capsys.readouterr().out
        exported = corrupt_db.export_all()
        records = list(iter_user_prompt_records(exported))
        html = generate_multi_session_html_export(exported)
        md_stamps = [_iso_timestamp(s["started_at"]) for s in exported]
        report = InsightsEngine(corrupt_db).generate(days=365_000)

    # Every session is still present on every surface; the bad cells degrade, the good one renders.
    assert all(sid in listing for sid in ("good", "bad-text", "bad-huge")) and "?" in listing
    assert {r["session_id"] for r in records} == {"good", "bad-text", "bad-huge"}
    assert "not-a-timestamp" not in html and html.count("N/A") >= 2
    assert md_stamps.count("not-a-timestamp") == 1 and any(stamp.endswith("Z") for stamp in md_stamps)
    assert report["overview"]["total_sessions"] == 3
    # The warning names the session so the corrupt row can be found.
    assert any("bad-huge" in rec.getMessage() for rec in caplog.records)


def test_writers_never_persist_an_out_of_window_timestamp(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "a", timestamp=8.4e252)
        db.append_messages_batch("s", [{"role": "assistant", "content": "b", "timestamp": "not-a-timestamp"}])
        stored = [row["timestamp"] for row in db.get_messages("s")]
    finally:
        db.close()
    assert len(stored) == 2 and all(isinstance(ts, float) and 0 < ts < 4.2e9 for ts in stored)


def test_bulk_delete_and_prune_stay_below_sqlite_variable_limit(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        old = 1_600_000_000.0
        ids = [f"cron_job_{i}" for i in range(1200)]

        def seed(conn):
            conn.executemany("INSERT INTO sessions (id, source, started_at, ended_at, message_count) "
                             "VALUES (?, 'cron', ?, ?, 1)", [(sid, old, old) for sid in ids])
            conn.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'x', ?)",
                             [(sid, old) for sid in ids])

        db._execute_write(seed)
        db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)  # the legacy ceiling, deterministic
        assert db.prune_sessions(older_than_days=14, source="cron") == 1200
        db._execute_write(seed)
        db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        assert db.delete_sessions(ids) == 1200
        assert db._read_one("SELECT COUNT(*) FROM messages WHERE session_id NOT IN (SELECT id FROM sessions)")[0] == 0
    finally:
        db.close()
