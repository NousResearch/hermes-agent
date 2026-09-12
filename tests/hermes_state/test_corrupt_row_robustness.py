"""One corrupt timestamp row must degrade to one '?' cell, never kill a whole listing/export/report.

Real SQLite fixtures: SQLite dynamic typing lets a TEXT cell or a garbage double sit in a REAL
timestamp column (#102399, #102352, #99959). Never monkeypatch the coercion helper.
"""

import argparse
import hashlib
import logging
import sqlite3

import pytest

from agent.insights import InsightsEngine
from hermes_cli.session_export import iter_user_prompt_records
from hermes_cli.session_export_html import generate_multi_session_html_export
from hermes_cli.session_export_md import _iso_timestamp
from hermes_cli.sessions_cmd import _cmd_list
from hermes_state import SessionDB, _corruption_warned_fingerprints


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


def test_corrupt_prompt_row_degrades_instead_of_killing_session_queries(tmp_path, caplog):
    """One truncated-UTF-8 system_prompts row must degrade to U+FFFD in that one cell, never
    abort every session-list/load query (#109450: one bad row made the desktop session panel
    unable to list or open ANY session)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        good_prompt = "You are Hermes ✓ 中文 round-trip"
        db.create_session("good", "cli", system_prompt=good_prompt)
        db.create_session("bad", "cli", system_prompt="You are Hermes truncated tail")

        def _corrupt(conn):
            row = conn.execute(
                "SELECT hash FROM system_prompts WHERE prompt LIKE '%truncated tail%'"
            ).fetchone()
            # A mid-write process death left the 3-byte ✓ (e2 9c 93) missing its last byte;
            # CAST keeps the bad value in TEXT storage so the strict decode path is exercised.
            conn.execute(
                "UPDATE system_prompts SET prompt = CAST(? AS TEXT) WHERE hash = ?",
                (b"You are Hermes \xe2\x9c", row["hash"]),
            )

        db._execute_write(_corrupt)
        kind = db._conn.execute(
            "SELECT typeof(sp.prompt) FROM system_prompts sp"
            " JOIN sessions s ON s.system_prompt_hash = sp.hash WHERE s.id = 'bad'"
        ).fetchone()[0]
        assert kind == "text"

        _corruption_warned_fingerprints.clear()  # module-level dedup: start from a known state
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            rich = {s["id"]: s["system_prompt"] for s in db.list_sessions_rich()}
            loaded_bad = db.get_session("bad")["system_prompt"]
            searched = {s["id"]: s["system_prompt"] for s in db.search_sessions()}

        # Every session still loads on every surface; only the corrupt cell degrades.
        assert set(rich) == {"good", "bad"}
        assert rich["good"] == good_prompt  # valid multi-byte text round-trips exactly
        assert rich["bad"] == "You are Hermes \ufffd"
        assert loaded_bad == rich["bad"]
        assert searched == rich
        # The warning is content-free: sha256 fingerprint (no raw bytes — cells can hold secrets),
        # and it fires once per distinct bad value — a second read pass stays silent.
        fingerprint = hashlib.sha256(b"You are Hermes \xe2\x9c").hexdigest()[:16]
        warned = [r for r in caplog.records if "degraded to U+FFFD" in r.getMessage()]
        assert warned and fingerprint in warned[0].getMessage()
        assert not any("\\xe2\\x9c" in r.getMessage() for r in caplog.records)
        caplog.clear()
        db.list_sessions_rich()
        assert not [r for r in caplog.records if "degraded to U+FFFD" in r.getMessage()]
    finally:
        db.close()


def test_corrupt_model_config_aborts_mutation_instead_of_rewriting(tmp_path):
    """A malformed model_config cell must abort a read-modify-write (the writer connection stays
    strict), never tolerate-decode to a JSON parse failure that turns into {} and lets the patch
    overwrite the original field (#109450 review: `{"keep":"yes"}\\xff` became only `{"new":1}`)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bad", "cli")
        raw = b'{"keep": "yes"}\xff'

        def _corrupt(conn):
            conn.execute(
                "UPDATE sessions SET model_config = CAST(? AS TEXT) WHERE id = 'bad'", (raw,)
            )

        db._execute_write(_corrupt)
        assert db._conn.execute(
            "SELECT typeof(model_config) FROM sessions WHERE id = 'bad'"
        ).fetchone()[0] == "text"

        with pytest.raises(sqlite3.OperationalError):
            db.patch_session_model_config("bad", {"new": 1})

        # Fail closed: the mutation aborted, the malformed cell is byte-identical, "keep" survived.
        stored = db._read_one(
            "SELECT CAST(model_config AS BLOB) FROM sessions WHERE id = 'bad'"
        )[0]
        assert bytes(stored) == raw
    finally:
        db.close()


def test_blob_stored_prompt_reads_back_as_bytes(tmp_path):
    """BLOB storage bypasses text_factory by sqlite3 contract: a value stored as BLOB reads back
    as ``bytes`` (pre-existing behavior, unchanged by this fix), unlike malformed TEXT which
    degrades to U+FFFD on read."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bad", "cli", system_prompt="placeholder tail")
        raw = b"You are Hermes \xe2\x9c"

        def _store_blob(conn):
            row = conn.execute(
                "SELECT hash FROM system_prompts WHERE prompt LIKE '%placeholder tail%'"
            ).fetchone()
            # No CAST: binding bytes stores a BLOB, which text_factory never sees.
            conn.execute("UPDATE system_prompts SET prompt = ? WHERE hash = ?", (raw, row["hash"]))

        db._execute_write(_store_blob)
        assert db._conn.execute(
            "SELECT typeof(sp.prompt) FROM system_prompts sp"
            " JOIN sessions s ON s.system_prompt_hash = sp.hash WHERE s.id = 'bad'"
        ).fetchone()[0] == "blob"
        assert db.get_session("bad")["system_prompt"] == raw  # bytes in, bytes out
    finally:
        db.close()


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
