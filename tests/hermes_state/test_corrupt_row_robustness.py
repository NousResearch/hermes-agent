"""One corrupt timestamp row must degrade to one '?' cell, never kill a whole listing/export/report.

Real SQLite fixtures: SQLite dynamic typing lets a TEXT cell or a garbage double sit in a REAL
timestamp column (#102399, #102352, #99959). Never monkeypatch the coercion helper.
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import time

import pytest

from agent.insights import InsightsEngine
from hermes_cli.session_export import iter_user_prompt_records
from hermes_cli.session_export_html import generate_multi_session_html_export
from hermes_cli.session_export_md import _iso_timestamp
from hermes_cli.sessions_cmd import _cmd_list
from hermes_state import SessionDB, _corruption_warned_fingerprints
from hermes_state_common import (
    _CORRUPTION_WARN_FINGERPRINT_CAP, warn_corrupt_cell as _warn_corrupt_cell)


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


def test_blob_stored_cells_degrade_to_str_never_escape_as_bytes(tmp_path, caplog):
    """BLOB storage bypasses text_factory by sqlite3 contract, but NO public read surface may
    hand out bytes: a BLOB system_prompt/message-content decodes to str (U+FFFD on undecodable
    bytes), so session dicts and message dicts stay JSON-serializable for BOTH storage classes
    (#109465 review: bytes escaped the public session API and json.dumps(message) raised
    TypeError)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bad", "cli", system_prompt="placeholder tail")
        db.append_message("bad", "user", "msg placeholder tail")
        undecodable = b"You are Hermes \xe2\x9c"

        def _store_blobs(conn):
            row = conn.execute(
                "SELECT hash FROM system_prompts WHERE prompt LIKE '%placeholder tail%'"
            ).fetchone()
            # No CAST: binding bytes stores a BLOB, which text_factory never sees.
            conn.execute("UPDATE system_prompts SET prompt = ? WHERE hash = ?", (undecodable, row["hash"]))
            conn.execute("UPDATE messages SET content = ? WHERE session_id = 'bad'", (undecodable,))

        db._execute_write(_store_blobs)
        assert db._conn.execute(
            "SELECT typeof(sp.prompt) FROM system_prompts sp"
            " JOIN sessions s ON s.system_prompt_hash = sp.hash WHERE s.id = 'bad'"
        ).fetchone()[0] == "blob"

        _corruption_warned_fingerprints.clear()  # module-level dedup: start from a known state
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            session = db.get_session("bad")
            messages = db.get_messages("bad")

        # Both storage classes degrade identically: str with U+FFFD, never bytes.
        assert session["system_prompt"] == "You are Hermes \ufffd"
        assert isinstance(messages[0]["content"], str)
        assert messages[0]["content"] == "You are Hermes \ufffd"
        # Stable serializable output contract: json.dumps works on every public dict.
        assert json.dumps(session) and json.dumps(messages)
        # The BLOB degrade path shares the content-free fingerprint warning with text_factory.
        fingerprint = hashlib.sha256(undecodable).hexdigest()[:16]
        warned = [r for r in caplog.records if "degraded to U+FFFD" in r.getMessage()]
        assert fingerprint in warned[0].getMessage()
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


def test_broken_json_model_config_aborts_mutation_instead_of_rewriting(tmp_path):
    """A syntactically malformed but valid-UTF-8 model_config cell aborts the patch too, not
    just undecodable bytes: the strict decode alone left `{"keep":"yes" BROKEN` parsing away
    to {} and letting the patch overwrite the stored field (#109465 review follow-up)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bad", "cli")
        raw = '{"keep": "yes" BROKEN'  # decodes fine, parses to nothing

        def _corrupt(conn):
            conn.execute("UPDATE sessions SET model_config = ? WHERE id = 'bad'", (raw,))

        db._execute_write(_corrupt)

        with pytest.raises(sqlite3.OperationalError):
            db.patch_session_model_config("bad", {"new": 1})

        stored = db._read_one("SELECT CAST(model_config AS BLOB) FROM sessions WHERE id = 'bad'")[0]
        assert bytes(stored) == raw.encode("utf-8")  # fail closed: byte-identical, "keep" survived
    finally:
        db.close()


def test_reaction_write_never_drops_unrelated_display_metadata(tmp_path):
    """set_message_reaction is a display_metadata read-modify-write seam: a malformed cell
    (undecodable UTF-8, broken JSON, or a non-object value) aborts the reaction write with
    OperationalError and the cell is preserved byte-identical; a healthy cell keeps its
    unrelated keys through the rewrite (#109465 review: the {}-fallback dropped them)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "hello")
        row_id = db.latest_message_row_id("s", role="user")

        def _undecodable(conn):
            conn.execute("UPDATE messages SET display_metadata = CAST(? AS TEXT) WHERE id = ?",
                         (b'{"task_count": 3}\xff', row_id))

        def _broken_json(conn):
            conn.execute("UPDATE messages SET display_metadata = ? WHERE id = ?",
                         ('{"task_count": 3 BROKEN', row_id))

        def _non_object(conn):
            conn.execute("UPDATE messages SET display_metadata = ? WHERE id = ?",
                         ('["not", "an", "object"]', row_id))

        for label, corrupt, expected in (
            ("undecodable UTF-8", _undecodable, b'{"task_count": 3}\xff'),
            ("broken JSON", _broken_json, b'{"task_count": 3 BROKEN'),
            ("non-object JSON", _non_object, b'["not", "an", "object"]'),
        ):
            db._execute_write(corrupt)
            with pytest.raises(sqlite3.OperationalError):
                db.set_message_reaction("s", row_id, "\U0001f44d", author="user")
            stored = db._read_one(
                "SELECT CAST(display_metadata AS BLOB) FROM messages WHERE id = ?", (row_id,))[0]
            assert bytes(stored) == expected, f"{label}: cell must stay byte-identical, not rewritten"

        # Healthy cell: the reaction merges and the unrelated key survives the rewrite.
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET display_metadata = ? WHERE id = ?",
            ('{"task_count": 3}', row_id)))
        reactions = db.set_message_reaction("s", row_id, "\U0001f44d", author="user")
        assert [r["emoji"] for r in reactions] == ["\U0001f44d"]
        meta = json.loads(db._read_one(
            "SELECT display_metadata FROM messages WHERE id = ?", (row_id,))[0])
        assert meta["task_count"] == 3  # unrelated metadata preserved through the seam
    finally:
        db.close()


def test_corruption_warning_dedupe_is_bounded_and_thread_safe(caplog):
    """The per-process fingerprint dedupe is bounded and thread-safe: N distinct malformed cells
    keep at most _CORRUPTION_WARN_FINGERPRINT_CAP entries (#109465 review: an unbounded plain
    set retained one fingerprint per distinct bad cell and check/add raced across reader
    threads); the oldest entry falls out at the cap and may warn again on a later read."""
    import threading

    _corruption_warned_fingerprints.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        for i in range(_CORRUPTION_WARN_FINGERPRINT_CAP + 50):
            _warn_corrupt_cell(b"distinct-bad-cell-%d-\xff" % i)
    assert len(_corruption_warned_fingerprints) == _CORRUPTION_WARN_FINGERPRINT_CAP

    # The oldest fingerprint fell out (warns again on a later read); a retained one stays silent.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        _warn_corrupt_cell(b"distinct-bad-cell-0-\xff")
        _warn_corrupt_cell(b"distinct-bad-cell-%d-\xff" % (_CORRUPTION_WARN_FINGERPRINT_CAP + 49))
    warns = [r for r in caplog.records if "degraded to U+FFFD" in r.getMessage()]
    assert len(warns) == 1

    # Concurrent warners: no exception, and the set never exceeds the cap.
    errors: list = []

    def _hammer(worker: int) -> None:
        try:
            for i in range(200):
                _warn_corrupt_cell(b"worker-%d-cell-%d-\xff" % (worker, i))
        except Exception as exc:  # pragma: no cover - only on a locking regression
            errors.append(exc)

    _corruption_warned_fingerprints.clear()
    threads = [threading.Thread(target=_hammer, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(_corruption_warned_fingerprints) <= _CORRUPTION_WARN_FINGERPRINT_CAP


def test_take_unseen_reactions_fails_closed_on_undecodable_metadata(tmp_path):
    """The take_unseen_reactions read-modify-write seam must abort byte-identically on an
    undecodable display_metadata cell — for BOTH storage classes — instead of parsing the
    tolerant U+FFFD text, marking the reaction seen, and rewriting the row's private fields
    as replacement soup (#109465 review, completeness gap 1)."""
    for storage in ("blob", "text"):
        db = SessionDB(db_path=tmp_path / f"state-{storage}.db")
        try:
            db.create_session("s", "cli")
            db.append_message("s", "user", "react to me")
            row_id = db.latest_message_row_id("s", role="user")
            db.set_message_reaction("s", row_id, "\U0001f44d", author="user")
            good_meta = db._conn.execute(
                "SELECT CAST(display_metadata AS BLOB) FROM messages WHERE id = ?", (row_id,)
            ).fetchone()[0]
            assert json.loads(good_meta.decode("utf-8"))["reactions"][0]["emoji"] == "\U0001f44d"

            # One private field carries undecodable bytes; the reaction itself stays valid.
            undecodable_meta = json.dumps({
                "private": "secret", "reactions": [
                    {"emoji": "\U0001f44d", "author": "user", "at": 1.0, "seen": False}],
            }).encode("utf-8").replace(b"secret", b"secret\xe2\x9c")

            def _damage(conn):
                if storage == "blob":
                    conn.execute("UPDATE messages SET display_metadata = ? WHERE id = ?",
                                 (undecodable_meta, row_id))
                else:  # TEXT storage: the tolerant text_factory would hide the raw bytes on read
                    conn.execute("UPDATE messages SET display_metadata = CAST(? AS TEXT) WHERE id = ?",
                                 (undecodable_meta, row_id))

            db._execute_write(_damage)
            stored = db._conn.execute(
                "SELECT CAST(display_metadata AS BLOB) FROM messages WHERE id = ?", (row_id,)
            ).fetchone()[0]
            assert stored == undecodable_meta

            # Fail closed: the whole take aborts; the stored cell is byte-identical afterwards.
            with pytest.raises(sqlite3.OperationalError, match="not valid UTF-8"):
                db.take_unseen_reactions("s", author="user")
            assert db._conn.execute(
                "SELECT CAST(display_metadata AS BLOB) FROM messages WHERE id = ?", (row_id,)
            ).fetchone()[0] == undecodable_meta

            # With the damaged cell healed, the same take surfaces the reaction exactly once.
            db._execute_write(lambda conn: conn.execute(
                "UPDATE messages SET display_metadata = ? WHERE id = ?", (good_meta, row_id)))
            assert db.take_unseen_reactions("s", author="user") == [{
                "row_id": row_id, "role": "user", "emoji": "\U0001f44d", "text": "react to me"}]
            assert db.take_unseen_reactions("s", author="user") == []
        finally:
            db.close()


def test_blob_stored_title_and_reasoning_stay_serializable(tmp_path):
    """BLOB-stored sessions.title and messages.reasoning must degrade to str in the public
    dicts too, not just system_prompt/content: json.dumps(get_session(...)) and
    json.dumps(get_messages(...)) keep working for BOTH storage classes (#109465 review,
    completeness gap 2)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "assistant", "answer placeholder head", reasoning="think placeholder head")
        undecodable = b"why so serious \xe2\x9c"

        def _store_blobs(conn):
            conn.execute("UPDATE sessions SET title = ? WHERE id = 's'", (undecodable,))
            conn.execute("UPDATE messages SET reasoning = ? WHERE session_id = 's'", (undecodable,))

        db._execute_write(_store_blobs)
        session = db.get_session("s")
        messages = db.get_messages("s")
        assert session["title"] == "why so serious \ufffd"
        assert messages[0]["reasoning"] == "why so serious \ufffd"
        assert json.dumps(session) and json.dumps(messages)
        # The listing surface shares the same normalization boundary.
        assert all(json.dumps(s) for s in db.list_recent_sessions_bounded())
    finally:
        db.close()


def test_take_unseen_reactions_normalizes_blob_stored_role(tmp_path):
    """A BLOB-stored role bypasses text_factory (sqlite3 contract) and used to escape the
    take_unseen_reactions payload as bytes, so json.dumps(result) raised TypeError. It must
    degrade to str exactly like every other public read surface (#109465 review, gap 3)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "react to me")
        row_id = db.latest_message_row_id("s", role="user")
        db.set_message_reaction("s", row_id, "\U0001f44d", author="user")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET role = ? WHERE id = ?", (b"user \xe2\x9c", row_id)))
        assert db._conn.execute(
            "SELECT typeof(role) FROM messages WHERE id = ?", (row_id,)).fetchone()[0] == "blob"

        result = db.take_unseen_reactions("s", author="user")
        assert result == [{
            "row_id": row_id, "role": "user \ufffd", "emoji": "\U0001f44d", "text": "react to me"}]
        assert json.dumps(result)
    finally:
        db.close()


def test_partial_projection_apis_normalize_blob_stored_cells(tmp_path):
    """list_never_active_keyed_sessions/list_skill_scaffolded_sessions used to hand out
    dict(row) directly, letting a corrupt BLOB-stored cell escape the public projection as
    bytes. They share the _session_row_dict normalization boundary now, so both payloads
    stay JSON-serializable (#109465 review, gap 3)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db._conn.execute(
            "INSERT INTO sessions (id, source, session_key, chat_id, chat_type, user_id, "
            "started_at, message_count, tool_call_count, api_call_count, input_tokens, "
            "output_tokens, archived, pinned) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0)",
            ("junk", "telegram", b"agent:main:telegram:dm:junk\xe2\x9c", "chat-1", "dm",
             "user-1", time.time() - 45 * 86400.0))
        db._conn.commit()
        keyed = db.list_never_active_keyed_sessions(older_than_days=30)
        assert [r["id"] for r in keyed] == ["junk"]
        assert keyed[0]["session_key"] == "agent:main:telegram:dm:junk\ufffd"
        assert json.dumps(keyed)

        # Content stays TEXT: the selector's LIKE only matches TEXT storage. The BLOB damage
        # lands on title, which the selector only checks for IS NOT NULL.
        db.create_session("titled", "cli")
        db.append_message("titled", "user", '[IMPORTANT: The user has invoked the "work" skill')
        db.set_session_title("titled", "placeholder title")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET title = ? WHERE id = 'titled'", (b"Setup \xe2\x9c",)))
        assert db._conn.execute(
            "SELECT typeof(title) FROM sessions WHERE id = 'titled'").fetchone()[0] == "blob"
        scaffolded = db.list_skill_scaffolded_sessions()
        assert [r["id"] for r in scaffolded] == ["titled"]
        assert scaffolded[0]["title"] == "Setup \ufffd"
        assert json.dumps(scaffolded)
    finally:
        db.close()
