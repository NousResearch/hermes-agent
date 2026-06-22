"""Recovery from a malformed state.db schema (duplicate sqlite_master rows).

This is the corruption class behind the user-reported symptom where Desktop /
Dashboard show "no sessions yet" while hundreds of session JSON files sit on
disk, and the backend logs:

    sqlite3.DatabaseError: malformed database schema (messages_fts) -
    table messages_fts already exists

The error fires on the *first* statement of any connection (PRAGMA
journal_mode in apply_wal_with_fallback), before _init_schema runs — so it
cannot be handled at the FTS-rebuild layer. These tests verify the
sqlite_master surgery path recovers the canonical data and self-heals on open.
"""
import sqlite3
import uuid
from pathlib import Path

import pytest

import hermes_state
from hermes_state import (
    SessionDB,
    _db_opens_cleanly,
    is_malformed_db_error,
    repair_state_db_schema,
)


def _build_healthy_db(db_path: Path) -> str:
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    for i in range(5):
        db.append_message(sid, role="user", content=f"hello world {i}")
        db.append_message(sid, role="assistant", content=f"reply about pizza {i}")
    db.close()
    return sid


def _corrupt_duplicate_fts(db_path: Path) -> None:
    """Inject a duplicate messages_fts row into sqlite_master.

    Reproduces 'malformed database schema (messages_fts) - table
    messages_fts already exists'.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
        "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
        "WHERE name='messages_fts'"
    )
    conn.commit()
    conn.close()


def test_duplicate_fts_makes_every_statement_fail(tmp_path):
    """Document the failure: not even PRAGMA journal_mode survives."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    conn = sqlite3.connect(str(db_path))
    with pytest.raises(sqlite3.DatabaseError) as exc_info:
        conn.execute("PRAGMA journal_mode").fetchone()
    conn.close()
    assert is_malformed_db_error(exc_info.value)


def test_repair_preserves_sessions_and_messages(tmp_path):
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    report = repair_state_db_schema(db_path)
    assert report["repaired"] is True
    assert report["strategy"] in {"dedup_schema", "drop_fts_rebuild"}
    # A backup of the malformed file is preserved.
    assert report["backup_path"] and Path(report["backup_path"]).exists()

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
    conn.close()


def test_repaired_db_search_works(tmp_path):
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)
    repair_state_db_schema(db_path)

    # Reopen and confirm the FTS index is usable (rebuilt or preserved).
    db = SessionDB(db_path=db_path)
    try:
        hits = db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pizza'"
        ).fetchone()[0]
        assert hits == 5
        msg_count = db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        assert msg_count == 10
    finally:
        db.close()


def test_sessiondb_auto_heals_on_open(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    sid = _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    # Fresh process-global guard so the attempt isn't pre-claimed.
    monkeypatch.setattr(hermes_state, "_repair_attempted_paths", set())

    db = SessionDB(db_path=db_path)
    try:
        assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert db._conn.execute(
            "SELECT id FROM sessions WHERE id=?", (sid,)
        ).fetchone() is not None
    finally:
        db.close()


def test_auto_heal_attempted_once_per_process(tmp_path, monkeypatch):
    """A still-broken DB must not loop: the second open just raises."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)
    monkeypatch.setattr(hermes_state, "_repair_attempted_paths", set())

    calls = {"n": 0}
    real_repair = hermes_state.repair_state_db_schema

    def fake_repair(path, **kw):
        calls["n"] += 1
        # Pretend repair failed so the guard's one-shot behavior is exercised.
        return {"repaired": False, "strategy": None, "backup_path": None, "error": "x"}

    monkeypatch.setattr(hermes_state, "repair_state_db_schema", fake_repair)

    with pytest.raises(sqlite3.DatabaseError):
        SessionDB(db_path=db_path)
    with pytest.raises(sqlite3.DatabaseError):
        SessionDB(db_path=db_path)
    assert calls["n"] == 1  # repair attempted only once across both opens

    monkeypatch.setattr(hermes_state, "repair_state_db_schema", real_repair)


def test_is_malformed_db_error_discriminates():
    assert is_malformed_db_error(
        sqlite3.DatabaseError("malformed database schema (messages_fts) - ...")
    )
    assert is_malformed_db_error(sqlite3.DatabaseError("database disk image is malformed"))
    assert not is_malformed_db_error(sqlite3.OperationalError("database is locked"))
    assert not is_malformed_db_error(ValueError("nope"))


def test_strategy_b_rebuild_when_dedup_insufficient(tmp_path, monkeypatch):
    """If the dedup pass can't fix it, the drop-FTS + rebuild pass must.

    Force strat 1 to be a no-op so the escalation path is exercised against a
    real malformed file. Data must still survive and search must work.
    """
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    # Make the post-strat-1 verification report "still broken" exactly once,
    # so the routine escalates to strat 2 (drop FTS + VACUUM) and runs its
    # real SQL against the file; the strat-2 verification then uses the real
    # check and passes.
    real_check = hermes_state._db_opens_cleanly
    calls = {"n": 0}

    def flaky_check(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return "pretend strat 1 was insufficient"
        return real_check(path)

    monkeypatch.setattr(hermes_state, "_db_opens_cleanly", flaky_check)
    report = repair_state_db_schema(db_path)
    monkeypatch.undo()

    assert report["repaired"] is True
    assert report["strategy"] == "drop_fts_rebuild"
    assert calls["n"] >= 2

    db = SessionDB(db_path=db_path)
    try:
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pizza'"
        ).fetchone()[0] == 5
    finally:
        db.close()


def test_unrepairable_file_fails_safely(tmp_path, monkeypatch):
    """A file too damaged to recover must report failure, keep a backup, and
    never raise from the repair routine itself."""
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"SQLite format 3\x00" + b"\x00\xde\xad\xbe\xef" * 200)

    report = repair_state_db_schema(db_path)
    assert report["repaired"] is False
    assert report["error"]
    # The (damaged) original bytes are preserved for manual restore.
    assert report["backup_path"] and Path(report["backup_path"]).exists()


def test_non_malformed_error_is_not_auto_repaired(tmp_path, monkeypatch):
    """Auto-heal must only trigger for the malformed-schema class, not for
    e.g. 'file is not a database' — those raise unchanged."""
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"this is definitely not a sqlite database")
    monkeypatch.setattr(hermes_state, "_repair_attempted_paths", set())

    called = {"n": 0}
    orig = hermes_state.repair_state_db_schema

    def spy(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(hermes_state, "repair_state_db_schema", spy)
    with pytest.raises(sqlite3.DatabaseError):
        SessionDB(db_path=db_path)
    assert called["n"] == 0  # never attempted repair for a non-malformed error


def test_repair_on_clean_db_is_noop(tmp_path):
    """Dedup-keyed repair must not damage a healthy DB if invoked."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)

    report = repair_state_db_schema(db_path, backup=False)
    assert report["repaired"] is True  # opens cleanly after a no-op dedup

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


# ---------------------------------------------------------------------------
# #50502 — FTS *write* corruption that reads alone report as healthy.
#
# A readable state.db can still be unusable for persistence: when the FTS5
# index is corrupt only on the write path, every read (integrity_check,
# COUNT(*) FROM sessions/messages) succeeds while every INSERT INTO messages
# fails through the messages_fts_insert trigger with
# "database disk image is malformed". The gateway swallows that write error
# and the next turn reloads an empty/stale transcript — immediate same-session
# amnesia. The health probe must therefore also exercise a (rolled-back)
# write, not just reads.
#
# These tests simulate the failure by making the messages_fts_insert trigger
# raise on insert. That faithfully reproduces "writes fail through the FTS
# triggers while reads pass" without sqlite_master surgery (which newer SQLite
# builds block), so the detection contract is exercised on every SQLite.
# ---------------------------------------------------------------------------
def _break_fts_insert_trigger(db_path: Path) -> None:
    """Make the messages_fts_insert trigger fail like a corrupt FTS index.

    Reads of sessions/messages and PRAGMA integrity_check stay healthy; only
    the trigger-driven write path raises 'database disk image is malformed'.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TRIGGER IF EXISTS messages_fts_insert")
        conn.execute(
            "CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN "
            "  SELECT RAISE(ABORT, 'database disk image is malformed'); "
            "END"
        )
        conn.commit()
    finally:
        conn.close()


def test_read_probes_miss_fts_write_corruption(tmp_path):
    """The old read-only checks all pass on a write-corrupt FTS index."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _break_fts_insert_trigger(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        # Every read-only signal the old probe relied on still reports healthy.
        assert conn.execute("PRAGMA journal_mode").fetchone() is not None
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
    finally:
        conn.close()


def test_db_opens_cleanly_detects_fts_write_corruption(tmp_path):
    """The write-aware probe flags the corruption the read checks miss."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)

    # Healthy DB: probe reports clean.
    assert _db_opens_cleanly(db_path) is None

    _break_fts_insert_trigger(db_path)

    # Corrupt write path: probe now returns the failure reason.
    reason = _db_opens_cleanly(db_path)
    assert reason is not None
    assert "malformed" in reason.lower()


def test_fts_write_probe_rolls_back_and_corruption_is_real(tmp_path):
    """The probe must persist nothing, and the simulated corruption must
    actually block real appends (not just be a synthetic flag)."""
    db_path = tmp_path / "state.db"
    sid = _build_healthy_db(db_path)
    _break_fts_insert_trigger(db_path)

    # The probe ran an INSERT internally; it must have rolled back — no probe
    # sentinel row leaks into messages.
    assert _db_opens_cleanly(db_path) is not None
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
        leaked = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE content LIKE "
            "'__hermes_fts_write_probe__%'"
        ).fetchone()[0]
        assert leaked == 0
    finally:
        conn.close()

    # A real append really does fail on this DB (the silent-drop class).
    db = SessionDB(db_path=db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            db.append_message(sid, role="user", content="next turn")
    finally:
        db.close()


def test_healthy_db_passes_write_probe(tmp_path):
    """A healthy DB stays healthy and the write probe leaves no residue."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)

    assert _db_opens_cleanly(db_path) is None
    # Probe is non-destructive: counts unchanged, no sentinel rows.
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE content LIKE "
            "'__hermes_fts_write_probe__%'"
        ).fetchone()[0] == 0
    finally:
        conn.close()
