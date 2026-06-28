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

    # Make every health verification report "still broken" until the drop-FTS
    # pass has actually removed the messages_fts schema, so the routine
    # escalates past the in-place-rebuild and dedup passes to strat 2 (drop FTS
    # + VACUUM) and runs its real SQL against the file. Keyed on whether the FTS
    # schema is still present rather than a call counter, so it stays correct as
    # earlier verification call sites are added/removed.
    real_check = hermes_state._db_opens_cleanly
    calls = {"n": 0}

    def flaky_check(path):
        calls["n"] += 1
        try:
            probe = sqlite3.connect(str(path))
            still_has_fts = probe.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE name LIKE 'messages_fts%'"
            ).fetchone()[0]
            probe.close()
        except sqlite3.DatabaseError:
            # sqlite_master still malformed (pre-dedup) — treat as broken.
            return "pretend still broken (schema unreadable)"
        if still_has_fts:
            return "pretend in-place/dedup passes were insufficient"
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


# ── FTS write-corruption class (#50502) ──────────────────────────────────
# A readable state.db can still reject every message write through the
# messages_fts* triggers when the FTS index is corrupt. Plain
# `SELECT COUNT(*)` reads succeed, so the old read-only health probe reported
# it healthy and the gateway silently dropped conversation history.


def _corrupt_fts_index_data(db_path: Path) -> None:
    """Overwrite the FTS5 shadow b-tree blocks with garbage bytes.

    Reproduces the runtime "database disk image is malformed" / "malformed
    inverted index for FTS5 table" failure that fires on writes through the
    triggers while base-table reads still return rows.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEF'")
    conn.close()


def test_fts_write_corruption_detected_by_write_probe(tmp_path):
    """_db_opens_cleanly's rolled-back write probe flags FTS write corruption."""
    from hermes_state import _db_opens_cleanly

    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    assert _db_opens_cleanly(db_path) is None  # healthy before

    _corrupt_fts_index_data(db_path)

    # Plain base-table reads still succeed — this is the silent class.
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] >= 1
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
    conn.close()

    # The write-aware probe reports the corruption (not a false "ok").
    reason = _db_opens_cleanly(db_path)
    assert reason is not None


def test_fts_write_corruption_repaired_in_place(tmp_path):
    """repair_state_db_schema rebuilds the FTS index; reads + writes resume."""
    from hermes_state import _db_opens_cleanly

    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_fts_index_data(db_path)

    report = repair_state_db_schema(db_path)
    assert report["repaired"] is True
    assert report["strategy"] in ("rebuild_fts", "dedup_schema", "drop_fts_rebuild")
    assert _db_opens_cleanly(db_path) is None

    # Canonical rows preserved AND new writes go through the triggers again.
    db = SessionDB(db_path=db_path)
    try:
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
        sid = db._conn.execute("SELECT id FROM sessions LIMIT 1").fetchone()[0]
        db.append_message(sid, role="user", content="post repair pizza message")
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 11
        hits = db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pizza'"
        ).fetchone()[0]
        assert hits >= 5
    finally:
        db.close()


def test_repair_noop_db_uses_already_healthy_shortcut(tmp_path):
    """A healthy DB returns the cheap already_healthy strategy, no surgery."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    report = repair_state_db_schema(db_path, backup=False)
    assert report["repaired"] is True
    assert report["strategy"] == "already_healthy"


def test_health_probe_detects_trigram_constructor_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    real_connect = hermes_state.sqlite3.connect

    class TrigramProbeFailureConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            if str(sql).startswith("SELECT * FROM messages_fts_trigram LIMIT 0"):
                raise sqlite3.DatabaseError(
                    "vtable constructor failed: messages_fts_trigram"
                )
            return self._conn.execute(sql, *args, **kwargs)

        def close(self):
            self._conn.close()

    def connect_with_broken_trigram(*args, **kwargs):
        return TrigramProbeFailureConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(hermes_state.sqlite3, "connect", connect_with_broken_trigram)

    reason = hermes_state._db_opens_cleanly(db_path)

    assert reason == "vtable constructor failed: messages_fts_trigram"


def test_repair_rebuilds_canonical_tables_when_trigram_constructor_fails(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    real_check = hermes_state._db_opens_cleanly
    checks = {"n": 0}

    def check_reports_constructor_failure_then_real(path):
        checks["n"] += 1
        if checks["n"] <= 3:
            return "vtable constructor failed: messages_fts_trigram"
        return real_check(path)

    monkeypatch.setattr(
        hermes_state,
        "_db_opens_cleanly",
        check_reports_constructor_failure_then_real,
    )

    report = repair_state_db_schema(db_path, backup=False)

    assert report["repaired"] is True
    assert report["strategy"] == "rebuild_canonical_tables"
    db = SessionDB(db_path=db_path)
    try:
        assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
        assert db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 10
        trigram_count = db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts_trigram"
        ).fetchone()[0]
        assert trigram_count == 10
    finally:
        db.close()


def test_select_cached_agent_history_prefers_longer_live_transcript():
    """Gateway guard keeps the live transcript when persisted history lags."""
    from gateway.run import _select_cached_agent_history

    persisted = [{"role": "user", "content": "only one"}]
    live = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    # Persisted lags (FTS write failed) → keep the longer live copy.
    out = _select_cached_agent_history(persisted, live)
    assert out == live
    assert out is not live  # returns a copy, not the live list

    # Persisted is current/longer → leave it untouched (identity preserved).
    longer_persisted = live + [{"role": "assistant", "content": "four"}]
    out2 = _select_cached_agent_history(longer_persisted, live)
    assert out2 is longer_persisted

    # No live transcript / not a list → no-op.
    assert _select_cached_agent_history(persisted, None) is persisted
    assert _select_cached_agent_history(persisted, "nope") is persisted
