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
import contextlib
import json
import sqlite3
import subprocess
import sys
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


def test_generic_malformed_open_does_not_attempt_schema_surgery(
    tmp_path, monkeypatch
):
    """A generic SQLITE_CORRUPT error has no schema/FTS provenance."""
    db_path = tmp_path / "state.db"
    repair_calls = []

    def _generic_corruption(*_args, **_kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(hermes_state, "apply_wal_with_fallback", _generic_corruption)
    monkeypatch.setattr(
        hermes_state,
        "repair_state_db_schema",
        lambda *args, **kwargs: repair_calls.append((args, kwargs)),
    )

    with pytest.raises(sqlite3.DatabaseError, match="disk image is malformed"):
        SessionDB(db_path=db_path)

    assert repair_calls == []


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






# ── FTS read-corruption class (#66724) ───────────────────────────────────
# Even when writes succeed, partial FTS5 shadow-table damage makes MATCH /
# snippet / rank queries fail with DatabaseError("database disk image is
# malformed") while plain reads of the FTS5 table still parse. The read
# probe in _db_opens_cleanly must surface this corruption class as a reason
# so the repair path triggers, but it must NOT misclassify the supported
# degraded-runtime path (no fts5 module / no trigram tokenizer) as
# corruption — doing so would route a healthy degraded DB through the
# repair fallback that deletes the messages_fts% schema.


def _corrupt_fts_shadow_segments(db_path: Path) -> None:
    """Overwrite the FTS5 shadow b-tree blocks for ``messages_fts`` only.

    Distinct from ``_corrupt_fts_index_data`` which targets the writes-side
    trigger path; this targets the MATCH query path so the read probe is
    what fires.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("UPDATE messages_fts_data SET block = X'BADC0FFEE0DDF00D'")
    conn.close()




def test_fts_read_corruption_repaired_in_place(tmp_path):
    """``repair_state_db_schema`` rebuilds the FTS index so reads resume."""
    from hermes_state import _db_opens_cleanly

    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_fts_shadow_segments(db_path)

    assert _db_opens_cleanly(db_path) is not None  # unhealthy before

    report = repair_state_db_schema(db_path)
    assert report["repaired"] is True
    assert _db_opens_cleanly(db_path) is None  # healthy after rebuild

    # Search back online.
    db = SessionDB(db_path=db_path)
    try:
        hits = db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pizza'"
        ).fetchone()[0]
        assert hits >= 5
    finally:
        db.close()


# ── Degraded-runtime compatibility (regression for #66906 review) ────────
# The read probe must NOT misclassify a supported degraded runtime (no
# fts5 module / no trigram tokenizer) as corruption. If it did, a healthy
# degraded DB would be sent into the repair path, whose final fallback
# deletes the messages_fts% schema — breaking the very FTS tables that
# may have been inherited from a prior build that did have FTS5.


class _NoFts5RuntimeCursor(sqlite3.Cursor):
    """Simulate a runtime without the fts5 module: fts5 table exists but
    MATCH queries raise the canonical capability error."""

    def execute(self, sql, parameters=()):
        probe = sql.strip()
        if "MATCH" in probe and '""' in probe and "messages_fts " in probe:
            raise sqlite3.OperationalError("no such module: fts5")
        return super().execute(sql, parameters)


class _NoFts5RuntimeConnection(sqlite3.Connection):
    def cursor(self, factory=None):
        return super().cursor(factory or _NoFts5RuntimeCursor)


class _NoTrigramRuntimeCursor(sqlite3.Cursor):
    """Simulate a runtime with FTS5 but without the trigram tokenizer."""

    def execute(self, sql, parameters=()):
        probe = sql.strip()
        if "MATCH" in probe and '""' in probe and "messages_fts_trigram" in probe:
            raise sqlite3.OperationalError("no such tokenizer: trigram")
        return super().execute(sql, parameters)


class _NoTrigramRuntimeConnection(sqlite3.Connection):
    def cursor(self, factory=None):
        return super().cursor(factory or _NoTrigramRuntimeCursor)






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




def _corrupt_btree_index(db_path: Path, index_name: str) -> None:
    """Make a real B-tree index stale so integrity_check reports
    'wrong # of entries in index <name>'.

    writable_schema hack: temporarily rewrite the index definition in
    sqlite_master to a partial index (``WHERE 0``), REINDEX so its b-tree is
    rebuilt EMPTY, then restore the original full definition. The stored
    b-tree now has zero entries while the schema says it must cover every
    row — exactly the on-disk state issue #63386 reported for
    idx_sessions_handoff_state, produced without any mocking.
    """
    raw = sqlite3.connect(str(db_path))
    orig_sql = raw.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    ).fetchone()[0]

    def _set_index_sql(conn, sql):
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_master SET sql=? WHERE type='index' AND name=?",
            (sql, index_name),
        )
        ver = conn.execute("PRAGMA schema_version").fetchone()[0]
        conn.execute(f"PRAGMA schema_version={ver + 1}")
        conn.execute("PRAGMA writable_schema=OFF")
        conn.commit()

    _set_index_sql(raw, orig_sql + " WHERE 0")
    raw.close()

    # Fresh connection so the doctored schema is re-parsed, then rebuild the
    # index under the WHERE 0 definition — empty b-tree on disk.
    raw = sqlite3.connect(str(db_path))
    raw.execute(f"REINDEX {index_name}")
    raw.commit()
    # Restore the original (full) definition: schema and b-tree now disagree.
    _set_index_sql(raw, orig_sql)
    raw.close()


def test_repair_rebuilds_stale_btree_indexes(tmp_path):
    """repair_state_db_schema repairs a REAL stale B-tree index via REINDEX.

    End-to-end, no mocks: a genuinely stale index (empty b-tree under a full
    index definition — the #63386 'wrong # of entries in index' class) is
    detected by the real _db_opens_cleanly, repaired by Strategy 0.5
    (REINDEX), and the DB verifies clean afterwards with real integrity
    checks.
    """
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)

    _corrupt_btree_index(db_path, "idx_messages_session")

    # The real detector must see the real corruption...
    reason = hermes_state._db_opens_cleanly(db_path)
    assert reason is not None
    assert "wrong # of entries in index idx_messages_session" in reason

    # ...and the real repair ladder must fix it via REINDEX.
    report = repair_state_db_schema(db_path)
    assert report["repaired"] is True
    assert report["strategy"] == "reindex_btree"

    # Post-repair the DB is genuinely healthy: detector and raw
    # integrity_check both agree, and the repaired index answers queries.
    assert hermes_state._db_opens_cleanly(db_path) is None
    raw = sqlite3.connect(str(db_path))
    assert raw.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    n = raw.execute(
        "SELECT count(*) FROM messages INDEXED BY idx_messages_session "
        "WHERE session_id IS NOT NULL"
    ).fetchone()[0]
    raw.close()
    assert n == 10  # every row visible through the rebuilt index


def test_repair_stale_btree_index_preserves_rows(tmp_path):
    """The REINDEX strategy is non-destructive: sessions/messages survive."""
    db_path = tmp_path / "state.db"
    sid = _build_healthy_db(db_path)
    _corrupt_btree_index(db_path, "idx_messages_session")

    report = repair_state_db_schema(db_path, backup=False)
    assert report["strategy"] == "reindex_btree"

    db = SessionDB(db_path=db_path)
    try:
        msgs = db.get_messages(sid)
        assert len(msgs) == 10
        assert msgs[0]["content"] == "hello world 0"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Cross-process serialisation of the schema surgery
# ---------------------------------------------------------------------------
# A normal host runs several independent processes against one state.db: the
# gateway service, the Desktop app's own `hermes serve` backend, interactive
# CLI sessions and the TUI slash worker. `_repair_attempt_lock` is a
# threading.Lock and covers none of that, so two of them hitting a malformed
# DB at once each ran the full writable_schema surgery + VACUUM on a private
# connection — one repairing while the other was mid-surgery.


_HOLD_LOCK_SCRIPT = """
import sys, time, fcntl, pathlib
sys.path.insert(0, {root!r})
lock_path = pathlib.Path({lock!r})
handle = lock_path.open("a+b")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep({hold})
"""


@contextlib.contextmanager
def _lock_held_by_other_process(db_path: Path, hold_seconds: float = 30.0):
    """Hold the repair flock for *db_path* in a real child process."""
    script = _HOLD_LOCK_SCRIPT.format(
        root=str(Path(hermes_state.__file__).parent),
        lock=str(db_path.with_name(db_path.name + ".repair.lock")),
        hold=hold_seconds,
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        # Wait for the child to actually own the lock before yielding.
        assert proc.stdout.readline().strip() == "locked"
        yield
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock test")
def test_repair_skips_surgery_while_another_process_holds_the_lock(
    tmp_path, monkeypatch
):
    """The losing process must NOT run writable_schema surgery in parallel."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)
    monkeypatch.setattr(hermes_state, "_REPAIR_LOCK_TIMEOUT_SECONDS", 0.5)

    with _lock_held_by_other_process(db_path):
        report = repair_state_db_schema(db_path)

    assert report["repaired"] is False
    assert "repair lock" in (report["error"] or "")
    # No surgery ran: no backup was taken and the DB is still malformed.
    assert report["backup_path"] is None
    assert not list(tmp_path.glob("state.db.malformed-backup-*"))
    assert hermes_state._db_opens_cleanly(db_path) is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock test")
def test_repair_reports_success_when_the_holder_already_healed_the_db(
    tmp_path, monkeypatch
):
    """Timing out against a healthy DB is a success, not an error."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    monkeypatch.setattr(hermes_state, "_REPAIR_LOCK_TIMEOUT_SECONDS", 0.5)

    with _lock_held_by_other_process(db_path):
        report = repair_state_db_schema(db_path)

    assert report["repaired"] is True
    assert report["strategy"] == "repaired_by_other_process"


_REPAIR_SCRIPT = """
import sys, json
sys.path.insert(0, {root!r})
from hermes_state import repair_state_db_schema
print(json.dumps(repair_state_db_schema({db!r})), flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock test")
def test_two_processes_repairing_at_once_perform_surgery_once(tmp_path):
    """Concurrent repairers serialise; the loser sees a healed DB and stops.

    Without the cross-process lock both processes back up and operate on
    sqlite_master, i.e. one runs surgery on a database the other is
    simultaneously rewriting. The backup count is the observable proxy for
    "how many processes entered the critical section".
    """
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    script = _REPAIR_SCRIPT.format(
        root=str(Path(hermes_state.__file__).parent), db=str(db_path)
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(2)
    ]
    reports = []
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
        reports.append(json.loads(out.strip().splitlines()[-1]))

    assert all(r["repaired"] for r in reports), reports
    # Exactly one process did the work; the other found the DB already healthy.
    strategies = sorted(r["strategy"] for r in reports)
    assert "already_healthy" in strategies or "repaired_by_other_process" in strategies
    assert len(list(tmp_path.glob("state.db.malformed-backup-*"))) == 1

    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
    finally:
        conn.close()


def test_schema_surgery_bumps_the_schema_cookie(tmp_path):
    """Live connections in other processes must be told to reload the schema.

    Editing sqlite_master under writable_schema=ON does not bump the cookie
    that every other connection checks before running a prepared statement,
    so they keep compiling against objects the surgery just deleted.
    """
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    probe = sqlite3.connect(str(db_path))
    try:
        probe.execute("PRAGMA writable_schema=ON")
        before = probe.execute("PRAGMA schema_version").fetchone()[0]
    finally:
        probe.close()

    report = repair_state_db_schema(db_path)
    assert report["repaired"] is True

    probe = sqlite3.connect(str(db_path))
    try:
        after = probe.execute("PRAGMA schema_version").fetchone()[0]
    finally:
        probe.close()
    assert after != before


# ---------------------------------------------------------------------------
# Backup refusal is a hard stop (#69603)
# ---------------------------------------------------------------------------
# The Aug 2026 incident on #69603: the pre-repair backup was refused because
# another same-process handle was open, and the repair proceeded anyway —
# every later strategy (writable_schema surgery, FTS deletion, VACUUM) was
# then reachable against the only remaining copy of the damaged DB.


def test_backup_refusal_hard_stops_the_repair(tmp_path, monkeypatch):
    """A refused pre-repair backup must abort the repair, not fail open."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)
    original_bytes = db_path.read_bytes()

    monkeypatch.setattr(
        hermes_state,
        "_backup_db_file",
        lambda p: (None, "a connection to it is still open in this process"),
    )

    report = repair_state_db_schema(db_path)

    assert report["repaired"] is False
    assert report["backup_path"] is None
    assert "backup refused" in (report["error"] or "")
    assert "still open" in report["error"]
    # No mutating strategy ran: the damaged source bytes are untouched.
    assert db_path.read_bytes() == original_bytes
    assert hermes_state._db_opens_cleanly(db_path) is not None


def test_backup_copy_failure_hard_stops_the_repair(tmp_path, monkeypatch):
    """An OS-level backup copy failure aborts the repair with the reason."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    monkeypatch.setattr(
        hermes_state,
        "_backup_db_file",
        lambda p: (None, "backup copy failed: [Errno 28] No space left on device"),
    )

    report = repair_state_db_schema(db_path)

    assert report["repaired"] is False
    assert "No space left on device" in (report["error"] or "")
    assert not list(tmp_path.glob("state.db.malformed-backup-*"))


def test_backup_false_still_skips_backup_and_repairs(tmp_path):
    """Explicit backup=False (CLI --no-backup) keeps working."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    report = repair_state_db_schema(db_path, backup=False)

    assert report["repaired"] is True
    assert report["backup_path"] is None
    assert not list(tmp_path.glob("state.db.malformed-backup-*"))


# ---------------------------------------------------------------------------
# Journal-mode restore after surgery (#89674): corruption drops the WAL bit
# from the header, the repair strategies rebuild the file in the default
# (delete) mode, and nothing used to record the flip. The configured
# database.journal_mode must be re-applied, with a WARNING naming it.
# ---------------------------------------------------------------------------


def _mode_of(db_path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def _configure_journal_mode(monkeypatch, tmp_path, mode) -> None:
    import yaml

    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        yaml.safe_dump({"database": {"journal_mode": mode}}), encoding="utf-8",
    )
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable", lambda **kwargs: False,
    )


def test_repair_restores_configured_wal_after_surgery(
    tmp_path, caplog, monkeypatch
):
    """A repaired file left in delete mode must return to the configured WAL.

    Simulates the reported sequence: corruption drops the WAL bit (the file
    reads back as delete), surgery heals the schema, and the restore — now
    resolved through the canonical apply_wal_with_fallback — brings the
    store back to database.journal_mode. The duplicate-FTS damage makes the
    pre-surgery probe fail, so no before/after comparison WARNING fires;
    the canonical path's own logging covers the restore."""
    import logging

    db_path = tmp_path / "state.db"
    _configure_journal_mode(monkeypatch, tmp_path, "wal")
    _build_healthy_db(db_path)
    assert _mode_of(db_path) == "wal"
    # Downgrade while the file is still healthy (the damaged file rejects
    # every statement — see test_duplicate_fts_makes_every_statement_fail),
    # simulating the header state the reporter's corruption storm left: the
    # WAL bit is gone and the file reads back as delete.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    assert _mode_of(db_path) == "delete"
    _corrupt_duplicate_fts(db_path)

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        report = repair_state_db_schema(db_path)

    assert report["repaired"] is True
    assert _mode_of(db_path) == "wal"


def test_repair_restore_matches_canonical_on_vulnerable_sqlite(
    tmp_path, monkeypatch
):
    """The restore must go through the WAL-reset gate, not around it.

    On a WAL-reset-vulnerable SQLite (the reporter ran 3.50.4), the
    canonical open path keeps a rebuilt non-WAL file in DELETE — a freshly
    repaired file IS a new database. The restore used to switch WAL
    directly and diverged from the front door; going through
    apply_wal_with_fallback must converge with the canonical behaviour."""
    db_path = tmp_path / "state.db"
    _configure_journal_mode(monkeypatch, tmp_path, "wal")
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable", lambda **kwargs: True
    )
    _build_healthy_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    _corrupt_duplicate_fts(db_path)

    report = repair_state_db_schema(db_path)

    assert report["repaired"] is True
    # Same outcome as the canonical open path on this runtime: DELETE.
    assert _mode_of(db_path) == "delete"


def test_repair_logs_mode_change_when_probe_succeeded(
    tmp_path, caplog, monkeypatch
):
    """When the pre-surgery probe read the file (header intact), the
    post-restore WARNING names the before/after modes — the #89393 signal,
    arriving through the repair door instead of the open door."""
    import logging
    from unittest.mock import patch

    db_path = tmp_path / "state.db"
    _configure_journal_mode(monkeypatch, tmp_path, "wal")
    _build_healthy_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    _corrupt_duplicate_fts(db_path)

    with (
        patch.object(
            hermes_state, "_probe_journal_mode_for_repair", return_value="delete"
        ),
        caplog.at_level(logging.WARNING, logger="hermes_state"),
    ):
        report = repair_state_db_schema(db_path)

    assert report["repaired"] is True
    assert report["journal_mode_before"] == "delete"
    assert _mode_of(db_path) == "wal"
    assert any(
        "changed journal_mode" in r.getMessage() for r in caplog.records
    ), f"expected the mode flip to be logged; got: {[r.getMessage() for r in caplog.records]}"


def test_repair_logs_nothing_when_mode_already_matches(
    tmp_path, caplog, monkeypatch
):
    """FTS-only corruption keeps the WAL bit; the restore is then a no-op and
    must not emit a mode-flip WARNING."""
    import logging

    db_path = tmp_path / "state.db"
    _configure_journal_mode(monkeypatch, tmp_path, "wal")
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        report = repair_state_db_schema(db_path)

    assert report["repaired"] is True
    assert _mode_of(db_path) == "wal"
    assert not any(
        "changed journal_mode" in r.getMessage() for r in caplog.records
    )


def test_repair_restore_failure_is_nonfatal_and_logged(
    tmp_path, caplog, monkeypatch
):
    """When the canonical restore path raises (locked/unsupported fs), the
    repair result stands and a WARNING is logged — never an exception out
    of the repair path."""
    import logging
    from unittest.mock import patch

    db_path = tmp_path / "state.db"
    _configure_journal_mode(monkeypatch, tmp_path, "wal")
    _build_healthy_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    _corrupt_duplicate_fts(db_path)

    def _refused(conn, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    with (
        patch.object(hermes_state, "apply_wal_with_fallback", _refused),
        caplog.at_level(logging.WARNING, logger="hermes_state"),
    ):
        report = repair_state_db_schema(db_path)

    assert report["repaired"] is True
    assert _mode_of(db_path) == "delete"
    assert any(
        "journal-mode restore failed" in r.getMessage()
        for r in caplog.records
    )


def test_repair_honors_configured_delete_mode(tmp_path, monkeypatch):
    """An explicit database.journal_mode=delete store stays delete after
    repair — the restore applies the operator's setting, not a hard-coded
    WAL."""
    db_path = tmp_path / "state.db"
    _configure_journal_mode(monkeypatch, tmp_path, "delete")
    _build_healthy_db(db_path)
    assert _mode_of(db_path) == "delete"
    _corrupt_duplicate_fts(db_path)

    report = repair_state_db_schema(db_path)

    assert report["repaired"] is True
    assert _mode_of(db_path) == "delete"
