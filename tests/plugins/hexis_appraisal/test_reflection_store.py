"""Schema v3 store contract tests: turn excerpts, meta watermark, lazy
decay, reflection delta keys, lock degradation.

Every test uses tmp_path with explicit db_path= — the real $HERMES_HOME is
never touched. Direct sqlite3 use is test instrumentation only (timestamp
rewinds, lock holders, column introspection); plugin code goes through
store.py exclusively.

Lock-semantics facts (03-01, verified): under WAL a held BEGIN EXCLUSIVE
never blocks READERS, so reader-degradation rows flip the tmp DB to
rollback journal mode where EXCLUSIVE genuinely blocks readers; write paths
degrade via the _DEFAULT_BUSY_TIMEOUT_MS monkeypatch.
"""

import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

from hexis_appraisal import store


def _quarantine_files(tmp_path):
    return [
        p
        for p in tmp_path.glob("state.db.quarantined-*")
        if not (p.name.endswith("-wal") or p.name.endswith("-shm"))
    ]


def _create_v2_db(db):
    """Structurally-valid v2 DB: all seven tables, turn_log WITHOUT
    assistant_excerpt, schema_version '2'."""
    conn = sqlite3.connect(str(db))
    try:
        with conn:
            conn.execute(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE affect_summary (id INTEGER PRIMARY KEY CHECK (id=1),"
                " summary TEXT, valence REAL, arousal REAL, intensity REAL,"
                " updated_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE concerns (id INTEGER PRIMARY KEY, text TEXT,"
                " weight REAL, status TEXT, created_at TEXT, updated_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE contradictions (id INTEGER PRIMARY KEY, kind TEXT,"
                " description TEXT, evidence TEXT, resolved INTEGER,"
                " created_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE trust_scores (key TEXT PRIMARY KEY, value REAL,"
                " updated_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE turn_log (id INTEGER PRIMARY KEY, session_id TEXT,"
                " turn_id TEXT, user_excerpt TEXT, appraisal_json TEXT,"
                " created_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE telemetry (id INTEGER PRIMARY KEY, ts TEXT,"
                " session_id TEXT, wall_ms INTEGER, model TEXT,"
                " tokens_in INTEGER, tokens_out INTEGER, outcome TEXT,"
                " error TEXT)"
            )
            conn.execute("INSERT INTO meta VALUES ('schema_version', '2')")
    finally:
        conn.close()


def _rewind_concern(db, text, days_old):
    """Push a concern's timestamps into the past (test instrumentation —
    apply_deltas always stamps now)."""
    past = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    conn = sqlite3.connect(str(db))
    try:
        with conn:
            conn.execute(
                "UPDATE concerns SET updated_at=?, created_at=? WHERE text=?",
                (past, past, text),
            )
    finally:
        conn.close()


def _turn_log_columns(db):
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        return [r[1] for r in conn.execute("PRAGMA table_info(turn_log)")]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema v3 quarantine
# ---------------------------------------------------------------------------


def test_v2_db_quarantined_and_recreated_at_v3(tmp_path):
    db = tmp_path / "state.db"
    _create_v2_db(db)
    assert "assistant_excerpt" not in _turn_log_columns(db)  # genuine v2

    assert store.ensure_db(db) is True
    assert len(_quarantine_files(tmp_path)) == 1
    assert store.SCHEMA_VERSION == 3
    assert "assistant_excerpt" in _turn_log_columns(db)
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("3",)


# ---------------------------------------------------------------------------
# meta_set + get_meta
# ---------------------------------------------------------------------------


def test_meta_set_get_meta_round_trip_and_overwrite(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.get_meta("last_reflected_turn_log_id", db_path=db) is None

    assert store.apply_deltas(
        {"meta_set": {"last_reflected_turn_log_id": 5}}, db
    ) is True
    assert store.get_meta("last_reflected_turn_log_id", db_path=db) == "5"

    # Overwrite works (INSERT-or-REPLACE semantics).
    assert store.apply_deltas(
        {"meta_set": {"last_reflected_turn_log_id": 12,
                      "last_seen_session_id": "sess-a"}}, db
    ) is True
    assert store.get_meta("last_reflected_turn_log_id", db_path=db) == "12"
    assert store.get_meta("last_seen_session_id", db_path=db) == "sess-a"
    # schema_version untouched by unrelated meta writes.
    assert store.get_meta("schema_version", db_path=db) == str(
        store.SCHEMA_VERSION
    )


# ---------------------------------------------------------------------------
# turn_log assistant excerpts + read_turns_since
# ---------------------------------------------------------------------------


def test_turn_log_assistant_excerpt_and_read_turns_since(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.apply_deltas(
        {
            "turn_log_add": [
                {"session_id": "s1", "turn_id": "t%d" % i,
                 "user_excerpt": "u%d" % i, "assistant_excerpt": "a%d" % i}
                for i in range(1, 4)
            ]
        },
        db,
    ) is True

    rows = store.read_turns_since(0, db_path=db)
    assert [r["turn_id"] for r in rows] == ["t1", "t2", "t3"]  # ordered
    assert [r["user_excerpt"] for r in rows] == ["u1", "u2", "u3"]
    assert [r["assistant_excerpt"] for r in rows] == ["a1", "a2", "a3"]

    # Only rows after the watermark.
    after_first = store.read_turns_since(rows[0]["id"], db_path=db)
    assert [r["turn_id"] for r in after_first] == ["t2", "t3"]

    # Limit honored.
    assert len(store.read_turns_since(0, db_path=db, limit=1)) == 1

    # Absent DB degrades to [].
    assert store.read_turns_since(0, db_path=tmp_path / "nope" / "x.db") == []


# ---------------------------------------------------------------------------
# Lazy decay at snapshot read (reads never write)
# ---------------------------------------------------------------------------


def test_decay_seven_day_half_life(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.apply_deltas(
        {"concerns_add": [{"text": "week-old", "weight": 1.0}]}, db
    ) is True
    _rewind_concern(db, "week-old", 7)

    snap = store.read_snapshot(db)
    (row,) = snap["concerns"]
    assert row["weight"] == 1.0  # raw weight untouched
    assert row["effective_weight"] == pytest.approx(0.5, abs=0.01)


def test_decay_excludes_below_threshold_without_writing(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.apply_deltas(
        {"concerns_add": [{"text": "stale", "weight": 0.5}]}, db
    ) is True
    _rewind_concern(db, "stale", 35)  # 0.5 * 0.5**5 = 0.015625 < 0.1

    snap = store.read_snapshot(db)
    assert snap["concerns"] == []  # excluded from the hot-path snapshot

    # READS NEVER WRITE: the row is still in the table…
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        count = conn.execute("SELECT COUNT(*) FROM concerns").fetchone()[0]
    finally:
        conn.close()
    assert count == 1

    # …and the reflection raw view sees it (prune candidate discovery).
    raw = store.read_snapshot(db, include_decayed=True)
    (row,) = raw["concerns"]
    assert row["effective_weight"] < store.DECAY_PRUNE_THRESHOLD


def test_fresh_concern_effective_weight_equals_weight(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.apply_deltas(
        {"concerns_add": [{"text": "fresh", "weight": 0.8}]}, db
    ) is True
    snap = store.read_snapshot(db)
    (row,) = snap["concerns"]
    assert row["effective_weight"] == pytest.approx(0.8, abs=0.001)


def test_malformed_timestamp_means_no_decay(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.apply_deltas(
        {"concerns_add": [{"text": "odd-ts", "weight": 0.7}]}, db
    ) is True
    conn = sqlite3.connect(str(db))
    try:
        with conn:
            conn.execute(
                "UPDATE concerns SET updated_at='not-a-date',"
                " created_at='also-bad' WHERE text='odd-ts'"
            )
    finally:
        conn.close()
    snap = store.read_snapshot(db)
    (row,) = snap["concerns"]
    assert row["effective_weight"] == 0.7  # undecayed, included


# ---------------------------------------------------------------------------
# Reflection delta keys
# ---------------------------------------------------------------------------


def test_concerns_update_prune_and_contradictions_resolve(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.apply_deltas(
        {
            "concerns_add": [
                {"text": "keep-me", "weight": 0.9},
                {"text": "prune-me", "weight": 0.9},
            ],
            "contradictions_add": [
                {"kind": "narrative", "description": "reversal",
                 "evidence": "ev"},
            ],
        },
        db,
    ) is True
    snap = store.read_snapshot(db)
    ids = {c["text"]: c["id"] for c in snap["concerns"]}
    contra_id = snap["contradictions"][0]["id"]

    assert store.apply_deltas(
        {
            "concerns_update": [{"id": ids["keep-me"], "weight": 0.3}],
            "concerns_prune": [ids["prune-me"]],
            "contradictions_resolve": [contra_id],
        },
        db,
    ) is True

    snap = store.read_snapshot(db)
    assert [c["text"] for c in snap["concerns"]] == ["keep-me"]
    assert snap["concerns"][0]["weight"] == 0.3
    assert snap["contradictions"][0]["resolved"] == 1

    # The prune was a real DELETE, not a status flip.
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM concerns WHERE text='prune-me'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_caps_still_enforced_alongside_new_keys(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.apply_deltas(
        {
            "concerns_add": [
                {"text": "concern-%02d" % i, "weight": 1.0} for i in range(30)
            ],
            "meta_set": {"last_reflected_turn_log_id": 1},
        },
        db,
    ) is True
    snap = store.read_snapshot(db)
    assert len(snap["concerns"]) <= store.CAPS["concerns"]
    assert store.get_meta("last_reflected_turn_log_id", db_path=db) == "1"


# ---------------------------------------------------------------------------
# Lock degradation — none of the new surfaces may raise
# ---------------------------------------------------------------------------


def test_locked_db_reflection_surfaces_degrade(tmp_path, monkeypatch):
    """get_meta -> None, read_turns_since -> [], apply_deltas meta_set ->
    False under a genuinely reader-blocking lock; no exception in any case.

    WAL EXCLUSIVE never blocks readers (03-01 verified), so the tmp DB is
    flipped to rollback journal mode; sqlite3.connect is wrapped with
    timeout=0.1 and _DEFAULT_BUSY_TIMEOUT_MS shrunk so the test stays fast.
    """
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
    finally:
        conn.close()

    monkeypatch.setattr(store, "_DEFAULT_BUSY_TIMEOUT_MS", 100)
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        store.sqlite3, "connect",
        lambda *args, **kwargs: real_connect(*args, timeout=0.1, **kwargs),
    )

    holder = real_connect(str(db))
    try:
        holder.execute("BEGIN EXCLUSIVE")
        start = time.monotonic()
        assert store.get_meta("schema_version", db_path=db) is None
        assert store.read_turns_since(0, db_path=db) == []
        assert store.apply_deltas(
            {"meta_set": {"last_seen_session_id": "s1"}}, db
        ) is False
        assert time.monotonic() - start < 3.0  # busy timeouts honored
    finally:
        holder.rollback()
        holder.close()
