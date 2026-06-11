"""Store contract tests: round-trip, degradation matrix, caps.

Every test uses tmp_path — the real $HERMES_HOME is never touched.
Direct sqlite3 use here is test instrumentation only; plugin code goes
through store.py exclusively.
"""

import sqlite3
import time

from hexis_appraisal import store

EXPECTED_TABLES = {
    "meta",
    "affect_summary",
    "concerns",
    "contradictions",
    "trust_scores",
    "turn_log",
}


def _quarantine_files(tmp_path):
    """Quarantined DB files (excluding WAL/SHM sidecars)."""
    return [
        p
        for p in tmp_path.glob("state.db.quarantined-*")
        if not (p.name.endswith("-wal") or p.name.endswith("-shm"))
    ]


def test_ensure_db_creates_schema(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert EXPECTED_TABLES <= tables
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert row == (str(store.SCHEMA_VERSION),)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_round_trip_identical_signals(tmp_path):
    """STATE-05: apply_deltas write -> read_snapshot reload -> identical signals."""
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True

    deltas = {
        "affect_summary": {
            "summary": "steady, curious",
            "valence": 0.25,
            "arousal": 0.5,
            "intensity": 0.75,
        },
        "concerns_add": [
            {"text": "deadline slipping", "weight": 0.5},
            {"text": "ambiguous requirement", "weight": 1.25},
        ],
        "contradictions_add": [
            {"kind": "semantic", "description": "sem-desc", "evidence": "sem-ev"},
            {"kind": "narrative", "description": "nar-desc", "evidence": "nar-ev"},
            {"kind": "relational", "description": "rel-desc", "evidence": "rel-ev"},
            {"kind": "emotional", "description": "emo-desc", "evidence": "emo-ev"},
        ],
        "trust_scores": {"user:drmani": 0.9, "source:webscrape": 0.1},
        "turn_log_add": [
            {
                "session_id": "s1",
                "turn_id": "t1",
                "user_excerpt": "hello",
                "appraisal_json": '{"a": 1}',
            },
            {"session_id": "s1", "turn_id": "t2", "user_excerpt": "again"},
            {"session_id": "s1", "turn_id": "t3", "user_excerpt": "bye"},
        ],
    }
    assert store.apply_deltas(deltas, db) is True

    def check(snap):
        assert snap is not None
        assert snap["schema_version"] == store.SCHEMA_VERSION
        affect = snap["affect_summary"]
        assert affect["summary"] == "steady, curious"
        assert affect["valence"] == 0.25
        assert affect["arousal"] == 0.5
        assert affect["intensity"] == 0.75
        concerns = {c["text"]: c for c in snap["concerns"]}
        assert set(concerns) == {"deadline slipping", "ambiguous requirement"}
        assert concerns["deadline slipping"]["weight"] == 0.5
        assert concerns["ambiguous requirement"]["weight"] == 1.25
        assert all(c["status"] == "open" for c in concerns.values())
        contras = {c["description"]: c for c in snap["contradictions"]}
        assert set(contras) == {"sem-desc", "nar-desc", "rel-desc", "emo-desc"}
        assert contras["sem-desc"]["kind"] == "semantic"
        assert contras["nar-desc"]["kind"] == "narrative"
        assert contras["rel-desc"]["kind"] == "relational"
        assert contras["emo-desc"]["kind"] == "emotional"
        assert contras["sem-desc"]["evidence"] == "sem-ev"
        assert snap["trust_scores"] == {"user:drmani": 0.9, "source:webscrape": 0.1}
        assert snap["turn_log_count"] == 3

    check(store.read_snapshot(db))
    # Second fresh read (new call, simulating reload): identical signals.
    check(store.read_snapshot(db))


def test_absent_db_read_returns_none(tmp_path):
    db = tmp_path / "absent" / "state.db"
    assert store.read_snapshot(db) is None
    assert not db.exists()


def test_corrupt_db_quarantined(tmp_path):
    """STATE-03: corrupt file -> quarantine sidecar + fresh usable DB."""
    db = tmp_path / "state.db"
    db.write_bytes(b"this is not a sqlite database")
    assert store.ensure_db(db) is True
    quarantined = _quarantine_files(tmp_path)
    assert len(quarantined) == 1
    snap = store.read_snapshot(db)
    assert snap is not None
    assert snap["schema_version"] == store.SCHEMA_VERSION


def test_schema_version_mismatch_quarantined(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    conn = sqlite3.connect(str(db))
    try:
        with conn:
            conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    finally:
        conn.close()
    assert store.ensure_db(db) is True
    assert len(_quarantine_files(tmp_path)) == 1
    snap = store.read_snapshot(db)
    assert snap is not None
    assert snap["schema_version"] == store.SCHEMA_VERSION


def test_locked_db_write_degrades(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    holder = sqlite3.connect(str(db))
    try:
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO concerns (text) VALUES ('lock holder')")
        start = time.monotonic()
        result = store.apply_deltas(
            {"concerns_add": [{"text": "blocked write"}]}, db, busy_timeout_ms=100
        )
        elapsed = time.monotonic() - start
        assert result is False
        assert elapsed < 2.0
    finally:
        holder.rollback()
        holder.close()


def test_locked_db_read_degrades(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True

    def raise_locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store.sqlite3, "connect", raise_locked)
    assert store.read_snapshot(db) is None


def test_caps_enforced(tmp_path):
    """STATE-04: caps 20/50/500 enforced; survivors are the most recent rows."""
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True

    assert store.apply_deltas(
        {"concerns_add": [{"text": "concern-%02d" % i} for i in range(30)]}, db
    ) is True
    kinds = ("semantic", "narrative", "relational", "emotional")
    assert store.apply_deltas(
        {
            "contradictions_add": [
                {"kind": kinds[i % 4], "description": "contra-%02d" % i}
                for i in range(60)
            ]
        },
        db,
    ) is True
    assert store.apply_deltas(
        {
            "turn_log_add": [
                {"session_id": "s", "turn_id": "turn-%03d" % i} for i in range(510)
            ]
        },
        db,
    ) is True

    snap = store.read_snapshot(db)
    assert snap is not None
    assert len(snap["concerns"]) <= 20
    assert len(snap["contradictions"]) <= 50
    assert snap["turn_log_count"] <= 500

    concern_texts = {c["text"] for c in snap["concerns"]}
    assert "concern-29" in concern_texts  # last-inserted survives
    assert "concern-00" not in concern_texts  # first-inserted evicted
    contra_descriptions = {c["description"] for c in snap["contradictions"]}
    assert "contra-59" in contra_descriptions
    assert "contra-00" not in contra_descriptions

    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        turn_ids = {r[0] for r in conn.execute("SELECT turn_id FROM turn_log")}
    finally:
        conn.close()
    assert "turn-509" in turn_ids
    assert "turn-000" not in turn_ids


def test_caps_enforced_trust_scores(tmp_path):
    """trust_scores capped at 64, evicting the oldest updated_at rows."""
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    # Two batches with distinct updated_at: the older batch should be evicted.
    assert store.apply_deltas(
        {"trust_scores": {"old-%02d" % i: 0.5 for i in range(40)}}, db
    ) is True
    time.sleep(0.01)  # ensure a later updated_at for the second batch
    assert store.apply_deltas(
        {"trust_scores": {"new-%02d" % i: 0.5 for i in range(40)}}, db
    ) is True
    snap = store.read_snapshot(db)
    assert snap is not None
    scores = snap["trust_scores"]
    assert len(scores) == 64
    assert all(("new-%02d" % i) in scores for i in range(40))  # newest all survive
    assert sum(1 for k in scores if k.startswith("old-")) == 24  # oldest evicted


def test_apply_deltas_unknown_keys_ignored(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    before = store.read_snapshot(db)
    assert store.apply_deltas({"bogus_key": [1, 2]}, db) is True
    after = store.read_snapshot(db)
    assert after == before
    assert after["concerns"] == []
    assert after["turn_log_count"] == 0
