"""Telemetry store (schema v2-era; current version = store.SCHEMA_VERSION) +
config module tests.

Every store test uses tmp_path with explicit db_path= — the real
$HERMES_HOME is never touched. Direct sqlite3 use is test instrumentation
only; plugin code goes through store.py exclusively.
"""

import sqlite3

import pytest

from hexis_appraisal import config, store


def _quarantine_files(tmp_path):
    return [
        p
        for p in tmp_path.glob("state.db.quarantined-*")
        if not (p.name.endswith("-wal") or p.name.endswith("-shm"))
    ]


def _create_v1_db(db):
    """Build a structurally-valid v1 DB (no telemetry table, version 1)."""
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
            conn.execute("CREATE TABLE concerns (id INTEGER PRIMARY KEY, text TEXT)")
            conn.execute(
                "CREATE TABLE contradictions (id INTEGER PRIMARY KEY, kind TEXT,"
                " description TEXT)"
            )
            conn.execute(
                "CREATE TABLE trust_scores (key TEXT PRIMARY KEY, value REAL)"
            )
            conn.execute("CREATE TABLE turn_log (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema quarantine (v1 -> current)
# ---------------------------------------------------------------------------


def test_v1_db_quarantined_and_recreated_at_current_schema(tmp_path):
    db = tmp_path / "state.db"
    _create_v1_db(db)
    assert store.ensure_db(db) is True
    assert len(_quarantine_files(tmp_path)) == 1
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "telemetry" in tables
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert row == (str(store.SCHEMA_VERSION),)
    finally:
        conn.close()
    assert store.CAPS["telemetry"] == 2000


# ---------------------------------------------------------------------------
# record_telemetry
# ---------------------------------------------------------------------------


def test_record_telemetry_full_row(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.record_telemetry(
        "ok",
        wall_ms=812,
        model="fake-model",
        tokens_in=120,
        tokens_out=80,
        error=None,
        session_id="s1",
        db_path=db,
    ) is True
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        row = conn.execute(
            "SELECT ts, session_id, wall_ms, model, tokens_in, tokens_out,"
            " outcome, error FROM telemetry"
        ).fetchone()
    finally:
        conn.close()
    assert row[0]  # ts populated
    assert row[1:] == ("s1", 812, "fake-model", 120, 80, "ok", None)


def test_record_telemetry_outcome_only_and_error_truncation(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    assert store.record_telemetry("skipped:disabled", db_path=db) is True
    assert store.record_telemetry(
        "llm_error", error="x" * 1000, db_path=db
    ) is True
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        rows = conn.execute(
            "SELECT outcome, error FROM telemetry ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert rows[0] == ("skipped:disabled", None)
    assert rows[1][0] == "llm_error"
    assert len(rows[1][1]) == 300


def test_telemetry_cap_evicts_oldest(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    # Direct bulk insert is test instrumentation (plan-sanctioned).
    conn = sqlite3.connect(str(db))
    try:
        with conn:
            conn.executemany(
                "INSERT INTO telemetry (ts, outcome) VALUES (?, ?)",
                [("t", "ok-%04d" % i) for i in range(2010)],
            )
    finally:
        conn.close()
    assert store.record_telemetry("ok", db_path=db) is True
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        count = conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
        oldest = conn.execute(
            "SELECT COUNT(*) FROM telemetry WHERE outcome='ok-0000'"
        ).fetchone()[0]
        newest = conn.execute(
            "SELECT COUNT(*) FROM telemetry WHERE outcome='ok'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 2000
    assert oldest == 0  # oldest evicted
    assert newest == 1  # newest survives


def test_record_telemetry_absent_or_corrupt_path_returns_false(tmp_path):
    absent = tmp_path / "missing-dir" / "state.db"
    assert store.record_telemetry("ok", db_path=absent) is False
    corrupt = tmp_path / "state.db"
    corrupt.write_bytes(b"this is not a sqlite database")
    assert store.record_telemetry("ok", db_path=corrupt) is False


# ---------------------------------------------------------------------------
# telemetry_summary
# ---------------------------------------------------------------------------


def test_telemetry_summary_counts_and_p50(tmp_path):
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    store.record_telemetry("ok", wall_ms=120, db_path=db)
    store.record_telemetry("skipped:social_close", db_path=db)
    store.record_telemetry("timeout", error="deadline hit", db_path=db)
    store.record_telemetry("ok", wall_ms=180, db_path=db)
    store.record_telemetry("llm_error", error="boom", db_path=db)
    store.record_telemetry("trust_fallback", wall_ms=240, db_path=db)

    summary = store.telemetry_summary(db)
    assert summary is not None
    assert summary["total"] == 6
    assert summary["by_outcome"] == {
        "ok": 2,
        "skipped:social_close": 1,
        "timeout": 1,
        "llm_error": 1,
        "trust_fallback": 1,
    }
    assert summary["failure_count"] == 2  # timeout + llm_error only
    assert summary["last_error"] == "boom"  # newest failure row
    assert summary["p50_wall_ms"] == 180  # median of [120, 180, 240]


def test_telemetry_summary_reflect_vocabulary(tmp_path):
    """Post-Phase-3 vocabulary (03-VERIFICATION finding 1): reflect_ok and
    reflect_skipped:* are NOT failures; reflect_timeout (and the other
    reflect_* error outcomes) are. p50_wall_ms stays appraisal-only."""
    db = tmp_path / "state.db"
    assert store.ensure_db(db) is True
    store.record_telemetry("ok", wall_ms=120, db_path=db)
    store.record_telemetry("timeout", error="deadline hit", db_path=db)
    store.record_telemetry("reflect_ok", wall_ms=4400, db_path=db)
    store.record_telemetry("reflect_skipped:debounce", db_path=db)
    store.record_telemetry(  # newest row, distinct error string
        "reflect_timeout", error="reflect deadline 8.0s exceeded", db_path=db
    )

    summary = store.telemetry_summary(db)
    assert summary is not None
    assert summary["total"] == 5
    assert summary["by_outcome"] == {
        "ok": 1,
        "timeout": 1,
        "reflect_ok": 1,
        "reflect_skipped:debounce": 1,
        "reflect_timeout": 1,
    }
    assert summary["failure_count"] == 2  # timeout + reflect_timeout only
    assert summary["last_error"] == "reflect deadline 8.0s exceeded"
    assert summary["p50_wall_ms"] == 120  # ok row only; reflect_ok wall excluded


def test_telemetry_summary_absent_db_returns_none(tmp_path):
    assert store.telemetry_summary(tmp_path / "absent" / "state.db") is None
    assert not (tmp_path / "absent" / "state.db").exists()  # never creates


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_config_cache():
    config.reset_cache()
    yield
    config.reset_cache()


def test_get_cfg_defaults_when_host_config_unavailable(monkeypatch):
    monkeypatch.setattr(config, "_load_host_entry", lambda: None)
    cfg = config.get_cfg(force_reload=True)
    assert cfg == {
        "enabled": True,
        "confidence_threshold": 0.6,
        "deadline_seconds": 8.0,
        "history_chars": 4000,
        "model": None,
        "max_tokens": 700,
        "reflection_enabled": True,
        "reflect_every_n_turns": 5,
        "reflect_max_tokens": 700,
        "reflect_deadline_seconds": 8.0,
    }
    # SAFE-01 (R1, 2026-06-10): the in-code default deadline is 8.0s.
    assert cfg["deadline_seconds"] == config.DEFAULT_DEADLINE_SECONDS == 8.0


def test_get_cfg_reads_entry_and_clamps(monkeypatch):
    monkeypatch.setattr(
        config,
        "_load_host_entry",
        lambda: {
            "enabled": False,
            "confidence_threshold": 1.7,  # clamp -> 1.0
            "deadline_seconds": 99,  # clamp -> 10.0
            "history_chars": 1234,
            "max_tokens": "not-an-int",  # default -> 700
            "llm": {"model": " gpt-4o-mini "},
        },
    )
    cfg = config.get_cfg(force_reload=True)
    assert cfg["enabled"] is False
    assert cfg["confidence_threshold"] == 1.0
    assert cfg["deadline_seconds"] == 10.0
    assert cfg["history_chars"] == 1234
    assert cfg["max_tokens"] == 700
    assert cfg["model"] == "gpt-4o-mini"

    monkeypatch.setattr(
        config,
        "_load_host_entry",
        lambda: {"confidence_threshold": -0.2, "deadline_seconds": 0.05},
    )
    cfg = config.get_cfg(force_reload=True)
    assert cfg["confidence_threshold"] == 0.0
    assert cfg["deadline_seconds"] == 0.5

    # Reflection keys (REFL-01): coerced + clamped like everything else.
    monkeypatch.setattr(
        config,
        "_load_host_entry",
        lambda: {
            "reflection_enabled": "off",       # recognized string -> False
            "reflect_every_n_turns": 999,      # clamp -> 50
            "reflect_max_tokens": "garbage",   # default -> 700
            "reflect_deadline_seconds": 0.05,  # clamp -> 0.5
        },
    )
    cfg = config.get_cfg(force_reload=True)
    assert cfg["reflection_enabled"] is False
    assert cfg["reflect_every_n_turns"] == 50
    assert cfg["reflect_max_tokens"] == 700
    assert cfg["reflect_deadline_seconds"] == 0.5
    monkeypatch.setattr(
        config, "_load_host_entry", lambda: {"reflect_every_n_turns": 0}
    )
    assert config.get_cfg(force_reload=True)["reflect_every_n_turns"] == 1


def test_get_cfg_cached_until_reset(monkeypatch):
    monkeypatch.setattr(config, "_load_host_entry", lambda: {"history_chars": 111})
    first = config.get_cfg(force_reload=True)
    assert first["history_chars"] == 111
    monkeypatch.setattr(config, "_load_host_entry", lambda: {"history_chars": 222})
    assert config.get_cfg()["history_chars"] == 111  # cached
    config.reset_cache()
    assert config.get_cfg()["history_chars"] == 222  # re-read after reset
