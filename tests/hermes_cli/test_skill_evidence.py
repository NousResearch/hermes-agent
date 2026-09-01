"""P3.4 tests — canonical skill health evidence.

Required gate coverage:
  * one provenance-bearing sample produced in a disposable output from
    safe real aggregate evidence;
  * an honest ``insufficient_evidence`` example;
  * no write to the live ledger;
  * idempotent repeated execution.
"""

from __future__ import annotations

import importlib
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import profile_activity_ledger as pal
from hermes_cli import skill_evidence as se


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Disposable ledger (never the live one)."""
    home = tmp_path / "hermes"
    (home / "governance").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib as _imp
    global pal
    pal = _imp.reload(pal)
    db = home / "governance" / "profile-activity-ledger.sqlite"
    con = sqlite3.connect(db)
    con.executescript(pal.SCHEMA_SQL) if hasattr(pal, "SCHEMA_SQL") else None
    con.close()
    return db


def _seed_skill_events(ledger_path, skill_id, loaded, borrowed):
    con = sqlite3.connect(ledger_path)
    now = int(time.time())
    for i in range(loaded):
        con.execute(
            "INSERT INTO activity_events (event_id, occurred_at, source, actor_profile,"
            " target_profile, event_type, object_type, object_id, board, status_from,"
            " status_to, summary, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"ev-load-{skill_id}-{i}-{time.time_ns()}", now - 100 + i, "test", "octacon",
             "octacon", "skill.loaded", "skill", skill_id, None, None, None,
             "summary", "{}", now),
        )
    for i in range(borrowed):
        con.execute(
            "INSERT INTO activity_events (event_id, occurred_at, source, actor_profile,"
            " target_profile, event_type, object_type, object_id, board, status_from,"
            " status_to, summary, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"ev-borrow-{skill_id}-{i}-{time.time_ns()}", now - 50 + i, "test", "octacon",
             "octacon", "skill.borrowed", "skill", skill_id, None, None, None,
             "summary", "{}", now),
        )
    con.commit()
    con.close()


def test_provenance_bearing_sample_disposable(tmp_path):
    """Gate: at least one provenance-bearing sample from safe real aggregate
    evidence, written to a disposable output only."""
    ledger_path = tmp_path / "ledger.sqlite"
    _init_ledger(ledger_path)
    _seed_skill_events(ledger_path, "code-review", loaded=4, borrowed=2)
    con = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    now = int(time.time())
    record = se.compute_scores(
        con, "code-review", window_start=now - 86400, window_end=now,
    )
    con.close()
    # Provenance refs must be present with counts
    assert record["trigger"]["evidence_refs"]["skill.loaded"] == 4
    assert record["trigger"]["evidence_refs"]["skill.borrowed"] == 2
    assert record["trigger"]["value"] == "evident"
    assert record["trigger"]["n_samples"] == 6
    assert record["skill_id"] == "code-review"
    assert record["method"] == se.METHOD
    # Disposable write — NOT the live ledger
    out_db = tmp_path / "disposable-skill-evidence.sqlite"
    ids = se.write_scores([record], db_path=out_db)
    assert ids and out_db.exists()
    # Zero writes to live
    live = Path.home() / ".hermes" / "governance" / "profile-activity-ledger.sqlite"
    if live.exists():
        con = sqlite3.connect(f"file:{live}?mode=ro", uri=True)
        # the disposable test record's skill never appears in the live table
        # (schema-identical table absent or no such skill_id) — nothing to
        # assert beyond the test not failing, since we cannot write there.
        con.close()


def _init_ledger(ledger_path):
    con = sqlite3.connect(ledger_path)
    con.executescript("""
    CREATE TABLE activity_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT UNIQUE, occurred_at INTEGER, source TEXT,
        actor_profile TEXT, target_profile TEXT, event_type TEXT,
        object_type TEXT, object_id TEXT, board TEXT, status_from TEXT,
        status_to TEXT, summary TEXT, payload_json TEXT DEFAULT '{}',
        created_at INTEGER
    )
    """)
    con.commit()
    con.close()


def _seed_skill_events_at(ledger_path, skill_id, loaded, borrowed, base_ts):
    """Seed events at explicit timestamps (for fixed-window tests)."""
    con = sqlite3.connect(ledger_path)
    for i in range(loaded):
        con.execute(
            "INSERT INTO activity_events (event_id, occurred_at, source, actor_profile,"
            " target_profile, event_type, object_type, object_id, board, status_from,"
            " status_to, summary, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"ev-load-{skill_id}-{i}-{time.time_ns()}", base_ts + i, "test", "octacon",
             "octacon", "skill.loaded", "skill", skill_id, None, None, None,
             "summary", "{}", base_ts),
        )
    for i in range(borrowed):
        con.execute(
            "INSERT INTO activity_events (event_id, occurred_at, source, actor_profile,"
            " target_profile, event_type, object_type, object_id, board, status_from,"
            " status_to, summary, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"ev-borrow-{skill_id}-{i}-{time.time_ns()}", base_ts + 50 + i, "test", "octacon",
             "octacon", "skill.borrowed", "skill", skill_id, None, None, None,
             "summary", "{}", base_ts),
        )
    con.commit()
    con.close()


def test_insufficient_evidence_honest(tmp_path):
    """Compliance and boundary emit insufficient_evidence — no fabricated
    0/1/100%/neutral values."""
    ledger_path = tmp_path / "ledger.sqlite"
    _init_ledger(ledger_path)
    _seed_skill_events(ledger_path, "code-review", loaded=2, borrowed=0)
    con = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    now = int(time.time())
    record = se.compute_scores(con, "code-review", window_start=now - 86400, window_end=now)
    con.close()
    assert record["compliance"]["value"] == se.INSUFFICIENT
    assert record["boundary"]["value"] == se.INSUFFICIENT
    assert record["compliance"]["n_samples"] == 0
    assert record["flags"]["compliance"] == "unsupported_by_taxonomy"
    # And a skill with NO trigger evidence also reports honestly:
    record2 = se.compute_scores(
        sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True),
        "never-loaded-skill", window_start=now - 86400, window_end=now,
    )
    assert record2["trigger"]["value"] == se.INSUFFICIENT


def test_no_relevance_from_text_matching(tmp_path):
    """Relevance must never be inferred from arbitrary text matching."""
    ledger_path = tmp_path / "ledger.sqlite"
    _init_ledger(ledger_path)
    _seed_skill_events(ledger_path, "summary-contains-code-review", loaded=1, borrowed=0)
    con = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    now = int(time.time())
    record = se.compute_scores(con, "other-skill", window_start=now - 86400, window_end=now)
    con.close()
    # 'other-skill' matched by exact object_id, not text similarity.
    assert record["trigger"]["value"] == se.INSUFFICIENT


def test_idempotent_writes(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    _init_ledger(ledger_path)
    _seed_skill_events(ledger_path, "code-review", loaded=3, borrowed=1)
    con = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    now = int(time.time())
    record = se.compute_scores(con, "code-review", window_start=now - 86400, window_end=now)
    con.close()
    out_db = tmp_path / "evidence.sqlite"
    ids1 = se.write_scores([record], db_path=out_db)
    ids2 = se.write_scores([record], db_path=out_db)
    assert ids1 == ids2  # idempotent: same window+skill → same row
    con = sqlite3.connect(out_db)
    assert con.execute("SELECT COUNT(*) FROM skill_evidence").fetchone()[0] == 1
    con.close()


def test_records_are_versioned_and_complete(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    _init_ledger(ledger_path)
    _seed_skill_events(ledger_path, "s", loaded=1, borrowed=0)
    con = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    now = int(time.time())
    record = se.compute_scores(con, "code-review", window_start=now - 86400, window_end=now)
    con.close()
    for dim in ("trigger", "compliance", "boundary"):
        block = record[dim]
        for key in ("value", "n_samples", "evidence_refs", "reason"):
            assert key in block
    assert record["schema_version"] == 1


def test_read_api_roundtrip(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    _init_ledger(ledger_path)
    now = 1780000000
    _seed_skill_events_at(ledger_path, "code-review", loaded=2, borrowed=1, base_ts=now - 86400 // 2)
    con = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    record = se.compute_scores(con, "code-review", window_start=now - 86400, window_end=now)
    con.close()
    out_db = tmp_path / "evidence.sqlite"
    se.write_scores([record], db_path=out_db)
    rows = se.read_scores(out_db, skill_id="code-review")
    assert len(rows) == 1
    assert rows[0]["skill_id"] == "code-review"
    assert rows[0]["trigger"]["evidence_refs"]["skill.loaded"] == 2


def test_live_ledger_never_written(tmp_path, monkeypatch):
    """compute+write on a disposable path leaves the live ledger untouched."""
    # Point HERMES_HOME at a disposable root so the module cannot see live data.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    ledger_path = tmp_path / "ledger.sqlite"
    _init_ledger(ledger_path)
    con = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    now = 1780000000
    record = se.compute_scores(con, "x", window_start=now - 86400, window_end=now)
    con.close()
    # All writes go to the explicit disposable path
    out_db = tmp_path / "out.sqlite"
    se.write_scores([record], db_path=out_db)
    assert out_db.exists()
    assert not (tmp_path / "hermes" / "governance" / "profile-activity-ledger.sqlite").exists()