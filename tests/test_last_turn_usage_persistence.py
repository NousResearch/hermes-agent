"""Tests for per-turn (last-turn) usage persistence on the sessions row.

Layer 1 of the "last-turn stats survive eviction" feature: the cumulative
session_* counters already persist, but the *last turn's* split (input /
output / cache_read / cache_write / reasoning) lived only on the ephemeral
AIAgent instance and was lost when the idle sweep evicted it.

These tests assert:
  1. The sessions table has last_turn_* columns (declarative migration).
  2. update_token_counts() writes a snapshot when last_turn_* values are passed.
  3. The snapshot OVERWRITES (not accumulates) across turns.
  4. get_last_turn_usage() reads the snapshot back (survives a fresh SessionDB
     handle == survives agent eviction).
  5. Omitting last_turn_* leaves any existing snapshot untouched (back-compat).
"""

from pathlib import Path

import pytest

from hermes_state import SessionDB

LAST_TURN_COLUMNS = [
    "last_turn_input_tokens",
    "last_turn_output_tokens",
    "last_turn_cache_read_tokens",
    "last_turn_cache_write_tokens",
    "last_turn_reasoning_tokens",
]


@pytest.fixture
def db(tmp_path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _columns(db: SessionDB) -> set:
    def _do(conn):
        rows = conn.execute('PRAGMA table_info("sessions")').fetchall()
        return {r[1] if isinstance(r, (tuple, list)) else r["name"] for r in rows}
    return db._execute_write(_do)


def test_schema_has_last_turn_columns(db):
    cols = _columns(db)
    for c in LAST_TURN_COLUMNS:
        assert c in cols, f"missing column {c}"


def test_update_writes_last_turn_snapshot(db):
    sid = "telegram-1"
    db.update_token_counts(
        sid,
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=80,
        cache_write_tokens=5,
        reasoning_tokens=10,
        last_turn_input_tokens=100,
        last_turn_output_tokens=20,
        last_turn_cache_read_tokens=80,
        last_turn_cache_write_tokens=5,
        last_turn_reasoning_tokens=10,
    )
    snap = db.get_last_turn_usage(sid)
    assert snap == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 80,
        "cache_write_tokens": 5,
        "reasoning_tokens": 10,
    }


def test_snapshot_overwrites_across_turns(db):
    sid = "telegram-2"
    # Turn 1
    db.update_token_counts(
        sid, input_tokens=100, output_tokens=20,
        last_turn_input_tokens=100, last_turn_output_tokens=20,
        last_turn_cache_read_tokens=0, last_turn_cache_write_tokens=0,
        last_turn_reasoning_tokens=0,
    )
    # Turn 2 — cumulative input grows to 350, but last-turn snapshot is just 250.
    db.update_token_counts(
        sid, input_tokens=250, output_tokens=40,
        last_turn_input_tokens=250, last_turn_output_tokens=40,
        last_turn_cache_read_tokens=200, last_turn_cache_write_tokens=0,
        last_turn_reasoning_tokens=0,
    )
    snap = db.get_last_turn_usage(sid)
    assert snap["input_tokens"] == 250  # overwritten, NOT 350
    assert snap["output_tokens"] == 40
    assert snap["cache_read_tokens"] == 200

    # Cumulative totals still accumulate (regression guard).
    row = db.get_session(sid)
    assert row["input_tokens"] == 350


def test_snapshot_survives_fresh_db_handle(tmp_path):
    """A new SessionDB on the same file == agent eviction. Snapshot must persist."""
    path = tmp_path / "state.db"
    sid = "gateway-1"
    db1 = SessionDB(db_path=path)
    db1.update_token_counts(
        sid, input_tokens=500, output_tokens=60,
        last_turn_input_tokens=500, last_turn_output_tokens=60,
        last_turn_cache_read_tokens=450, last_turn_cache_write_tokens=0,
        last_turn_reasoning_tokens=12,
    )
    # Brand-new handle — no in-memory agent state carried over.
    db2 = SessionDB(db_path=path)
    snap = db2.get_last_turn_usage(sid)
    assert snap["input_tokens"] == 500
    assert snap["reasoning_tokens"] == 12


def test_update_without_snapshot_leaves_existing(db):
    sid = "telegram-3"
    db.update_token_counts(
        sid, input_tokens=100, output_tokens=20,
        last_turn_input_tokens=100, last_turn_output_tokens=20,
        last_turn_cache_read_tokens=0, last_turn_cache_write_tokens=0,
        last_turn_reasoning_tokens=0,
    )
    # A later call that does NOT pass last_turn_* (e.g. a path that didn't opt in)
    # must not clobber the snapshot to zero.
    db.update_token_counts(sid, input_tokens=10, output_tokens=2)
    snap = db.get_last_turn_usage(sid)
    assert snap["input_tokens"] == 100  # untouched


def test_get_last_turn_usage_missing_session(db):
    assert db.get_last_turn_usage("does-not-exist") is None
