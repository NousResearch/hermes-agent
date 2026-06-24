"""Emitter tests — the hot-path invariant is the one that matters most.

Invariant: emit() never blocks, never raises, and a broken writer cannot slow or
break the caller. Plus: JSONL + SQLite round-trip, and the SQLite index is rebuildable
from the JSONL source of truth.
"""

from __future__ import annotations

import sqlite3
import time

import hermes_state
from agent.telemetry.emitter import TelemetryEmitter
from agent.telemetry.events import ModelCallEvent, RunEvent, ToolCallEvent


def _fresh_db(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(hermes_state.SCHEMA_SQL)
    conn.close()
    return db


def test_emit_is_fast_even_when_writer_is_broken(tmp_path, monkeypatch):
    """The core guarantee: a writer that raises AND sleeps cannot stall emit()."""
    db = _fresh_db(tmp_path)
    em = TelemetryEmitter(events_path=tmp_path / "telemetry" / "events.jsonl", db_path=db)

    # Sabotage the row indexer to raise after a long sleep.
    def broken(_conn, _ev):
        time.sleep(5.0)
        raise RuntimeError("writer exploded")

    monkeypatch.setattr(em, "_index_one", broken)

    start = time.monotonic()
    for i in range(50):
        em.emit(ModelCallEvent(span_id=f"s{i}", run_id="r1", input_tokens=10))
    elapsed = time.monotonic() - start

    # 50 emits must complete in well under the writer's single 5s sleep.
    assert elapsed < 1.0, f"emit() blocked: {elapsed:.2f}s"
    em.close()


def test_emit_never_raises_on_bad_event(tmp_path):
    db = _fresh_db(tmp_path)
    em = TelemetryEmitter(events_path=tmp_path / "telemetry" / "events.jsonl", db_path=db)
    # Non-serializable / wrong-shaped inputs must not raise out of emit().
    em.emit(object())            # no to_dict, not a mapping
    em.emit({"event": "run"})    # minimal dict
    em.close()


def test_jsonl_and_sqlite_roundtrip(tmp_path):
    db = _fresh_db(tmp_path)
    jsonl = tmp_path / "telemetry" / "events.jsonl"
    em = TelemetryEmitter(events_path=jsonl, db_path=db)

    em.emit(RunEvent(run_id="run1", trace_id="t1", entrypoint="cli", end_reason="completed"))
    em.emit(ModelCallEvent(span_id="m1", run_id="run1", provider="anthropic",
                           model="claude-opus-4", input_tokens=100, output_tokens=20))
    em.emit(ToolCallEvent(span_id="tc1", run_id="run1", tool_name="web_search",
                          duration_ms=120, result_class="ok"))
    em.flush()
    em.close()

    # JSONL has all three lines
    lines = [l for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3

    # SQLite index has the rows in the right tables
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM tel_runs").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM tel_model_calls").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM tel_tool_calls").fetchone()[0] == 1
    row = conn.execute("SELECT provider, model, input_tokens FROM tel_model_calls").fetchone()
    assert row == ("anthropic", "claude-opus-4", 100)
    conn.close()


def test_unknown_event_kind_is_ignored_not_fatal(tmp_path):
    db = _fresh_db(tmp_path)
    em = TelemetryEmitter(events_path=tmp_path / "telemetry" / "events.jsonl", db_path=db)
    em.emit({"event": "totally_unknown", "foo": "bar"})
    em.emit(RunEvent(run_id="r2", trace_id="t2", entrypoint="cli"))
    em.flush()
    em.close()
    conn = sqlite3.connect(db)
    # The unknown event is in JSONL but skipped by the indexer; the known one indexes.
    assert conn.execute("SELECT COUNT(*) FROM tel_runs").fetchone()[0] == 1
    conn.close()


def test_disabled_emitter_writes_nothing(tmp_path):
    db = _fresh_db(tmp_path)
    jsonl = tmp_path / "telemetry" / "events.jsonl"
    em = TelemetryEmitter(events_path=jsonl, db_path=db, enabled=False)
    em.emit(RunEvent(run_id="r3", trace_id="t3", entrypoint="cli"))
    em.flush()
    em.close()
    assert not jsonl.exists()
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM tel_runs").fetchone()[0] == 0
    conn.close()
