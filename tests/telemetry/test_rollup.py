"""rollup tests: tel_* -> per-run summary events with REAL values (local only)."""

from __future__ import annotations

import sqlite3
import time

import hermes_state
from agent.telemetry import rollup
from agent.telemetry.emitter import TelemetryEmitter
from agent.telemetry.events import ModelCallEvent, RunEvent, ToolCallEvent


def _seed(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(hermes_state.SCHEMA_SQL)
    conn.close()
    em = TelemetryEmitter(events_path=tmp_path / "tel" / "e.jsonl", db_path=db)
    now = time.time_ns()
    em.emit(RunEvent(run_id="r1", trace_id="t1", entrypoint="gateway",
                     platform="telegram", end_reason="completed",
                     start_ns=now - 90_000_000, end_ns=now,
                     model_call_count=2, tool_call_count=2, estimated_cost_usd=2.1))
    em.emit(ModelCallEvent(span_id="m1", run_id="r1", provider="anthropic",
                           model="claude-opus-4", input_tokens=60000, output_tokens=8000))
    em.emit(ModelCallEvent(span_id="m2", run_id="r1", provider="anthropic",
                           model="claude-opus-4", input_tokens=5000, output_tokens=500))
    em.emit(ToolCallEvent(span_id="tc1", run_id="r1", tool_name="web_search",
                          result_class="ok"))
    em.emit(ToolCallEvent(span_id="tc2", run_id="r1", tool_name="browser_navigate",
                          result_class="ok"))
    # an in-progress run (no end_ns) must be excluded
    em.emit(RunEvent(run_id="r2", trace_id="t2", entrypoint="cli", start_ns=now))
    em.flush()
    em.close()
    return db


def test_builds_one_event_per_completed_run_with_real_values(tmp_path):
    db = _seed(tmp_path)
    events = rollup.build_aggregate_events(install_id="fixed-id", db_path=db,
                                           include_heartbeat=False)
    wf = [e for e in events if e["event_name"] == "workflow_completed"]
    assert len(wf) == 1  # r2 (no end_ns) excluded
    e = wf[0]
    assert e["entrypoint"] == "gateway"
    assert e["platform"] == "telegram"
    # REAL model id + provider, not a bucket/class
    models = {m["model"] for m in e["models_used"]}
    assert models == {"claude-opus-4"}
    assert e["models_used"][0]["provider"] == "anthropic"
    assert sorted(e["tools_used"]) == ["browser_navigate", "web_search"]
    # real token totals, not buckets
    assert e["input_tokens"] == 65000
    assert e["output_tokens"] == 8500


def test_real_model_and_tool_names_present(tmp_path):
    db = _seed(tmp_path)
    events = rollup.build_aggregate_events(install_id="fixed-id", db_path=db)
    blob = " ".join(str(v) for e in events for v in e.values())
    assert "claude-opus-4" in blob
    assert "web_search" in blob


def test_heartbeat_included_by_default(tmp_path):
    db = _seed(tmp_path)
    events = rollup.build_aggregate_events(install_id="fixed-id", db_path=db)
    assert any(e["event_name"] == "heartbeat" for e in events)


def test_summarize_counts_by_event_name(tmp_path):
    db = _seed(tmp_path)
    events = rollup.build_aggregate_events(install_id="fixed-id", db_path=db)
    s = rollup.summarize(events)
    assert s["total"] == len(events)
    assert s["by_event_name"]["workflow_completed"] == 1
    assert s["by_event_name"]["heartbeat"] == 1


def test_empty_db_yields_only_heartbeat(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(hermes_state.SCHEMA_SQL)
    conn.close()
    events = rollup.build_aggregate_events(install_id="x", db_path=db)
    assert [e["event_name"] for e in events] == ["heartbeat"]
