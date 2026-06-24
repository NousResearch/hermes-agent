"""Insights ↔ telemetry integration: the observability section in /insights output."""

from __future__ import annotations

import time

import pytest

from hermes_state import SessionDB
from agent.insights import InsightsEngine
from agent.telemetry.emitter import TelemetryEmitter
from agent.telemetry.events import ModelCallEvent, RunEvent, ToolCallEvent


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "ins_tel.db")
    yield session_db
    session_db.close()


def _seed_telemetry(db_path):
    em = TelemetryEmitter(events_path=db_path.parent / "tel" / "events.jsonl", db_path=db_path)
    now = time.time_ns()
    em.emit(RunEvent(run_id="r1", trace_id="t1", entrypoint="gateway",
                     platform="telegram", end_reason="completed",
                     start_ns=now - 90_000_000, end_ns=now))
    em.emit(RunEvent(run_id="r2", trace_id="t2", entrypoint="cli",
                     end_reason="failed", start_ns=now - 11_000_000, end_ns=now))
    em.emit(ModelCallEvent(span_id="m1", run_id="r1", provider="anthropic",
                           model="claude-opus-4", input_tokens=5000, output_tokens=800,
                           cache_read_tokens=1000, latency_ms=2200))
    em.emit(ToolCallEvent(span_id="tc1", run_id="r1", tool_name="web_search", result_class="ok"))
    em.emit(ToolCallEvent(span_id="tc2", run_id="r1", tool_name="browser_navigate", result_class="error"))
    em.flush()
    em.close()


def test_report_includes_telemetry_when_present(db):
    # A session so generate() isn't the empty branch
    db.create_session(session_id="s1", source="cli", model="anthropic/claude-sonnet-4")
    _seed_telemetry(db.db_path)

    engine = InsightsEngine(db)
    report = engine.generate(days=30)
    tel = report.get("telemetry")
    assert tel, "telemetry section missing"
    assert tel["workflows"]["total_runs"] == 2
    assert tel["workflows"]["success_rate"] == 0.5
    assert tel["tool_calls"]["total"] == 2
    assert tel["tool_calls"]["failure_rate"] == 0.5
    assert tel["model_calls"]["by_provider"]["anthropic"] == 1


def test_terminal_output_renders_observability_section(db):
    db.create_session(session_id="s1", source="cli", model="anthropic/claude-sonnet-4")
    _seed_telemetry(db.db_path)

    engine = InsightsEngine(db)
    out = engine.format_terminal(engine.generate(days=30))
    assert "Observability" in out
    assert "Workflows:" in out
    assert "Failure rate:" in out
    assert "Providers:" in out


def test_telemetry_section_absent_when_no_tel_rows(db):
    # Session present, but no telemetry events seeded.
    db.create_session(session_id="s1", source="cli", model="anthropic/claude-sonnet-4")
    engine = InsightsEngine(db)
    report = engine.generate(days=30)
    assert report.get("telemetry") == {}
    out = engine.format_terminal(report)
    assert "Observability" not in out


def test_empty_report_has_telemetry_key(db):
    # No sessions at all -> empty branch still carries the key (renderer-safe).
    engine = InsightsEngine(db)
    report = engine.generate(days=30)
    assert report.get("empty") is True
    assert report.get("telemetry") == {}
