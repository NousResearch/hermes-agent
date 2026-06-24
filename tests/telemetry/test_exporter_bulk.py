"""Bulk export tests — telemetry always, content only behind the trajectories gate."""

from __future__ import annotations

import io
import json
import sqlite3
import time

import hermes_state
from agent.telemetry import exporter_bulk
from agent.telemetry.emitter import TelemetryEmitter
from agent.telemetry.events import ModelCallEvent, RunEvent, ToolCallEvent


def _seed(tmp_path, with_secret_content=True):
    db = tmp_path / "state.db"
    sdb = hermes_state.SessionDB(db_path=db)
    # telemetry
    em = TelemetryEmitter(events_path=tmp_path / "tel" / "e.jsonl", db_path=db)
    now = time.time_ns()
    em.emit(RunEvent(run_id="r1", trace_id="t1", entrypoint="cli", end_reason="completed",
                     start_ns=now - 60_000_000, end_ns=now, model_call_count=1, tool_call_count=1))
    em.emit(ModelCallEvent(span_id="m1", run_id="r1", provider="anthropic",
                           model="claude-sonnet-4", input_tokens=1000, output_tokens=100))
    em.emit(ToolCallEvent(span_id="w1", run_id="r1", tool_name="web_search", result_class="ok"))
    em.flush()
    em.close()
    # session + message content (with an embedded secret + email)
    sdb.create_session(session_id="s1", source="cli", model="anthropic/claude-sonnet-4")
    if with_secret_content:
        sdb.append_message("s1", role="user",
                           content="my key is AKIAIOSFODNN7EXAMPLE and email me at carol@corp.com")
        sdb.append_message("s1", role="assistant", content="ok")
    sdb.close()
    return db


def test_telemetry_exported_content_excluded_by_default(tmp_path):
    db = _seed(tmp_path)
    buf = io.StringIO()
    counts = exporter_bulk.export(buf, fmt="ndjson", include_content=False, config={}, db_path=db)
    assert counts["telemetry"] >= 3
    assert counts["content_included"] == 0
    text = buf.getvalue()
    # message bodies must NOT be present
    assert "carol@corp.com" not in text
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    # but a session record (structural) is present with message structure
    lines = [json.loads(l) for l in text.splitlines() if l.strip()]
    sess = [r for r in lines if r.get("_kind") == "session"]
    assert sess and sess[0]["messages"]
    assert "content" not in sess[0]["messages"][0]   # structural only
    assert "content_chars" in sess[0]["messages"][0]


def test_include_content_ignored_without_trajectories(tmp_path):
    db = _seed(tmp_path)
    buf = io.StringIO()
    # request content but trajectories disabled -> forced off
    counts = exporter_bulk.export(buf, fmt="ndjson", include_content=True,
                                  config={"telemetry": {"trajectories": {"enabled": False}}}, db_path=db)
    assert counts["content_included"] == 0
    assert "carol@corp.com" not in buf.getvalue()


def test_content_included_when_trajectories_on_but_redacted(tmp_path):
    db = _seed(tmp_path)
    buf = io.StringIO()
    cfg = {"telemetry": {"trajectories": {"enabled": True}, "content_redaction": "pii"}}
    counts = exporter_bulk.export(buf, fmt="ndjson", include_content=True, config=cfg, db_path=db)
    assert counts["content_included"] == 1
    text = buf.getvalue()
    # content present but secret + pii scrubbed
    assert "AKIAIOSFODNN7EXAMPLE" not in text           # secret always gone
    assert "carol@corp.com" not in text      # pii gone in pii mode
    lines = [json.loads(l) for l in text.splitlines() if l.strip()]
    sess = [r for r in lines if r.get("_kind") == "session"][0]
    # the user message now has a (redacted) content field
    user_msg = [m for m in sess["messages"] if m["role"] == "user"][0]
    assert "content" in user_msg
    assert "[email]" in user_msg["content"]


def test_json_format_roundtrips(tmp_path):
    db = _seed(tmp_path)
    buf = io.StringIO()
    exporter_bulk.export(buf, fmt="json", include_content=False, config={}, db_path=db)
    obj = json.loads(buf.getvalue())
    assert "records" in obj
    assert any(r["_kind"] == "tel_runs" for r in obj["records"])


def test_since_window_filters_runs(tmp_path):
    db = _seed(tmp_path)
    buf = io.StringIO()
    # since 1ns ago in the future-ish -> the run (start ~60ms ago) excluded
    future_ns = int((time.time() + 1) * 1e9)
    counts = exporter_bulk.export(buf, fmt="ndjson", since_ns=future_ns, config={}, db_path=db)
    lines = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
    runs = [r for r in lines if r.get("_kind") == "tel_runs"]
    assert runs == []
