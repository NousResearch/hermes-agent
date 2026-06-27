"""Telemetry plugin hook tests — feed realistic kwargs, assert tel_* rows + no leak."""

from __future__ import annotations

import sqlite3

import pytest

import hermes_state
from agent.telemetry import emitter as emitter_mod
from agent.telemetry.emitter import TelemetryEmitter


@pytest.fixture
def wired(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(hermes_state.SCHEMA_SQL)
    conn.close()
    em = TelemetryEmitter(events_path=tmp_path / "telemetry" / "events.jsonl", db_path=db)
    emitter_mod.reset_emitter_for_tests(em)
    # reset the plugin's per-run accumulators between tests
    import plugins.telemetry as plug
    plug._runs.clear()
    yield db, em, plug
    em.flush()
    em.close()
    emitter_mod.reset_emitter_for_tests(None)


def test_full_session_lifecycle_produces_rows(wired):
    db, em, plug = wired

    plug._on_session_start(session_id="sess1", platform="telegram")
    plug._on_post_api_request(
        session_id="sess1", platform="telegram",
        provider="anthropic", base_url=None, model="claude-opus-4",
        api_duration=2.5,
        usage={"input_tokens": 5000, "output_tokens": 800, "cache_read_tokens": 1000,
               "cache_write_tokens": 0, "reasoning_tokens": 0},
    )
    plug._on_post_tool_call(
        session_id="sess1", platform="telegram",
        function_name="web_search", duration_ms=812, result="{\"data\": \"...\"}",
    )
    # Production finalize callers pass `reason` (e.g. "shutdown"), not cost.
    plug._on_session_finalize(
        session_id="sess1", platform="telegram", reason="shutdown",
    )
    em.flush()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM tel_runs").fetchone()
    assert run is not None
    assert run["entrypoint"] == "gateway"
    assert run["platform"] == "telegram"
    assert run["end_reason"] == "completed"
    assert run["model_call_count"] == 1
    assert run["tool_call_count"] == 1

    mc = conn.execute("SELECT * FROM tel_model_calls").fetchone()
    assert mc["provider"] == "anthropic"
    assert mc["model"] == "claude-opus-4"
    assert mc["input_tokens"] == 5000

    tc = conn.execute("SELECT * FROM tel_tool_calls").fetchone()
    assert tc["tool_name"] == "web_search"
    assert tc["result_class"] == "ok"
    conn.close()


def test_tool_error_result_classified_and_counted(wired):
    db, em, plug = wired
    plug._on_session_start(session_id="s2", platform="cli")
    plug._on_post_tool_call(
        session_id="s2", function_name="terminal", duration_ms=10,
        result="{\"error\": \"command failed\"}",
    )
    plug._on_session_finalize(session_id="s2", reason="shutdown")
    em.flush()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    tc = conn.execute("SELECT result_class, tool_name FROM tel_tool_calls").fetchone()
    assert tc["result_class"] == "error"
    assert tc["tool_name"] == "terminal"
    run = conn.execute("SELECT error_count FROM tel_runs").fetchone()
    assert run["error_count"] == 1
    conn.close()


def test_api_error_recorded(wired):
    db, em, plug = wired
    plug._on_session_start(session_id="s3", platform="cli")
    plug._on_api_request_error(session_id="s3", error_type="provider timeout after 60s")
    plug._on_session_finalize(session_id="s3", failed=True)
    em.flush()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    err = conn.execute("SELECT error_class, subsystem FROM tel_error_events").fetchone()
    assert err["error_class"] == "provider_timeout"
    assert err["subsystem"] == "model_api"
    run = conn.execute("SELECT end_reason FROM tel_runs").fetchone()
    assert run["end_reason"] == "failed"
    conn.close()


def test_hooks_never_raise_on_garbage_kwargs(wired):
    _, em, plug = wired
    # Missing everything — must be swallowed by the _safe wrapper.
    plug._on_post_api_request()
    plug._on_post_tool_call()
    plug._on_session_finalize()
    plug._on_api_request_error()
    em.flush()


def test_no_message_content_in_tool_rows(wired):
    """The tool hook receives a result blob; only the classification persists, not content."""
    db, em, plug = wired
    plug._on_session_start(session_id="s4", platform="cli")
    secret = "{\"data\": \"USER SECRET sk-ABCDEF and /Users/alice/file.txt\"}"
    plug._on_post_tool_call(session_id="s4", function_name="web_search",
                            duration_ms=5, result=secret)
    plug._on_session_finalize(session_id="s4")
    em.flush()
    # The whole tel_tool_calls row must contain none of the result content.
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT * FROM tel_tool_calls").fetchone()
    conn.close()
    blob = " ".join(str(x) for x in row)
    assert "sk-ABCDEF" not in blob
    assert "/Users/alice" not in blob
    assert "SECRET" not in blob
