"""Trace/span layer: tel_spans is populated as a connected run -> calls tree.

Drives the real dispatch chain (discover_plugins -> invoke_hook) and asserts the
timing/lineage backbone in tel_spans:
  - one root span per run (kind="run", parent_span_id NULL),
  - one child span per model/tool call parented to the root,
  - a single trace_id across the run,
  - call detail rows (tel_model_calls / tel_tool_calls) JOIN to their span by span_id,
  - reconstructed durations match the reported latency/duration.

This is the regression guard for the waterfall a desktop trace viewer renders.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

import hermes_state


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"
    hermes_state.SessionDB(db_path=db)
    import hermes_cli.plugins as plugins_mod
    monkeypatch.setattr(plugins_mod, "_plugin_manager", None, raising=False)
    from agent.telemetry import emitter as emitter_mod
    emitter_mod.reset_emitter_for_tests(None)
    import plugins.telemetry as plug
    plug._runs.clear()
    yield db, plugins_mod, emitter_mod
    try:
        emitter_mod.get_emitter().flush()
    except Exception:
        pass
    emitter_mod.reset_emitter_for_tests(None)
    monkeypatch.setattr(plugins_mod, "_plugin_manager", None, raising=False)


def _one_turn(invoke_hook):
    invoke_hook("on_session_start", session_id="s1",
                model="anthropic/claude-opus-4", platform="cli")
    invoke_hook("post_api_request", session_id="s1", platform="cli",
                provider="anthropic", model="claude-opus-4", api_duration=0.9,
                usage={"input_tokens": 1000, "output_tokens": 120})
    invoke_hook("post_tool_call", session_id="s1", platform="cli",
                function_name="web_search", duration_ms=210, result='{"data": "ok"}')
    invoke_hook("on_session_finalize", session_id="s1", platform="cli",
                turn_exit_reason="completed", estimated_cost_usd=0.01, cost_status="known")


def test_tel_spans_forms_connected_trace(runtime):
    db, plugins_mod, emitter_mod = runtime
    plugins_mod.discover_plugins(force=True)
    _one_turn(plugins_mod.invoke_hook)
    time.sleep(0.5)
    emitter_mod.get_emitter().flush()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    spans = conn.execute(
        "SELECT span_id, parent_span_id, kind, name, start_ns, end_ns, status, trace_id "
        "FROM tel_spans"
    ).fetchall()

    # root + model + tool
    assert len(spans) == 3
    roots = [s for s in spans if s["parent_span_id"] is None]
    children = [s for s in spans if s["parent_span_id"] is not None]
    assert len(roots) == 1
    assert roots[0]["kind"] == "run"
    assert len(children) == 2

    # single trace, all children parented to the root
    assert len({s["trace_id"] for s in spans}) == 1
    assert all(c["parent_span_id"] == roots[0]["span_id"] for c in children)

    # spans are time-ordered and carry real durations
    by_kind = {s["kind"]: s for s in spans}
    assert (by_kind["model"]["end_ns"] - by_kind["model"]["start_ns"]) == 900 * 1_000_000
    assert (by_kind["tool"]["end_ns"] - by_kind["tool"]["start_ns"]) == 210 * 1_000_000
    assert by_kind["run"]["end_ns"] >= by_kind["run"]["start_ns"]


def test_detail_rows_join_to_spans(runtime):
    db, plugins_mod, emitter_mod = runtime
    plugins_mod.discover_plugins(force=True)
    _one_turn(plugins_mod.invoke_hook)
    time.sleep(0.5)
    emitter_mod.get_emitter().flush()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    mc = conn.execute(
        "SELECT m.model, s.kind, s.trace_id FROM tel_model_calls m "
        "JOIN tel_spans s ON m.span_id = s.span_id"
    ).fetchone()
    assert mc is not None and mc["model"] == "claude-opus-4" and mc["kind"] == "model"

    tc = conn.execute(
        "SELECT t.tool_name, s.kind FROM tel_tool_calls t "
        "JOIN tel_spans s ON t.span_id = s.span_id"
    ).fetchone()
    assert tc is not None and tc["tool_name"] == "web_search" and tc["kind"] == "tool"
    conn.close()
