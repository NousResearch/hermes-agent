"""`hermes telemetry` handler smoke tests (local-only; no upload)."""

from __future__ import annotations

import sqlite3
import time
import types

import pytest

import hermes_state


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # state.db with tel_* + seeded data
    db = tmp_path / "state.db"
    hermes_state.SessionDB(db_path=db)
    from agent.telemetry.emitter import TelemetryEmitter
    from agent.telemetry.events import RunEvent, ModelCallEvent, ToolCallEvent
    em = TelemetryEmitter(events_path=tmp_path / "telemetry" / "e.jsonl", db_path=db)
    now = time.time_ns()
    em.emit(RunEvent(run_id="r1", trace_id="t1", entrypoint="cli", end_reason="completed",
                     start_ns=now - 60_000_000, end_ns=now, model_call_count=1,
                     tool_call_count=1, estimated_cost_usd=0.3))
    em.emit(ModelCallEvent(span_id="m1", run_id="r1", provider="anthropic",
                           model="claude-opus-4",
                           input_tokens=20000, output_tokens=2000))
    em.emit(ToolCallEvent(span_id="w1", run_id="r1", tool_name="web_search",
                          result_class="ok"))
    em.flush()
    em.close()
    yield tmp_path


def _run(action, **kw):
    from hermes_cli.main import cmd_telemetry
    args = types.SimpleNamespace(telemetry_action=action, days=30, limit=10, json=False)
    for k, v in kw.items():
        setattr(args, k, v)
    cmd_telemetry(args)


def test_status_runs(home, capsys):
    _run("status")
    out = capsys.readouterr().out
    assert "Telemetry status" in out
    assert "Upload:" in out and "DISABLED" in out
    assert "Local data:" in out


def test_preview_shows_real_values(home, capsys):
    _run("preview")
    out = capsys.readouterr().out
    assert "NOT uploaded" in out
    assert "workflow_completed" in out
    # real model + tool names ARE shown (this is the user's own local data)
    assert "claude-opus-4" in out
    assert "web_search" in out


def test_status_reflects_consent_set_via_config(home, capsys):
    # Opting in is a plain config write now (no `enable` verb). status should
    # reflect consent_state=aggregate as the aggregate plane being on.
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("telemetry", {})["consent_state"] = "aggregate"
    save_config(cfg)
    _run("status")
    out = capsys.readouterr().out
    assert "consent_state=aggregate" in out
    assert "Aggregate plane: on" in out


def test_status_shows_optin_hint_when_unknown(home, capsys):
    _run("status")
    out = capsys.readouterr().out
    assert "Aggregate plane: off" in out
    assert "config set telemetry.consent_state aggregate" in out


def test_allow_aggregate_false_keeps_plane_off_in_status(home, capsys):
    # Even with consent opted in, a managed allow_aggregate:false wins.
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    tel = cfg.setdefault("telemetry", {})
    tel["consent_state"] = "aggregate"
    tel["allow_aggregate"] = False
    save_config(cfg)
    _run("status")
    out = capsys.readouterr().out
    assert "Aggregate plane: off" in out
    assert "allow_aggregate is false" in out
