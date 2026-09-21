"""Tests for the loop-diagnostics failure-path integration.

Covers the glue between the diagnosis engine and the Kanban worker failure
lifecycle (``hermes_cli/observability/loop_diagnostics_integration.py``):

* disabled config -> no event, no diagnosis file, no metrics;
* enabled + trace present -> diagnosis event + ``<run_id>.diagnosis.json``
  written + metrics row;
* missing trace -> ``unknown`` diagnosis with escalate intervention
  (fallback, never an error);
* engine error -> ``diagnosis_failed`` event, original error preserved,
  no raise;
* missing run id -> ``diagnosis_skipped`` event;
* ``diagnose_on_failure=false`` -> skipped even when ``enabled``;
* human-readable summary shape;
* metrics redaction (error presence only, never raw text).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _integ():
    """Live ``loop_diagnostics_integration`` module.

    Resolved at CALL time, not import time. A sibling test may purge every
    ``hermes_cli.*`` entry from ``sys.modules`` (see
    ``test_kanban_cli_dispatch_passthrough.isolated_kanban_home``) to force a
    fresh config load. That orphans any module-level reference bound here at
    collection time: the tests would then exercise a stale module object whose
    ``_emit_event`` is NOT the one the fixtures patch, and every assertion on
    captured events would see an empty list. Same hazard for the recorder
    module the fixtures patch — see ``_live_recorder_module``.
    """
    return importlib.import_module(
        "hermes_cli.observability.loop_diagnostics_integration"
    )


SCHEMA_VERSION = "hermes.loop_diagnostics.v1"


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _live_recorder_module():
    """Return the CURRENT ``loop_diagnostics_recorder`` module object.

    A sibling test (e.g. ``test_kanban_cli_dispatch_passthrough``) may purge
    every ``hermes_cli.*`` entry from ``sys.modules`` to get a fresh config
    load. When that happens the previously imported module object is orphaned:
    patching IT would leave the live module's ``load_recorder_config`` intact,
    so ``_integ()._load_enabled()`` (which does a function-local import and therefore
    resolves the live object) would read the real, disabled config and every
    test here would silently see ``enabled=False``.

    Re-importing inside the fixture body binds the module actually in
    ``sys.modules`` at test time, so ``monkeypatch.setattr`` lands on the
    object the code under test will use.
    """
    import importlib
    import hermes_cli.observability.loop_diagnostics_recorder as _rec

    # Re-resolve from sys.modules: after a purge, a stale reference can linger
    # in this test module's globals, so reload rather than trust it.
    return importlib.import_module("hermes_cli.observability.loop_diagnostics_recorder")


@pytest.fixture
def enabled_cfg(monkeypatch):
    """Force the integration config to enabled=True for the test."""
    recorder_mod = _live_recorder_module()

    def fake_load_config(config=None):
        return {
            "enabled": True,
            "diagnose_on_failure": True,
            "max_events_per_run": 100,
            "retain_runs": 2,
        }

    monkeypatch.setattr(recorder_mod, "load_recorder_config", fake_load_config)


@pytest.fixture
def disabled_cfg(monkeypatch):
    recorder_mod = _live_recorder_module()

    def fake_load_config(config=None):
        return {
            "enabled": False,
            "diagnose_on_failure": True,
            "max_events_per_run": 100,
            "retain_runs": 2,
        }

    monkeypatch.setattr(recorder_mod, "load_recorder_config", fake_load_config)


def _header(run_id: int, task_id: str = "t_int") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run_header",
        "task_id": task_id,
        "run_id": run_id,
        "attempt": 1,
        "profile": "default",
        "goal_mode": False,
        "ts": 1785740000 + run_id,
    }


def _start(run_id: int, seq: int, tool: str, **kw) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_start",
        "task_id": "t_int",
        "run_id": run_id,
        "action_id": f"{run_id}:{seq}",
        "parent_action_id": None,
        "loop_id": None,
        "iteration": None,
        "ts": 1785740000 + run_id + seq,
        "action_kind": "tool_call",
        "tool_name": tool,
        "summary": f"{tool} call",
    }
    rec.update(kw)
    return rec


def _end(run_id: int, seq: int, status: str, **kw) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_end",
        "task_id": "t_int",
        "run_id": run_id,
        "action_id": f"{run_id}:{seq}",
        "ts": 1785740000 + run_id + seq + 10,
        "status": status,
        "duration_ms": 100,
    }
    rec.update(kw)
    return rec


def _edge(run_id: int, from_seq: int, to_seq: int, kind: str = "data") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "edge",
        "task_id": "t_int",
        "run_id": run_id,
        "from_action_id": f"{run_id}:{from_seq}",
        "to_action_id": f"{run_id}:{to_seq}",
        "edge_kind": kind,
        "ts": 1785740000 + run_id + to_seq + 10,
    }


def _footer(run_id: int, outcome: str = "blocked", error: str = "") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run_footer",
        "task_id": "t_int",
        "run_id": run_id,
        "ts": 1785740000 + run_id + 99,
        "outcome": outcome,
        "error": error or None,
    }


def _failure_trace(run_id: int) -> list[dict]:
    """terminal (clone) failed -> read_file (data) failed (input_invalid)."""
    return [
        _header(run_id),
        _start(run_id, 1, "terminal", summary="clone repo"),
        _end(run_id, 1, "error", error_type="ExitCodeError",
             error_message="repository not found", result_hash="deadbeef1"),
        _edge(run_id, 1, 2, "data"),
        _start(run_id, 2, "read_file", summary="read design doc"),
        _end(run_id, 2, "error", error_type="FileNotFoundError",
             error_message="path does not exist", result_hash="c0ffee1"),
        _footer(run_id),
    ]


def _write_trace(tmp_path: Path, run_id: int, records: list[dict]) -> Path:
    """Write a trace into the expected storage layout (loop-traces/<task>/)."""
    task_dir = tmp_path / "loop-traces" / "t_int"
    task_dir.mkdir(parents=True, exist_ok=True)
    p = task_dir / f"{run_id}.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return p


class _FakeConn:
    """Minimal event sink capturing _append_event calls."""

    def __init__(self):
        self.events: list[tuple[str, str, dict, int | None]] = []

    def append(self, task_id, kind, payload, run_id=None):
        self.events.append((task_id, kind, payload, run_id))


def _capture(monkeypatch, conn):
    """Route the integration's event emit through the fake conn."""
    import hermes_cli.observability.loop_diagnostics_integration as integ

    def fake_emit_event(c, task_id, kind, payload, run_id=None):
        if c is not None:
            c.append(task_id, kind, payload, run_id)

    monkeypatch.setattr(integ, "_emit_event", fake_emit_event)


# ---------------------------------------------------------------------------
# Config gating
# ---------------------------------------------------------------------------


def test_disabled_config_no_side_effects(disabled_cfg, tmp_path, monkeypatch):
    """enabled=False -> no event, no diagnosis file, no metrics row."""
    conn = _FakeConn()
    _capture(monkeypatch, conn)
    trace = _write_trace(tmp_path, 7, _failure_trace(7))

    result = _integ().attach_failure_diagnosis(
        conn, "t_int", run_id=7, outcome="crashed", error="boom",
        trace_path=trace, board="default",
    )
    assert result is None
    assert conn.events == []
    assert not (tmp_path / "loop-traces" / "t_int" / "7.diagnosis.json").exists()
    assert not _integ()._metrics_path().exists()


def test_diagnose_on_failure_false_skips(enabled_cfg, tmp_path, monkeypatch):
    """diagnose_on_failure=False -> skipped even when enabled=True."""
    recorder_mod = _live_recorder_module()

    def fake_load_config(config=None):
        return {
            "enabled": True,
            "diagnose_on_failure": False,
            "max_events_per_run": 100,
            "retain_runs": 2,
        }

    monkeypatch.setattr(recorder_mod, "load_recorder_config", fake_load_config)
    conn = _FakeConn()
    _capture(monkeypatch, conn)
    trace = _write_trace(tmp_path, 8, _failure_trace(8))

    result = _integ().attach_failure_diagnosis(
        conn, "t_int", run_id=8, outcome="crashed", error="boom",
        trace_path=trace, board="default",
    )
    assert result is None
    assert conn.events == []


# ---------------------------------------------------------------------------
# Happy path: enabled + trace
# ---------------------------------------------------------------------------


def test_enabled_writes_diagnosis_file_and_event(enabled_cfg, tmp_path, monkeypatch):
    """enabled=True + failure trace -> diagnosis event + file + metrics."""
    conn = _FakeConn()
    _capture(monkeypatch, conn)
    trace = _write_trace(tmp_path, 9, _failure_trace(9))

    result = _integ().attach_failure_diagnosis(
        conn, "t_int", run_id=9, outcome="blocked", error="read_file failed",
        trace_path=trace, board="default",
    )
    assert result is not None
    assert result["status"] == "root_cause_found"
    assert result["category"] == "input_invalid"
    assert result["root_cause_action_ids"] == ["9:1"]
    assert result["propagation_path"] == ["9:2", "9:1"]

    # Diagnosis file written next to trace.
    diag_path = tmp_path / "loop-traces" / "t_int" / "9.diagnosis.json"
    assert diag_path.exists()
    saved = json.loads(diag_path.read_text(encoding="utf-8"))
    assert saved["status"] == "root_cause_found"
    assert saved["category"] == "input_invalid"

    # The persisted diagnosis conforms to the contract surface (the repo
    # has no jsonschema dep; mirror the schema test's stdlib-only checks).
    assert saved["schema_version"] == "hermes.loop_diagnostics.v1"
    assert saved["kind"] == "diagnosis_result"
    assert saved["diagnosis_version"] == "hermes.loop_diagnostics.diagnosis.v1"
    assert saved["task_id"] == "t_int"
    assert saved["run_id"] == 9
    assert saved["status"] in {
        "root_cause_found", "candidates", "unknown", "malformed_trace",
    }
    assert saved["category"] in {
        "action_error", "input_invalid", "loop_repeated", "retry_exhausted",
        "timeout_propagation", "cancellation_propagation", "malformed_trace",
        "unknown",
    }
    assert isinstance(saved["evidence"], dict)
    assert isinstance(saved["interventions"], list)

    # Event emitted with machine + human readable payload.
    kinds = [e[1] for e in conn.events]
    assert "diagnosis" in kinds
    ev = [e for e in conn.events if e[1] == "diagnosis"][0]
    payload = ev[2]
    assert payload["status"] == "root_cause_found"
    assert payload["category"] == "input_invalid"
    assert payload["root_cause_action_ids"] == ["9:1"]
    assert payload["propagation_path"] == ["9:2", "9:1"]
    assert payload["outcome"] == "blocked"
    assert payload["original_error"] == "read_file failed"
    assert payload["summary"]  # human-readable line
    assert "diagnosis=root_cause_found" in payload["summary"]
    assert "category=input_invalid" in payload["summary"]
    assert "intervention=" in payload["summary"]

    # Metrics row.
    assert _integ()._metrics_path().exists()
    lines = _integ()._metrics_path().read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    metric = json.loads(lines[0])
    assert metric["task_id"] == "t_int"
    assert metric["run_id"] == 9
    assert metric["status"] == "root_cause_found"
    assert metric["category"] == "input_invalid"
    assert metric["latency_ms"] >= 0
    assert metric["error"] is True


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def test_missing_trace_unknown_escalate(enabled_cfg, tmp_path, monkeypatch):
    """No trace file -> status=unknown, intervention=escalate, never raises."""
    conn = _FakeConn()
    _capture(monkeypatch, conn)

    result = _integ().attach_failure_diagnosis(
        conn, "t_int", run_id=99, outcome="crashed", error="boom",
        board="default",
    )
    assert result is not None
    assert result["status"] == "unknown"
    assert result["category"] == "unknown"
    assert result["interventions"] == [{
        "kind": "escalate",
        "action_id": "",
        "rationale": "Diagnosis inconclusive; escalate for manual review.",
        "payload": {},
    }]

    kinds = [e[1] for e in conn.events]
    assert "diagnosis" in kinds
    ev = [e for e in conn.events if e[1] == "diagnosis"][0]
    assert ev[2]["status"] == "unknown"
    assert ev[2]["summary"].startswith("diagnosis=unknown category=unknown")


def test_engine_error_emits_diagnosis_failed(enabled_cfg, tmp_path, monkeypatch):
    """Engine exception -> diagnosis_failed event, original error preserved."""
    conn = _FakeConn()
    _capture(monkeypatch, conn)

    import hermes_cli.observability.loop_diagnostics_engine as engine_mod

    def boom(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(engine_mod, "diagnose", boom)

    trace = _write_trace(tmp_path, 10, _failure_trace(10))
    result = _integ().attach_failure_diagnosis(
        conn, "t_int", run_id=10, outcome="crashed", error="original worker error",
        trace_path=trace, board="default",
    )
    assert result is None
    kinds = [e[1] for e in conn.events]
    assert "diagnosis_failed" in kinds
    ev = [e for e in conn.events if e[1] == "diagnosis_failed"][0]
    assert "engine exploded" in ev[2]["reason"]
    assert ev[2]["original_error"] == "original worker error"

    # Metrics recorded the failure.
    assert _integ()._metrics_path().exists()
    metric = json.loads(
        _integ()._metrics_path().read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert metric["status"] == "diagnosis_error"


def test_missing_run_id_emits_skipped(enabled_cfg, tmp_path, monkeypatch):
    """run_id=None -> diagnosis_skipped event, no file."""
    conn = _FakeConn()
    _capture(monkeypatch, conn)

    result = _integ().attach_failure_diagnosis(
        conn, "t_int", run_id=None, outcome="crashed", error="boom",
        board="default",
    )
    assert result is None
    kinds = [e[1] for e in conn.events]
    assert "diagnosis_skipped" in kinds
    ev = [e for e in conn.events if e[1] == "diagnosis_skipped"][0]
    assert "missing run identity" in ev[2]["reason"]
    assert ev[2]["original_error"] == "boom"


# ---------------------------------------------------------------------------
# Human-readable + config loader
# ---------------------------------------------------------------------------


def test_human_readable_shape():
    r = {
        "status": "root_cause_found",
        "category": "input_invalid",
        "root_cause_action_ids": ["42:1"],
        "propagation_path": ["42:2", "42:1"],
        "interventions": [{"kind": "retry_from_checkpoint", "action_id": "42:1"}],
        "explanation": "Action 42:2 failed because its data predecessor 42:1 failed",
    }
    s = _integ()._human_readable(r)
    assert "diagnosis=root_cause_found" in s
    assert "category=input_invalid" in s
    assert "root=42:1" in s
    assert "path=42:2,42:1" in s
    assert "intervention=retry_from_checkpoint" in s


def test_human_readable_unknown():
    r = {"status": "unknown", "category": "unknown", "interventions": []}
    s = _integ()._human_readable(r)
    assert s == "diagnosis=unknown category=unknown"


def test_config_loader_shape(enabled_cfg):
    """load_recorder_config returns diagnose_on_failure when stubbed."""
    cfg = _integ()._load_enabled()
    assert cfg.get("enabled") is True
    assert cfg.get("diagnose_on_failure") is True
