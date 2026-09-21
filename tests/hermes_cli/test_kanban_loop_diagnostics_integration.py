"""End-to-end tests: loop-diagnostics attached through the Kanban failure lifecycle.

These verify the integration contract from the task:

> Completion requires a failed worker run to expose the report through the
> system's existing task-status or event interface while successful runs
> remain unaffected.

We drive the real dispatcher-side failure paths in ``kanban_db``
(``detect_crashed_workers`` / timeout reclaim / ``_record_spawn_failure`` /
``block_task``) with the loop-diagnostics config enabled and a trace on disk,
then assert:

* a ``diagnosis`` event is appended to the task's event stream with the
  machine + human readable report;
* the ``<run_id>.diagnosis.json`` file is written next to the trace;
* the original failure event (``crashed`` / ``timed_out`` / ``spawn_failed``
  / ``blocked``) is still emitted and the task status is unchanged — the
  integration never masks the original error;
* successful completion emits NO ``diagnosis`` event (success paths
  unaffected).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli.kanban_db_dispatch import _record_spawn_failure
from hermes_cli.observability.loop_diagnostics_recorder import (
    loop_traces_dir,
)

SCHEMA_VERSION = "hermes.loop_diagnostics.v1"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def loop_diag_enabled(monkeypatch):
    """Force loop_diagnostics.enabled=True + diagnose_on_failure=True."""
    import hermes_cli.observability.loop_diagnostics_recorder as recorder_mod

    def fake_load_config(config=None):
        return {
            "enabled": True,
            "diagnose_on_failure": True,
            "max_events_per_run": 100,
            "retain_runs": 2,
        }

    monkeypatch.setattr(recorder_mod, "load_recorder_config", fake_load_config)


def _header(run_id: int, task_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run_header",
        "task_id": task_id,
        "run_id": run_id,
        "attempt": 1,
        "profile": "default",
        "goal_mode": False,
        "ts": 1785741000 + run_id,
    }


def _start(run_id: int, seq: int, task_id: str, tool: str, **kw) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_start",
        "task_id": task_id,
        "run_id": run_id,
        "action_id": f"{run_id}:{seq}",
        "parent_action_id": None,
        "loop_id": None,
        "iteration": None,
        "ts": 1785741000 + run_id + seq,
        "action_kind": "tool_call",
        "tool_name": tool,
        "summary": f"{tool} call",
    }
    rec.update(kw)
    return rec


def _end(run_id: int, seq: int, task_id: str, status: str, **kw) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_end",
        "task_id": task_id,
        "run_id": run_id,
        "action_id": f"{run_id}:{seq}",
        "ts": 1785741000 + run_id + seq + 10,
        "status": status,
        "duration_ms": 100,
    }
    rec.update(kw)
    return rec


def _edge(run_id: int, from_seq: int, to_seq: int, task_id: str, kind: str = "data") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "edge",
        "task_id": task_id,
        "run_id": run_id,
        "from_action_id": f"{run_id}:{from_seq}",
        "to_action_id": f"{run_id}:{to_seq}",
        "edge_kind": kind,
        "ts": 1785741000 + run_id + to_seq + 10,
    }


def _footer(run_id: int, task_id: str, outcome: str = "crashed", error: str = "") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run_footer",
        "task_id": task_id,
        "run_id": run_id,
        "ts": 1785741000 + run_id + 99,
        "outcome": outcome,
        "error": error or None,
    }


def _failure_trace(run_id: int, task_id: str) -> list[dict]:
    return [
        _header(run_id, task_id),
        _start(run_id, 1, task_id, "terminal", summary="clone repo"),
        _end(run_id, 1, task_id, "error", error_type="ExitCodeError",
             error_message="repository not found", result_hash="deadbeef1"),
        _edge(run_id, 1, 2, task_id, "data"),
        _start(run_id, 2, task_id, "read_file", summary="read design doc"),
        _end(run_id, 2, task_id, "error", error_type="FileNotFoundError",
             error_message="path does not exist", result_hash="c0ffee1"),
        _footer(run_id, task_id),
    ]


def _write_trace_for_task(task_id: str, run_id: int, records: list[dict]) -> Path:
    """Write a trace into the real loop-traces layout for a task."""
    task_dir = loop_traces_dir("default") / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    p = task_dir / f"{run_id}.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return p


def _events(conn, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT kind, payload, run_id FROM task_events "
        "WHERE task_id=? ORDER BY id", (task_id,),
    ).fetchall()
    out = []
    for r in rows:
        payload = json.loads(r["payload"]) if r["payload"] else None
        out.append({"kind": r["kind"], "payload": payload, "run_id": r["run_id"]})
    return out


# ---------------------------------------------------------------------------
# Crash path
# ---------------------------------------------------------------------------


def test_crashed_worker_exposes_diagnosis_event(
    kanban_home, loop_diag_enabled, monkeypatch,
):
    """A crashed worker run emits both ``crashed`` and ``diagnosis`` events."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="crash-diag", assignee="a")

        # Claim to open a real run.
        kb.claim_task(conn, tid, claimer=f"{host}:w1")
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        run_id = int(row["current_run_id"])

        # Point the claim at a dead pid so the crash path acts on it.
        conn.execute(
            "UPDATE tasks SET worker_pid=?, status='running' WHERE id=?",
            (70001, tid),
        )
        conn.commit()

        # Trace on disk for this run.
        _write_trace_for_task(tid, run_id, _failure_trace(run_id, tid))

        crashed = kb.detect_crashed_workers(conn)
        assert tid in crashed

        evs = _events(conn, tid)
        kinds = [e["kind"] for e in evs]
        assert "crashed" in kinds
        assert "diagnosis" in kinds

        diag = [e for e in evs if e["kind"] == "diagnosis"][0]
        assert diag["run_id"] == run_id
        p = diag["payload"]
        assert p["status"] == "root_cause_found"
        assert p["category"] == "input_invalid"
        assert p["root_cause_action_ids"] == [f"{run_id}:1"]
        assert p["original_error"]  # original crash text preserved
        assert "diagnosis=root_cause_found" in p["summary"]

        # Diagnosis file written next to the trace.
        diag_path = loop_traces_dir("default") / tid / f"{run_id}.diagnosis.json"
        assert diag_path.exists()

        # Task back to ready (crash semantics unchanged).
        assert kb.get_task(conn, tid).status == "ready"


def test_crashed_worker_without_trace_still_emits_unknown(
    kanban_home, loop_diag_enabled, monkeypatch,
):
    """Crash without a trace -> diagnosis event with status=unknown (fallback)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="crash-nodiag", assignee="a")
        kb.claim_task(conn, tid, claimer=f"{host}:w1")
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        run_id = int(row["current_run_id"])
        conn.execute(
            "UPDATE tasks SET worker_pid=?, status='running' WHERE id=?",
            (70002, tid),
        )
        conn.commit()

        # NO trace on disk.
        crashed = kb.detect_crashed_workers(conn)
        assert tid in crashed

        evs = _events(conn, tid)
        kinds = [e["kind"] for e in evs]
        assert "crashed" in kinds
        assert "diagnosis" in kinds
        diag = [e for e in evs if e["kind"] == "diagnosis"][0]
        assert diag["payload"]["status"] == "unknown"
        assert diag["payload"]["category"] == "unknown"


# ---------------------------------------------------------------------------
# Timeout path
# ---------------------------------------------------------------------------


def test_timed_out_worker_exposes_diagnosis_event(
    kanban_home, loop_diag_enabled, monkeypatch,
):
    """A timed-out worker run emits ``timed_out`` + ``diagnosis`` events."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="timeout-diag", assignee="a")
        kb.claim_task(conn, tid, claimer=f"{host}:wt")
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        run_id = int(row["current_run_id"])
        # Give the task a tiny max_runtime and make the run look expired.
        conn.execute(
            "UPDATE tasks SET max_runtime_seconds=? WHERE id=?",
            (1, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at=? WHERE id=?",
            (int(__import__("time").time()) - 99999, run_id),
        )
        conn.execute(
            "UPDATE tasks SET worker_pid=?, status='running' WHERE id=?",
            (70003, tid),
        )
        conn.commit()

        _write_trace_for_task(tid, run_id, _failure_trace(run_id, tid))

        timed_out = kb.enforce_max_runtime(
            conn, signal_fn=lambda _pid, _sig: None,
        )
        assert tid in timed_out

        evs = _events(conn, tid)
        kinds = [e["kind"] for e in evs]
        assert "timed_out" in kinds
        assert "diagnosis" in kinds
        diag = [e for e in evs if e["kind"] == "diagnosis"][0]
        assert diag["run_id"] == run_id
        assert diag["payload"]["status"] == "root_cause_found"
        assert diag["payload"]["outcome"] == "timed_out"


# ---------------------------------------------------------------------------
# Spawn-failure path
# ---------------------------------------------------------------------------


def test_spawn_failure_exposes_diagnosis_event(
    kanban_home, loop_diag_enabled,
):
    """A spawn failure closes the run + emits ``spawn_failed`` + ``diagnosis``."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="spawn-diag", assignee="a")
        # Claim to open a real run (dispatch_once claims before spawning).
        kb.claim_task(conn, tid, claimer="host:ws")
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        run_id = int(row["current_run_id"])

        _write_trace_for_task(tid, run_id, _failure_trace(run_id, tid))

        blocked = _record_spawn_failure(conn, tid, "spawn boom", failure_limit=99)
        assert blocked is False

        evs = _events(conn, tid)
        kinds = [e["kind"] for e in evs]
        assert "spawn_failed" in kinds
        assert "diagnosis" in kinds
        diag = [e for e in evs if e["kind"] == "diagnosis"][0]
        assert diag["run_id"] == run_id
        assert diag["payload"]["original_error"] == "spawn boom"
        assert diag["payload"]["outcome"] == "spawn_failed"


# ---------------------------------------------------------------------------
# Block path
# ---------------------------------------------------------------------------


def test_block_task_exposes_diagnosis_event(
    kanban_home, loop_diag_enabled,
):
    """A worker block with an open run emits ``blocked`` + ``diagnosis``."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="block-diag", assignee="a")
        kb.claim_task(conn, tid, claimer="host:wb")
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        run_id = int(row["current_run_id"])

        _write_trace_for_task(tid, run_id, _failure_trace(run_id, tid))

        blocked = kb.block_task(conn, tid, reason="needs human input")
        assert blocked is True

        evs = _events(conn, tid)
        kinds = [e["kind"] for e in evs]
        assert "blocked" in kinds
        assert "diagnosis" in kinds
        diag = [e for e in evs if e["kind"] == "diagnosis"][0]
        assert diag["run_id"] == run_id
        assert diag["payload"]["original_error"] == "needs human input"
        assert diag["payload"]["outcome"] == "blocked"

        assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Success path unaffected
# ---------------------------------------------------------------------------


def test_successful_completion_no_diagnosis_event(
    kanban_home, loop_diag_enabled,
):
    """A successful run emits NO diagnosis event (success unaffected)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ok-diag", assignee="a")
        kb.claim_task(conn, tid, claimer="host:wok")
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        run_id = int(row["current_run_id"])

        # Trace on disk but NO failure (all ok).
        ok_trace = [
            _header(run_id, tid),
            _start(run_id, 1, tid, "terminal", summary="clone repo"),
            _end(run_id, 1, tid, "ok", summary="repo cloned"),
            _footer(run_id, tid, outcome="completed", error=""),
        ]
        _write_trace_for_task(tid, run_id, ok_trace)

        done = kb.complete_task(
            conn, tid, result="all good", summary="finished",
        )
        assert done is True

        evs = _events(conn, tid)
        kinds = [e["kind"] for e in evs]
        assert "completed" in kinds
        assert "diagnosis" not in kinds
        assert kb.get_task(conn, tid).status == "done"


# ---------------------------------------------------------------------------
# Spawn-failure breaker TRIP (regression)
# ---------------------------------------------------------------------------


def test_spawn_failure_breaker_trip_blocks_and_persists_counter(
    kanban_home, loop_diag_enabled,
):
    """Tripping the spawn-failure breaker blocks the task and persists the counter.

    Regression: every other test in this module passes ``failure_limit=99``, so
    the breaker never trips and the trip block is never executed. That block's
    spawn path bound a 3-tuple to a 4-placeholder UPDATE, raising
    ``sqlite3.ProgrammingError`` the moment a real limit was reached — so a
    genuinely failing card was never blocked and its counter never persisted.

    ``failure_limit=1`` trips on the first failure, which is the path
    ``dispatch_once`` uses when a worker cannot be spawned.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="spawn-trip", assignee="a")
        kb.claim_task(conn, tid, claimer="host:ws")
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        run_id = int(row["current_run_id"])

        _write_trace_for_task(tid, run_id, _failure_trace(run_id, tid))

        # failure_limit=1 -> the breaker trips on this very call.
        blocked = _record_spawn_failure(conn, tid, "spawn boom", failure_limit=1)
        assert blocked is True

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1
        assert task.last_failure_error == "spawn boom"
        # The spawn path must also release the claim it cleared.
        assert task.claim_lock is None

        kinds = [e["kind"] for e in _events(conn, tid)]
        assert "gave_up" in kinds

        gave_up = [e for e in _events(conn, tid) if e["kind"] == "gave_up"][0]
        assert gave_up["payload"]["failures"] == 1
        assert gave_up["payload"]["effective_limit"] == 1
