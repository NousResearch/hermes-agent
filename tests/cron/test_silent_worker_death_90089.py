"""Regression for #90089 — Windows manual cron run worker subprocess dies
silently, execution stuck 'running' forever.

Two related failure modes under test:

1. ``reclaim_stale_executions()`` — a context-agnostic version of
   ``recover_interrupted_executions`` that can be called from any context
   (not just the gateway tick loop).  When a manual CLI ``hermes cron run``
   spawns a worker that dies without reaching the finally block that calls
   ``finish_execution``, the execution row stays ``running`` forever because
   the dead-owner reaper only fires from the gateway tick every 300 s — and
   if the gateway isn't running, it never fires.

2. ``_job_action("run", ...)`` in the CLI must call
   ``reclaim_stale_executions()`` before dispatching a new run so zombie
   records from previous crashed runs are cleaned up.  The existing recovery
   call inside ``_try_dispatch_background_run`` is gated behind the
   ``async_delivery_supported()`` check, which the CLI path explicitly
   disables (stateless channel) — so the recovery never runs for one-shot
   CLI invocations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_ledger(monkeypatch, tmp_path):
    """Point the execution ledger at a temp DB."""
    import cron.executions as executions

    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions


def _dead_pid() -> int:
    """PID of a real process that has already exited."""
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def _orphan_running_row(executions, job_id: str) -> str:
    """Persist a *running* execution owned by a process that no longer exists.

    Mirrors what a dead worker leaves behind: a row stuck in ``running``
    whose owner PID is dead — the exact symptom from issue #90089.
    """
    record = executions.create_execution(job_id, source="direct")
    executions.mark_execution_running(record["id"])
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='dead-worker-process', pid=?, "
            "process_started_at=NULL WHERE id=?",
            (_dead_pid(), record["id"]),
        )
    return record["id"]


# ---------------------------------------------------------------------------
# reclaim_stale_executions — callable from any context
# ---------------------------------------------------------------------------

class TestReclaimStaleExecutions:
    """``reclaim_stale_executions`` is the context-agnostic variant of
    ``recover_interrupted_executions``: same semantics (mark provably-dead
    owner rows ``unknown``), callable from the CLI path without a gateway.
    """

    def test_reclaims_running_row_from_dead_owner(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        execution_id = _orphan_running_row(executions, "stale-running-job")

        reclaimed = executions.reclaim_stale_executions()

        assert reclaimed >= 1
        record = executions.latest_execution("stale-running-job")
        assert record["id"] == execution_id
        assert record["status"] == "unknown"
        assert record["finished_at"] is not None

    def test_reclaims_claimed_row_from_dead_owner(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        record = executions.create_execution("stale-claimed-job", source="direct")
        with executions._transaction() as conn:
            conn.execute(
                "UPDATE executions SET process_id='dead-cli-process', pid=?, "
                "process_started_at=NULL WHERE id=?",
                (_dead_pid(), record["id"]),
            )

        assert executions.reclaim_stale_executions() >= 1
        assert executions.latest_execution("stale-claimed-job")["status"] == "unknown"

    def test_does_not_touch_live_owner_row(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        record = executions.create_execution("live-job", source="builtin")
        executions.mark_execution_running(record["id"])

        assert executions.reclaim_stale_executions() == 0
        assert executions.latest_execution("live-job")["status"] == "running"

    def test_does_not_touch_terminal_rows(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        record = executions.create_execution("done-job", source="builtin")
        executions.finish_execution(record["id"], success=True)

        assert executions.reclaim_stale_executions() == 0
        assert executions.latest_execution("done-job")["status"] == "completed"

    def test_returns_count_of_reclaimed_rows(self, monkeypatch, tmp_path):
        executions = _point_ledger(monkeypatch, tmp_path)
        _orphan_running_row(executions, "stale-1")
        _orphan_running_row(executions, "stale-2")

        reclaimed = executions.reclaim_stale_executions()
        assert reclaimed == 2

    def test_real_subprocess_stale_running_is_reclaimed(self, tmp_path):
        """End-to-end: a subprocess creates a running row and exits; a fresh
        process calls ``reclaim_stale_executions`` and clears it."""
        home = tmp_path / "home"
        repo = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["HERMES_HOME"] = str(home)
        env["PYTHONPATH"] = str(repo)

        create = subprocess.run(
            [
                sys.executable, "-c",
                "from cron.executions import create_execution, mark_execution_running; "
                "r=create_execution('e2e-stale', source='direct'); "
                "mark_execution_running(r['id']); print(r['id'])",
            ],
            cwd=repo, env=env, text=True, capture_output=True, check=True,
        )
        execution_id = create.stdout.strip()

        recover = subprocess.run(
            [
                sys.executable, "-c",
                "import json; "
                "from cron.executions import reclaim_stale_executions, list_executions; "
                "print(reclaim_stale_executions()); "
                "print(json.dumps(list_executions(job_id='e2e-stale')))",
            ],
            cwd=repo, env=env, text=True, capture_output=True, check=True,
        )
        lines = recover.stdout.strip().splitlines()
        assert lines[0] == "1"
        records = json.loads(lines[1])
        assert records[0]["id"] == execution_id
        assert records[0]["status"] == "unknown"


# ---------------------------------------------------------------------------
# CLI _job_action sweep — reclaim before dispatch
# ---------------------------------------------------------------------------

class TestCliRunReclaimsBeforeDispatch:
    """The CLI ``hermes cron run`` path must call
    ``reclaim_stale_executions`` before dispatching a new run, because the
    background-dispatch path's own recovery call is gated behind
    ``async_delivery_supported()`` — which the CLI explicitly disables.
    """

    def test_run_action_calls_reclaim_before_cron_api(self, monkeypatch):
        from hermes_cli import cron as cron_cli

        calls: list[str] = []

        def _fake_reclaim():
            calls.append("reclaim")
            return 0

        monkeypatch.setattr(
            "cron.executions.reclaim_stale_executions", _fake_reclaim,
        )

        def _fake_cron_api(**kwargs):
            calls.append("cron_api")
            return {"success": True, "job": {"executed": True, "execution_success": True}}

        monkeypatch.setattr(cron_cli, "_cron_api", _fake_cron_api)

        assert cron_cli._job_action("run", "job-123", "Triggered") == 0
        assert calls[0] == "reclaim", "reclaim must run before cron_api dispatch"
        assert "cron_api" in calls

    def test_non_run_actions_do_not_call_reclaim(self, monkeypatch):
        from hermes_cli import cron as cron_cli

        calls: list[str] = []

        monkeypatch.setattr(
            "cron.executions.reclaim_stale_executions",
            lambda: calls.append("reclaim") or 0,
        )

        def _fake_cron_api(**kwargs):
            calls.append("cron_api")
            return {"success": True, "job": {"name": "j"}}

        monkeypatch.setattr(cron_cli, "_cron_api", _fake_cron_api)

        cron_cli._job_action("pause", "job-123", "Paused")
        assert "reclaim" not in calls, "pause must not trigger stale reclaim"

    def test_reclaim_failure_does_not_block_run(self, monkeypatch):
        from hermes_cli import cron as cron_cli

        def _boom():
            raise RuntimeError("ledger unavailable")

        monkeypatch.setattr("cron.executions.reclaim_stale_executions", _boom)

        def _fake_cron_api(**kwargs):
            return {"success": True, "job": {"executed": True, "execution_success": True}}

        monkeypatch.setattr(cron_cli, "_cron_api", _fake_cron_api)

        # Must not raise — best-effort self-heal.
        assert cron_cli._job_action("run", "job-456", "Triggered") == 0


# ---------------------------------------------------------------------------
# run_one_job safety net — execution still 'running' after job body
# ---------------------------------------------------------------------------


class TestRunOneJobSafetyNet:
    """Missing terminal evidence must settle as ``unknown``, not ``failed``."""

    def test_safety_net_marks_still_running_as_unknown(
        self, monkeypatch, tmp_path
    ):
        import cron.scheduler as scheduler
        import cron.executions as executions

        monkeypatch.setattr(
            executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
        )

        record = executions.create_execution("safety-net-job", source="direct")
        executions.mark_execution_running(record["id"])
        execution_id = record["id"]

        monkeypatch.setattr(
            scheduler,
            "run_job",
            lambda job, **kw: (True, "output", "response", None),
        )
        monkeypatch.setattr(scheduler, "claim_dispatch", lambda _jid: True)
        monkeypatch.setattr(
            scheduler, "save_job_output", lambda *_a, **_k: "/tmp/fake"
        )
        monkeypatch.setattr(
            scheduler, "_deliver_result", lambda *_a, **_k: None
        )
        monkeypatch.setattr(
            scheduler, "mark_job_run", lambda *_a, **_k: True
        )
        # Suppress the ordinary terminal writer. The finally-block CAS must
        # be the only path that can close this execution.
        monkeypatch.setattr(
            scheduler, "finish_execution", lambda *_a, **_k: None
        )

        job = {
            "id": "safety-net-job",
            "execution_id": execution_id,
            "prompt": "test",
        }
        scheduler.run_one_job(job)

        final = executions.latest_execution("safety-net-job")
        assert final["status"] == "unknown"
        assert "side effects ran is unknown" in final["error"]

    def test_unknown_settlement_is_terminal_and_immutable(
        self, monkeypatch, tmp_path
    ):
        executions = _point_ledger(monkeypatch, tmp_path)
        record = executions.create_execution("unknown-cas-job", source="direct")
        executions.mark_execution_running(record["id"])

        settled = executions.mark_execution_unknown(
            record["id"], error="outcome cannot be proved"
        )

        assert settled["status"] == "unknown"
        assert executions.finish_execution(record["id"], success=True) is None
        assert executions.mark_execution_unknown(record["id"]) is None
        final = executions.latest_execution("unknown-cas-job")
        assert final["status"] == "unknown"
        assert final["error"] == "outcome cannot be proved"
