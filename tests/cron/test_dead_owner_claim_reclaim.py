"""Dead-owner cron claim reclaim + one-shot CLI `cron run` sync gate (#86721).

A one-shot ``hermes cron run <job_id>`` used to background-dispatch the run
onto a daemon thread of the calling process when the CLI inherited a
gateway/desktop session env. The process exited immediately, the runner died
mid-LLM-call, and the job's execution row stayed ``claimed`` forever —
blocking every future run.

Two-part fix under test here:

1. ``hermes_cli.cron._job_action("run", ...)`` declares the channel stateless
   before invoking the cron API, so the background-dispatch path is gated off
   and the run executes synchronously to completion in the CLI process.
2. ``cron.scheduler.tick`` periodically reaps execution rows whose owner
   process is provably dead (``recover_interrupted_executions``), so a stale
   ``claimed`` row from a crashed/exited owner auto-clears without a gateway
   restart.
"""

from __future__ import annotations

import subprocess
import sys
import time
from unittest.mock import patch

import pytest

import cron.scheduler as scheduler_mod


@pytest.fixture()
def executions(monkeypatch, tmp_path):
    import cron.executions as executions_mod

    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions_mod


@pytest.fixture(autouse=True)
def _fresh_reap_window(monkeypatch):
    """Each test starts with the reap throttle open."""
    monkeypatch.setattr(scheduler_mod, "_last_dead_owner_reap_at", None)


def _dead_pid() -> int:
    """PID of a real process that has already exited."""
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def _orphan_claimed_row(executions, job_id: str) -> str:
    """Persist a claimed execution owned by a process that no longer exists.

    Mirrors what a one-shot ``hermes cron run`` leaves behind: a row stuck in
    ``claimed`` whose owner pid is dead.
    """
    record = executions.create_execution(job_id, source="direct")
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='dead-cli-process', pid=?, "
            "process_started_at=NULL WHERE id=?",
            (_dead_pid(), record["id"]),
        )
    return record["id"]


def _run_tick():
    with (
        patch.object(scheduler_mod, "get_due_jobs", return_value=[]),
        patch("tools.mcp_tool._kill_orphaned_mcp_children", lambda: None),
    ):
        return scheduler_mod.tick(verbose=False)


class TestTickReapsDeadOwnerClaims:
    def test_stale_claimed_row_from_dead_owner_is_cleared_by_tick(self, executions):
        """The exact #86721 wedge: dead-owner 'claimed' row unblocks on tick."""
        execution_id = _orphan_claimed_row(executions, "orphaned-job")

        assert _run_tick() == 0

        record = executions.latest_execution("orphaned-job")
        assert record["id"] == execution_id
        assert record["status"] == "unknown"
        assert record["finished_at"]

    def test_running_row_from_dead_owner_is_also_reclaimed(self, executions):
        record = executions.create_execution("orphaned-running", source="direct")
        executions.mark_execution_running(record["id"])
        with executions._transaction() as conn:
            conn.execute(
                "UPDATE executions SET process_id='dead-cli-process', pid=?, "
                "process_started_at=NULL WHERE id=?",
                (_dead_pid(), record["id"]),
            )

        _run_tick()

        assert executions.latest_execution("orphaned-running")["status"] == "unknown"

    def test_live_owner_claim_is_never_rewritten(self, executions):
        """A claim owned by a live process (this one) must survive the reap."""
        record = executions.create_execution("live-job", source="builtin")
        executions.mark_execution_running(record["id"])

        _run_tick()

        assert executions.latest_execution("live-job")["status"] == "running"

    def test_reap_is_throttled_between_ticks(self, monkeypatch, executions):
        calls = []
        monkeypatch.setattr(
            "cron.executions.recover_interrupted_executions",
            lambda: calls.append(1) or 0,
        )

        _run_tick()
        _run_tick()
        assert len(calls) == 1, "back-to-back ticks must not reap twice"

        monkeypatch.setattr(
            scheduler_mod,
            "_last_dead_owner_reap_at",
            time.monotonic() - scheduler_mod._DEAD_OWNER_REAP_INTERVAL_SECONDS - 1,
        )
        _run_tick()
        assert len(calls) == 2, "an expired throttle window must reap again"

    def test_reap_failure_does_not_break_the_tick(self, monkeypatch):
        def _boom():
            raise RuntimeError("ledger unavailable")

        monkeypatch.setattr(
            "cron.executions.recover_interrupted_executions", _boom
        )

        assert _run_tick() == 0


class TestOneShotCliRunIsSynchronous:
    @pytest.fixture(autouse=True)
    def _restore_async_delivery_flag(self):
        from gateway.session_context import _SESSION_ASYNC_DELIVERY, _UNSET

        token = _SESSION_ASYNC_DELIVERY.set(_UNSET)
        yield
        _SESSION_ASYNC_DELIVERY.reset(token)

    def test_cli_run_declares_stateless_channel_before_dispatch(self, monkeypatch):
        """`hermes cron run` must gate off async delivery so the run executes
        synchronously in the CLI process instead of on a doomed daemon thread."""
        from gateway.session_context import async_delivery_supported
        from hermes_cli import cron as cron_cli

        observed = {}

        def _fake_cron_api(**kwargs):
            observed["async_delivery"] = async_delivery_supported()
            return {"success": True, "job": {"executed": True, "execution_success": True}}

        monkeypatch.setattr(cron_cli, "_cron_api", _fake_cron_api)

        assert cron_cli._job_action("run", "job-123", "Triggered") == 0
        assert observed["async_delivery"] is False
        # Scoped declaration: the capability must be restored after the call
        # so in-process callers (tests, embedding apps) are not tainted.
        assert async_delivery_supported() is True

    def test_non_run_actions_leave_channel_capability_alone(self, monkeypatch):
        from gateway.session_context import async_delivery_supported
        from hermes_cli import cron as cron_cli

        observed = {}

        def _fake_cron_api(**kwargs):
            observed["async_delivery"] = async_delivery_supported()
            return {"success": True, "job": {"name": "j"}}

        monkeypatch.setattr(cron_cli, "_cron_api", _fake_cron_api)

        cron_cli._job_action("pause", "job-123", "Paused")
        assert observed["async_delivery"] is True

    def test_background_dispatch_refused_when_channel_stateless(self, monkeypatch):
        """End-to-end gate: with the stateless declaration active, the cron
        tool's background dispatcher must fall back to synchronous execution
        (return None) even when a session key is inherited from a gateway env."""
        from gateway.session_context import declare_stateless_channel
        from tools.cronjob_tools import _try_dispatch_background_run

        declare_stateless_channel()
        monkeypatch.setenv("HERMES_SESSION_KEY", "inherited-gateway-session")

        result = _try_dispatch_background_run(
            {"id": "job-x", "name": "job-x"}, session_id="sess-1"
        )
        assert result is None
