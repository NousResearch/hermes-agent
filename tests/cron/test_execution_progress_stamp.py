"""The run monitor stamps ``progress_at`` on the execution row while the agent is active.

The live-owner stale-claim sweep (#115692) reclaims a running row once it has been silent
for the derived bound. Before this stamp existed the bound was measured from ``claimed_at``,
so any healthy job that simply ran longer than ~2 h was reclaimed mid-run, its fire claim
yanked and the run aborted with "lost its durable fire claim ownership".
"""

from __future__ import annotations

import threading

import pytest

import cron.executions as executions_mod
from cron import scheduler
from cron.executions import _transaction, create_execution, mark_execution_running


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")


def _progress_at(execution_id: str):
    with _transaction() as conn:
        return conn.execute(
            "SELECT progress_at FROM executions WHERE id=?", (execution_id,)
        ).fetchone()["progress_at"]


def test_active_run_stamps_progress_and_idle_run_does_not(monkeypatch):
    stamp_every = 0.05
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", stamp_every)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 3600.0)
    # The monitor polls concurrent.futures.wait(timeout=_POLL_INTERVAL); shorten the local
    # constant so the stamp cadence is observable within the test.
    monkeypatch.setattr(
        scheduler.concurrent.futures, "wait",
        lambda fs, timeout=None: scheduler.concurrent.futures._base.wait(fs, timeout=0.01),
    )

    finish = threading.Event()

    class Agent:
        def __init__(self, idle):
            self._idle = idle

        def get_activity_summary(self):
            return {"seconds_since_activity": self._idle}

        def run_conversation(self, *a, **k):
            finish.wait(2)
            return {"final_response": "ok"}

        def interrupt(self, *a, **k):
            finish.set()

    def run(idle_seconds):
        finish.clear()
        execution = create_execution("job-progress", source="cron")
        mark_execution_running(execution["id"])
        job = {"id": "job-progress", "execution_id": execution["id"], "schedule": {"kind": "every"}}
        timer = threading.Timer(0.4, finish.set)
        timer.start()
        try:
            scheduler._run_agent_with_watchdog(
                Agent(idle_seconds), "probe", job, "job-progress", "job-progress", "task", None)
        finally:
            timer.cancel()
        return _progress_at(execution["id"])

    # Active agent (idle 0s): the row is stamped.
    assert run(idle_seconds=0.0) is not None
    # An agent that reports idleness past the stamp cadence is silent in the ledger too: a wedged
    # worker must not keep refreshing its own liveness.
    assert run(idle_seconds=stamp_every * 10) is None
