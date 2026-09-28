"""Live-but-deadlocked cron workers (#115692).

A worker wedged on a lock passes ``_owner_is_live()`` forever, so its ``running`` row was never
reclaimed and every future fire of the job was skipped. ``recover_interrupted_executions`` now
also releases a live-owned claim once it outlives a bound derived from the existing timeouts
(``max(3 × HERMES_CRON_TIMEOUT, script timeout, 7200)``) and fails closed when the inactivity
timeout is unlimited or not a finite positive number.
"""

from __future__ import annotations

import time
import uuid
from datetime import timedelta

import pytest

import cron.executions as executions_mod
from cron.executions import _hermes_now, _transaction, recover_interrupted_executions


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    # The owner is alive for every row in this module; the bound alone decides.
    monkeypatch.setattr(executions_mod, "_owner_is_live", lambda pid, started_at: True)


def _seed_running(job_id: str, *, age_seconds: float, pid: int = 99999) -> str:
    claimed_at = (_hermes_now() - timedelta(seconds=age_seconds)).isoformat()
    with _transaction() as conn:
        eid = uuid.uuid4().hex
        conn.execute(
            """INSERT INTO executions
               (id, job_id, status, source, process_id, pid, process_started_at, claimed_at)
               VALUES (?, ?, 'running', 'cron', ?, ?, ?, ?)""",
            (eid, job_id, "ext-proc", pid, int(time.time()), claimed_at),
        )
    return eid


def _status(eid: str) -> str:
    with _transaction() as conn:
        return conn.execute("SELECT status FROM executions WHERE id=?", (eid,)).fetchone()["status"]


def test_live_owner_stale_claim_gets_recovered(monkeypatch):
    """A live-owned claim older than the derived bound is released as ``unknown``; the bound
    follows the configured timeouts rather than a fixed wall-clock constant."""
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    bound = executions_mod._live_owner_stale_after_seconds()
    assert bound is not None
    assert bound == pytest.approx(7200.0)  # max(3×600, 3600 script default, 7200 floor)

    eid = _seed_running("job-stale", age_seconds=bound + 60)
    assert recover_interrupted_executions() == 1
    assert _status(eid) == "unknown"

    # A larger inactivity timeout widens the bound: the same age is no longer stale.
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "3600")
    assert executions_mod._live_owner_stale_after_seconds() == pytest.approx(10800.0)
    eid2 = _seed_running("job-long", age_seconds=bound + 60)
    assert recover_interrupted_executions() == 0
    assert _status(eid2) == "running"


@pytest.mark.parametrize("timeout", ["600", "0", "nan", "inf"])
def test_live_owner_recent_claim_not_recovered(monkeypatch, timeout):
    """A live owner with a recent claim is left alone; with an unlimited (0) or non-finite
    inactivity timeout there is no bound to derive, so even an ancient live-owned claim is
    left alone (fail closed)."""
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", timeout)
    recent = _seed_running("job-recent", age_seconds=60, pid=88888)
    ancient = _seed_running("job-ancient", age_seconds=30 * 24 * 3600, pid=77777)

    recovered = recover_interrupted_executions()

    assert _status(recent) == "running"
    if timeout == "600":
        assert recovered == 1 and _status(ancient) == "unknown"
    else:
        assert executions_mod._live_owner_stale_after_seconds() is None
        assert recovered == 0 and _status(ancient) == "running"
