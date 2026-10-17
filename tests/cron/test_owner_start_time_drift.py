"""Start-time fingerprint drift between claim and recovery reads (#117505).

``_owner_is_live()`` used to require the recomputed start-time fingerprint to match the one
recorded at claim time within a fixed 2 s tolerance. On hosts where same-host readings drift
without bound (measured 2026-10-17: 1064 of 1073 live-owner rows mismatched the tolerance, one
pid spanning +4400 centiseconds in 13 days), a live execution was reconciled to ``unknown``
while its process was still alive. A mismatched fingerprint is now NEVER treated as death: a
live pid stays owned, and a genuinely wedged claim is released by the stale-bound sweep instead.
"""

from __future__ import annotations

import pytest

import cron.executions as executions_mod


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    import gateway.status

    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    # The owner process exists for every row in this module; only the fingerprint decides.
    monkeypatch.setattr(gateway.status, "_pid_exists", lambda pid: True)


def _seed_external_running(monkeypatch, *, claimed_start_time: int) -> str:
    """Running row owned by a foreign process, with the given claim-time fingerprint."""
    with monkeypatch.context() as m:
        m.setattr(executions_mod, "_PROCESS_ID", "ext-proc")
        m.setattr(executions_mod, "_process_start_time", lambda pid: claimed_start_time)
        record = executions_mod.create_execution("job-drift", source="cron")
        executions_mod.mark_execution_running(record["id"])
    return record["id"]


def _status(execution_id: str) -> str:
    return executions_mod.latest_execution("job-drift")["status"]


@pytest.mark.parametrize(
    "drift",
    [100, 3600, 500000],
)
def test_any_drift_keeps_execution_running(monkeypatch, drift):
    """Recovery never reaps a claimed/running row whose owner pid is alive, no matter how far
    the fingerprint drifted: no fixed magnitude can distinguish drift from PID reuse. The
    stale-bound sweep reclaims a genuinely wedged claim."""
    claimed = 178864182760  # the exact reading from the report
    execution_id = _seed_external_running(monkeypatch, claimed_start_time=claimed)
    monkeypatch.setattr(executions_mod, "_process_start_time", lambda pid: claimed + drift)

    recovered = executions_mod.recover_interrupted_executions()

    assert recovered == 0
    assert _status(execution_id) == "running"


def test_unreadable_start_time_fail_safe(monkeypatch):
    """A start time that cannot be read now proves nothing about death: the row must survive."""
    execution_id = _seed_external_running(monkeypatch, claimed_start_time=178864182760)
    monkeypatch.setattr(executions_mod, "_process_start_time", lambda pid: None)

    assert executions_mod.recover_interrupted_executions() == 0
    assert _status(execution_id) == "running"
