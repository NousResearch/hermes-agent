"""Reclaimed (unknown) cron executions must not disappear silently (#108802).

An owner that dies before a durable terminal state leaves its execution row to
``recover_interrupted_executions()``; the reaper only marked the row ``unknown``
— no incident, no failure-deliver — so a broken job stayed invisible in
``hermes cron incidents`` and never reached the operator's failure lane.
"""
from cron import scheduler as sched
from cron.executions import (
    create_execution,
    mark_execution_running,
    recover_interrupted_executions,
)


def _seed_dead_owner_execution(monkeypatch, job_id="job-reclaimed"):
    """Create a claimed/running execution whose owner looks provably gone."""
    monkeypatch.setattr(sched, "_last_dead_owner_reap_at", None)
    executions_mod = __import__("cron.executions", fromlist=["executions"])
    monkeypatch.setattr(executions_mod, "_PROCESS_ID", "proc-other")
    record = create_execution(job_id, source="direct")
    mark_execution_running(record["id"])
    # From the reaper's point of view: a different process, owner proved dead.
    monkeypatch.setattr(executions_mod, "_PROCESS_ID", "proc-current")
    monkeypatch.setattr(executions_mod, "_owner_is_live", lambda pid, started: False)
    return record


def test_recover_invokes_on_recovered_with_record(monkeypatch):
    from cron import executions as ex

    record = _seed_dead_owner_execution(monkeypatch)
    seen = []
    count = recover_interrupted_executions(on_recovered=seen.append)

    assert count == 1
    assert len(seen) == 1
    assert seen[0]["id"] == record["id"]
    assert seen[0]["job_id"] == "job-reclaimed"
    assert seen[0]["status"] == "unknown"


def test_recover_without_callback_is_backward_compatible(monkeypatch):
    _seed_dead_owner_execution(monkeypatch)
    assert recover_interrupted_executions() == 1


def test_maybe_reap_dead_owners_delivers_and_opens_incident(monkeypatch):
    delivered = []

    def fake_recover(*, on_recovered=None):
        on_recovered({"id": "exec-1", "job_id": "job-x", "status": "unknown",
                      "error": "Scheduler restarted after this execution's owner exited"})
        return 1

    monkeypatch.setattr(sched, "_last_dead_owner_reap_at", None)
    monkeypatch.setattr("cron.executions.recover_interrupted_executions", fake_recover)
    monkeypatch.setattr(
        sched, "_deliver_crash_failure",
        lambda job, err, *, adapters=None, loop=None:
            delivered.append((job["id"], err)) or (None, "delivered"))
    monkeypatch.setattr("cron.jobs.get_job", lambda job_id: {"id": job_id, "name": "weekly"})

    sched._maybe_reap_dead_owners(adapters=object(), loop=None)

    assert delivered, "reclaimed-unknown execution must reach the failure lane"
    assert delivered[0][0] == "job-x"
    assert "owner exited" in delivered[0][1]


def test_maybe_reap_dead_owners_job_missing_is_safe(monkeypatch):
    delivered = []

    def fake_recover(*, on_recovered=None):
        on_recovered({"id": "exec-2", "job_id": "gone", "status": "unknown", "error": "x"})
        return 1

    monkeypatch.setattr(sched, "_last_dead_owner_reap_at", None)
    monkeypatch.setattr("cron.executions.recover_interrupted_executions", fake_recover)
    monkeypatch.setattr(
        sched, "_deliver_crash_failure",
        lambda job, err, *, adapters=None, loop=None: delivered.append(job) or (None, "delivered"))
    monkeypatch.setattr("cron.jobs.get_job", lambda job_id: None)

    sched._maybe_reap_dead_owners(adapters=object(), loop=None)

    assert delivered == []
