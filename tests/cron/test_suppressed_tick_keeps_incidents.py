"""Agent-less ticks (monitor no_change / wake-gate skip) are not recovery evidence (#140178).

A monitor-gated job that keeps failing records an incident and a failure_streak on each agent
run. Between runs the monitor keeps ticking; every unchanged tick used to be recorded as a
successful run, which resolved the open incident and reset ``failure_streak`` — so
``cron.failure_repeat_alert_hours`` (alert-once) and the streak nudge never applied and the
operator got a fresh failure ping on every single failing run. A tick where no agent ran says
nothing about job health: it must leave the incident open and the streak intact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron import incidents
import cron.scheduler as sched
from cron.jobs import create_job, mark_job_run


def _point_db(monkeypatch, tmp_path):
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return incidents


def test_suppressed_success_run_leaves_open_incident_open(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "provider 503")
    sched._mark_incident_alerted(inc_id)

    sched._compose_run_delivery(
        {"id": "job-1", "name": "j1"}, success=True, error=None, final_response="ignored",
        output_file=None, suppressed=True)

    body = inc.get_incident(inc_id)
    assert body["state"] == "alerted"  # unchanged: still open, still in its alert window
    assert body["closed_at"] is None


def test_suppressed_success_run_does_not_reset_streak_or_status(monkeypatch, tmp_path):
    _point_db(monkeypatch, tmp_path)
    from cron.jobs import get_job
    job = create_job("monitored", "*/5 * * * *")
    assert mark_job_run(job["id"], False, "provider 503")
    assert get_job(job["id"])["failure_streak"] == 1

    # An agent-less tick is recorded exactly as the scheduler does it (suppressed=True),
    # mirroring finish_completed_run -> mark_job_run(..., suppressed=d.suppressed).
    assert mark_job_run(job["id"], True, None, suppressed=True)

    after = get_job(job["id"])
    assert after["failure_streak"] == 1  # NOT reset by the no_change tick
    assert after["last_status"] == "error"  # last run of the agent is still the truth
    assert after["last_error"] == "provider 503"

    # A real (non-suppressed) success still recovers the job — regression guard.
    assert mark_job_run(job["id"], True, None)
    healthy = get_job(job["id"])
    assert healthy["failure_streak"] == 0
    assert healthy["last_status"] == "ok"
    assert healthy["last_error"] is None


def test_suppressed_tick_still_releases_fire_claims(monkeypatch, tmp_path):
    _point_db(monkeypatch, tmp_path)
    from cron.jobs import get_job
    job = create_job("monitored", "*/5 * * * *")
    assert mark_job_run(job["id"], True, None, suppressed=True)
    after = get_job(job["id"])
    assert after["fire_claim"] is None  # fire machinery must release on every recorded run
    assert after["last_run_at"]  # the tick did happen — last_run_at advances
