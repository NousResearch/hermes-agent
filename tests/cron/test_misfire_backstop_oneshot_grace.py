"""Misfire-backstop grace gate for one-shot cron jobs (#93526).

``fire_overdue_jobs`` in ``cron/scheduler_provider.py`` is the hosted-provider
misfire catch-up: any runnable job whose ``next_run_at`` is older than
``cron.misfire_grace_minutes`` gets fired locally. Before the fix it applied
no one-shot grace check, so a stored past-due one-shot bypassed
ONESHOT_GRACE_SECONDS and fired arbitrarily late after downtime — the exact
"will never fire" contract violation the due-scan gate (#89571) closes on the
built-in scheduler path.

Pins:
  - one-shot beyond ONESHOT_GRACE_SECONDS -> backstop skips it (no claim)
  - recurring job equally overdue          -> backstop still fires it
"""

from datetime import timedelta

from cron.jobs import ONESHOT_GRACE_SECONDS, _hermes_now
from cron.scheduler_provider import fire_overdue_jobs


class _RecordingProvider:
    """Non-InProcess provider double that records claim attempts."""

    def __init__(self):
        self.claimed = []

    def claim_fire(self, job_id):
        self.claimed.append(job_id)
        return None  # decline the claim so no thread is spawned

    def fire_claimed(self, *a, **k):  # pragma: no cover - never reached
        raise AssertionError("fire_claimed must not run when claim declined")


def _job(jid, kind, run_at_dt):
    schedule = {"kind": kind}
    if kind == "once":
        schedule["run_at"] = run_at_dt.isoformat()
    else:
        schedule["expr"] = "*/5 * * * *"
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": schedule,
        "next_run_at": run_at_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 1 if kind == "once" else None, "completed": 0},
        "deliver": "local",
    }


def _run_backstop(monkeypatch, jobs):
    provider = _RecordingProvider()
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
    monkeypatch.setattr(
        "cron.scheduler_provider._misfire_grace_minutes", lambda: 5.0
    )
    fired = fire_overdue_jobs(provider)
    return provider, fired


class TestMisfireBackstopOneShotGrace:
    def test_stale_oneshot_beyond_grace_is_not_fired(self, monkeypatch):
        now = _hermes_now()
        stale = _job("stale-once", "once", now - timedelta(hours=2))
        provider, fired = _run_backstop(monkeypatch, [stale])
        assert fired == 0
        assert provider.claimed == []

    def test_overdue_recurring_job_still_fires(self, monkeypatch):
        now = _hermes_now()
        overdue = _job("overdue-cron", "cron", now - timedelta(hours=2))
        provider, fired = _run_backstop(monkeypatch, [overdue])
        assert provider.claimed == ["overdue-cron"]

    def test_oneshot_within_oneshot_grace_but_past_misfire_grace(
        self, monkeypatch
    ):
        # misfire grace patched to ~0 so a 60s-late one-shot is "overdue" for
        # the backstop but still inside ONESHOT_GRACE_SECONDS -> must fire.
        assert ONESHOT_GRACE_SECONDS > 60
        now = _hermes_now()
        recent = _job("recent-once", "once", now - timedelta(seconds=60))
        provider = _RecordingProvider()
        monkeypatch.setattr("cron.jobs.load_jobs", lambda: [recent])
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        monkeypatch.setattr(
            "cron.scheduler_provider._misfire_grace_minutes", lambda: 0.5
        )
        fire_overdue_jobs(provider)
        assert provider.claimed == ["recent-once"]
