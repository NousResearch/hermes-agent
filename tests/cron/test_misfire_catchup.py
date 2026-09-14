"""Tests for fire_overdue_jobs — misfire catch-up for external cron providers.

External providers (Chronos) deliver scheduled fires over HTTP; when the
loopback hop is down at fire time and the scheduler's retry budget exhausts,
the job's next_run_at stays parked in the past and nothing ever runs it
(external providers have no local tick loop). fire_overdue_jobs, called from
gateway housekeeping, claims and fires those jobs after a grace window.
"""

import threading
import time
from datetime import timedelta

import pytest

from cron.jobs import _hermes_now, create_job, get_job, load_jobs, save_jobs
from cron.scheduler_provider import (
    CronScheduler,
    InProcessCronScheduler,
    fire_overdue_jobs,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class RecordingProvider(CronScheduler):
    """External-provider stand-in: real base-class claim_fire (store CAS),
    recorded fire_claimed instead of the blocking run_one_job."""

    def __init__(self):
        self.fired = []
        self._done = threading.Event()

    @property
    def name(self):
        return "recording"

    def start(self, stop_event, **kw):  # pragma: no cover - unused
        return None

    def fire_claimed(self, claimed_job, *, adapters=None, loop=None,
                     cancel_event=None):
        self.fired.append(claimed_job["id"])
        self._done.set()
        return True

    def wait_fired(self, timeout=5.0):
        return self._done.wait(timeout)


def _park_in_past(job_id, minutes):
    """Rewind a job's next_run_at into the past (simulates missed fires)."""
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["next_run_at"] = (
                _hermes_now() - timedelta(minutes=minutes)
            ).isoformat()
    save_jobs(jobs)


class TestFireOverdueJobs:
    def test_noop_for_builtin_provider(self, tmp_cron_dir):
        """The in-process ticker self-heals past-due jobs — the sweep must
        never double-dispatch under it."""
        job = create_job(prompt="p", schedule="every 1h")
        _park_in_past(job["id"], minutes=60)
        assert fire_overdue_jobs(InProcessCronScheduler()) == 0

    def test_fires_job_past_grace(self, tmp_cron_dir):
        job = create_job(prompt="p", schedule="every 1h")
        _park_in_past(job["id"], minutes=30)
        provider = RecordingProvider()
        assert fire_overdue_jobs(provider) == 1
        assert provider.wait_fired()
        assert provider.fired == [job["id"]]

    def test_respects_grace_window(self, tmp_cron_dir):
        """A job only a few minutes overdue is still the external
        scheduler's to retry — the backstop must not race it."""
        job = create_job(prompt="p", schedule="every 1h")
        _park_in_past(job["id"], minutes=5)  # < default 10 min grace
        provider = RecordingProvider()
        assert fire_overdue_jobs(provider) == 0
        assert provider.fired == []

    def test_grace_zero_disables(self, tmp_cron_dir, monkeypatch):
        monkeypatch.setattr(
            "cron.scheduler_provider._misfire_grace_minutes", lambda: 0.0
        )
        job = create_job(prompt="p", schedule="every 1h")
        _park_in_past(job["id"], minutes=600)
        assert fire_overdue_jobs(RecordingProvider()) == 0

    def test_future_job_not_fired(self, tmp_cron_dir):
        create_job(prompt="p", schedule="every 1h")  # next_run_at in future
        provider = RecordingProvider()
        assert fire_overdue_jobs(provider) == 0

    def test_paused_job_not_fired(self, tmp_cron_dir):
        from cron.jobs import pause_job

        job = create_job(prompt="p", schedule="every 1h")
        _park_in_past(job["id"], minutes=30)
        pause_job(job["id"])
        assert fire_overdue_jobs(RecordingProvider()) == 0

    def test_fresh_external_claim_wins(self, tmp_cron_dir):
        """A concurrent external fire holds the store claim — the sweep's
        claim_fire loses the CAS and must not dispatch (at-most-once)."""
        from cron.jobs import claim_job_for_fire

        job = create_job(prompt="p", schedule="every 1h")
        _park_in_past(job["id"], minutes=30)
        assert claim_job_for_fire(job["id"]) is True  # external fire claims

        # The external claim also advanced next_run_at (recurring bump), so
        # re-park it to isolate the claim-CAS as the thing that blocks us.
        _park_in_past(job["id"], minutes=30)

        provider = RecordingProvider()
        assert fire_overdue_jobs(provider) == 0
        assert provider.fired == []

    def test_claim_advances_next_run_no_refire(self, tmp_cron_dir):
        """After a catch-up fire, the recurring job's next_run_at moved to
        the future — the next sweep pass must not fire it again."""
        job = create_job(prompt="p", schedule="every 1h")
        _park_in_past(job["id"], minutes=30)
        provider = RecordingProvider()
        assert fire_overdue_jobs(provider) == 1
        assert provider.wait_fired()

        stamped = get_job(job["id"])
        assert stamped["next_run_at"] > _hermes_now().isoformat()

        provider2 = RecordingProvider()
        assert fire_overdue_jobs(provider2) == 0

    def test_dispatch_is_nonblocking(self, tmp_cron_dir):
        """fire_claimed runs off-thread — a slow job must not stall the
        sweep (housekeeping loop) for the length of an agent run."""

        class SlowProvider(RecordingProvider):
            def fire_claimed(self, claimed_job, **kw):
                time.sleep(3.0)
                return super().fire_claimed(claimed_job, **kw)

        job = create_job(prompt="p", schedule="every 1h")
        _park_in_past(job["id"], minutes=30)
        provider = SlowProvider()
        start = time.monotonic()
        assert fire_overdue_jobs(provider) == 1
        assert time.monotonic() - start < 1.0  # returned before the run
        assert provider.wait_fired(timeout=10)
        assert provider.fired == [job["id"]]
