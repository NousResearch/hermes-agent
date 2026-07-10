"""Regression tests for cron reliability safeguards (Kanban t_b934a5b3).

Four safeguards, each proven independently:

  1. job-loss detection      — a job that vanishes from jobs.json out-of-band
                               is detectable against the persisted census.
  2. missed-run detection    — an enabled job whose next_run_at is long past is
                               surfaced (the scheduler never fired it).
  3. durable delivery status — mark_job_run() persists a positive/negative
                               delivery outcome, not just an error string.
  4. visible failure state   — failing jobs are surfaced even when they are
                               enabled=False (which list_jobs() hides by default).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cron import jobs as cron_jobs
from cron import reliability


UTC = timezone.utc


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# 2. Missed-run detection (pure)
# ---------------------------------------------------------------------------
class TestFindMissedRuns:
    def test_overdue_enabled_job_is_missed(self):
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        jobs = [
            {"id": "a", "name": "old", "enabled": True,
             "next_run_at": _iso(now - timedelta(hours=3))},
        ]
        missed = reliability.find_missed_runs(jobs, now, grace_seconds=600)
        assert [j["id"] for j in missed] == ["a"]

    def test_future_job_not_missed(self):
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        jobs = [{"id": "a", "enabled": True, "next_run_at": _iso(now + timedelta(hours=1))}]
        assert reliability.find_missed_runs(jobs, now, grace_seconds=600) == []

    def test_within_grace_not_missed(self):
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        jobs = [{"id": "a", "enabled": True, "next_run_at": _iso(now - timedelta(seconds=30))}]
        assert reliability.find_missed_runs(jobs, now, grace_seconds=600) == []

    def test_disabled_and_paused_not_missed(self):
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        past = _iso(now - timedelta(hours=3))
        jobs = [
            {"id": "disabled", "enabled": False, "next_run_at": past},
            {"id": "paused", "enabled": True, "state": "paused", "next_run_at": past},
        ]
        assert reliability.find_missed_runs(jobs, now, grace_seconds=600) == []

    def test_missing_or_unparseable_next_run_ignored(self):
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        jobs = [
            {"id": "none", "enabled": True, "next_run_at": None},
            {"id": "bad", "enabled": True, "next_run_at": "not-a-date"},
        ]
        assert reliability.find_missed_runs(jobs, now, grace_seconds=600) == []

    def test_in_flight_job_not_missed(self):
        """A job currently running (run_claim/fire_claim stamped) keeps its
        next_run_at in the past for the duration of the run — not a miss."""
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        past = _iso(now - timedelta(hours=3))
        jobs = [
            {"id": "running", "enabled": True, "next_run_at": past,
             "run_claim": {"at": past, "by": "host"}},
            {"id": "firing", "enabled": True, "next_run_at": past,
             "fire_claim": {"at": past, "by": "host"}},
        ]
        assert reliability.find_missed_runs(jobs, now, grace_seconds=600) == []

    def test_once_job_not_missed(self):
        """One-shot jobs never advance next_run_at, so a past value is expected,
        not a missed run."""
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        jobs = [
            {"id": "once", "enabled": True, "schedule": {"kind": "once"},
             "next_run_at": _iso(now - timedelta(hours=3))},
        ]
        assert reliability.find_missed_runs(jobs, now, grace_seconds=600) == []

    def test_recurring_overdue_still_missed(self):
        """A recurring job that is not in flight and overdue is still a miss."""
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        jobs = [
            {"id": "cronjob", "enabled": True, "schedule": {"kind": "interval"},
             "next_run_at": _iso(now - timedelta(hours=3))},
        ]
        assert [j["id"] for j in reliability.find_missed_runs(jobs, now, grace_seconds=600)] == ["cronjob"]


# ---------------------------------------------------------------------------
# 4. Visible failure state (pure) — surfaces jobs list_jobs() hides
# ---------------------------------------------------------------------------
class TestFindFailedJobs:
    def test_last_error_surfaced_even_when_disabled(self):
        jobs = [
            {"id": "ok", "enabled": True, "last_status": "ok"},
            {"id": "err", "enabled": False, "last_status": "error",
             "last_error": "boom"},
        ]
        assert [j["id"] for j in reliability.find_failed_jobs(jobs)] == ["err"]

    def test_delivery_failure_surfaced(self):
        jobs = [{"id": "d", "enabled": True, "last_status": "ok",
                 "last_delivery_error": "telegram 500"}]
        assert [j["id"] for j in reliability.find_failed_jobs(jobs)] == ["d"]

    def test_error_state_surfaced(self):
        jobs = [{"id": "s", "enabled": True, "state": "error"}]
        assert [j["id"] for j in reliability.find_failed_jobs(jobs)] == ["s"]

    def test_healthy_jobs_not_surfaced(self):
        jobs = [{"id": "ok", "enabled": True, "last_status": "ok",
                 "state": "scheduled", "last_error": None,
                 "last_delivery_error": None}]
        assert reliability.find_failed_jobs(jobs) == []


# ---------------------------------------------------------------------------
# 1. Job-loss detection
# ---------------------------------------------------------------------------
class TestDetectDroppedJobs:
    def test_pure_diff_reports_missing_ids(self):
        current = [{"id": "b"}, {"id": "c"}]
        dropped = reliability.detect_dropped_jobs(["a", "b", "c"], current)
        assert dropped == ["a"]

    def test_no_loss_when_all_present(self):
        current = [{"id": "a"}, {"id": "b"}]
        assert reliability.detect_dropped_jobs(["a", "b"], current) == []

    def test_census_written_on_save_and_detects_out_of_band_loss(self, tmp_path):
        import json

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.create_job(prompt="one", schedule="every 1h")
            cron_jobs.create_job(prompt="two", schedule="every 2h")
            saved_ids = {j["id"] for j in cron_jobs.load_jobs()}
            # save_jobs() must persist a census of the known job ids.
            census_ids = cron_jobs.known_job_ids()
            assert census_ids == saved_ids

            # Simulate silent out-of-band loss: a writer that bypassed the
            # sanctioned API drops one job from jobs.json.
            jobs_file = cron_jobs._current_cron_store().jobs_file
            data = json.loads(jobs_file.read_text())
            dropped_id = data["jobs"].pop()["id"]
            jobs_file.write_text(json.dumps(data))

            missing = reliability.detect_dropped_jobs(
                cron_jobs.known_job_ids(), cron_jobs.load_jobs()
            )
            assert missing == [dropped_id]


# ---------------------------------------------------------------------------
# 3. Durable delivery status
# ---------------------------------------------------------------------------
class TestDurableDeliveryStatus:
    def test_successful_delivery_recorded(self, tmp_path):
        with cron_jobs.use_cron_store(tmp_path):
            job = cron_jobs.create_job(prompt="x", schedule="every 1h")
            cron_jobs.mark_job_run(job["id"], success=True, delivered=True)
            stored = cron_jobs.get_job(job["id"])
            assert stored["last_delivery_status"] == "delivered"
            assert stored.get("last_delivered_at")

    def test_delivery_failure_recorded(self, tmp_path):
        with cron_jobs.use_cron_store(tmp_path):
            job = cron_jobs.create_job(prompt="x", schedule="every 1h")
            cron_jobs.mark_job_run(
                job["id"], success=True, delivery_error="telegram 500", delivered=True
            )
            stored = cron_jobs.get_job(job["id"])
            assert stored["last_delivery_status"] == "failed"

    def test_skipped_delivery_recorded(self, tmp_path):
        with cron_jobs.use_cron_store(tmp_path):
            job = cron_jobs.create_job(prompt="x", schedule="every 1h")
            cron_jobs.mark_job_run(job["id"], success=True, delivered=False)
            stored = cron_jobs.get_job(job["id"])
            assert stored["last_delivery_status"] == "skipped"


# ---------------------------------------------------------------------------
# Combined health audit
# ---------------------------------------------------------------------------
class TestLogCronHealth:
    def test_never_raises_on_non_string_id(self, tmp_path):
        """A corrupt jobs.json (non-string id) is exactly the input this feature
        observes; log_cron_health must audit it without raising (the ticker loop
        depends on it being fully inert on failure)."""
        import json

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.ensure_dirs()
            jobs_file = cron_jobs._current_cron_store().jobs_file
            jobs_file.write_text(json.dumps(
                {"jobs": [{"id": 123, "enabled": True, "last_status": "error"}]}
            ))
            report = reliability.log_cron_health()  # must not raise
            assert 123 in report["failed"]

    def test_persistent_condition_logged_once(self, tmp_path, caplog):
        """A standing failure state is warned when it appears, not re-flooded
        every tick."""
        import json
        import logging

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.ensure_dirs()
            jobs_file = cron_jobs._current_cron_store().jobs_file
            jobs_file.write_text(json.dumps(
                {"jobs": [{"id": "x", "enabled": True, "state": "error"}]}
            ))
            reliability._last_warned.clear()
            with caplog.at_level(logging.WARNING, logger="cron.reliability"):
                reliability.log_cron_health()
                first = len([r for r in caplog.records if "failure state" in r.getMessage()])
                reliability.log_cron_health()  # unchanged → should not re-warn
                total = len([r for r in caplog.records if "failure state" in r.getMessage()])
        assert first == 1
        assert total == 1


class TestAuditCronHealth:
    def test_audit_reports_all_dimensions(self):
        now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        jobs = [
            {"id": "missed", "enabled": True,
             "next_run_at": _iso(now - timedelta(hours=3))},
            {"id": "failed", "enabled": False, "last_delivery_error": "x"},
        ]
        report = reliability.audit_cron_health(
            jobs, now, expected_ids=["missed", "failed", "gone"], grace_seconds=600
        )
        assert "missed" in report["missed_runs"]
        assert "failed" in report["failed"]
        assert "gone" in report["dropped"]
