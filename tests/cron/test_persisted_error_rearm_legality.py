"""The persisted-error re-arm must respect cron schedule legality.

Salvage follow-up on PR #87261: re-arming a wedged CRON job to ``now`` would
fire it at times the expression explicitly excludes (a weekday-only 9am job
whose Friday run errored would fire on SATURDAY).  Cron jobs re-arm to the
next LEGAL occurrence instead; interval jobs (the 2026-08-14 incident class)
still re-arm to now — an immediate catch-up is always legal for intervals.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

pytest.importorskip("croniter")

import cron.jobs as J


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(J, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(J, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(J, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(J, "OUTPUT_DIR", hermes_home / "cron" / "output")
    J._cron_cadence_cache.clear()
    monkeypatch.setattr(J, "_persisted_error_recoveries", 0)
    monkeypatch.setattr(J, "_persisted_error_recoveries_recent", [])
    return hermes_home


def _wedge(job_id: str, *, next_run_at: datetime, last_run_at: datetime) -> None:
    J.update_job(
        job_id,
        {
            "next_run_at": next_run_at.isoformat(),
            "last_status": "error",
            "last_error": "EAGAIN [Errno 11] Resource temporarily unavailable",
            "last_run_at": last_run_at.isoformat(),
        },
    )


class TestCronRearmRespectsScheduleLegality:
    def test_weekday_job_errored_friday_does_not_fire_on_saturday(self, cron_store):
        """A '0 9 * * 1-5' job whose Friday 9am run errored must re-arm to
        Monday 9am (the next legal occurrence), NOT to Saturday-now."""
        job = J.create_job(prompt="weekday report", schedule="0 9 * * 1-5")
        # Saturday 2026-08-22 12:00 UTC; Friday's run errored at 9am.
        now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
        _wedge(
            job["id"],
            # parked well past Monday (the wedge shape: stale error + future park)
            next_run_at=datetime(2026, 8, 26, 9, 0, tzinfo=timezone.utc),
            last_run_at=datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc),
        )
        with mock.patch.object(J, "_hermes_now", lambda: now):
            due = J.get_due_jobs()
        assert due == [], "Saturday must not fire a weekday-only job"
        rearmed = J.get_job(job["id"])
        next_dt = datetime.fromisoformat(rearmed["next_run_at"])
        assert next_dt == datetime(2026, 8, 24, 9, 0, tzinfo=timezone.utc), (
            f"expected re-arm to Monday 9am, got {rearmed['next_run_at']}"
        )
        # The recovery is still counted (the wedge WAS repaired).
        assert J.get_persisted_error_recovery_stats()["persisted_error_recoveries"] >= 1

    def test_cron_value_parked_at_next_legal_occurrence_left_alone(self, cron_store):
        """A wedged-looking job whose next_run_at already IS the next legal
        occurrence needs no repair — re-arm must be a no-op."""
        job = J.create_job(prompt="daily", schedule="30 14 * * *")
        now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
        legal_next = datetime(2026, 8, 22, 14, 30, tzinfo=timezone.utc)
        _wedge(
            job["id"],
            next_run_at=legal_next,
            last_run_at=now - timedelta(hours=27),
        )
        with mock.patch.object(J, "_hermes_now", lambda: now):
            J.get_due_jobs()
        assert datetime.fromisoformat(J.get_job(job["id"])["next_run_at"]) == legal_next
        assert J.get_persisted_error_recovery_stats()["persisted_error_recoveries"] == 0

    def test_interval_job_still_rearms_to_now(self, cron_store):
        """Interval jobs (the 2026-08-14 incident class) keep the immediate
        catch-up: re-armed to now, due on this same scan."""
        scripts_dir = cron_store / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        (scripts_dir / "p.py").write_text("print('probe')\n", encoding="utf-8")
        job = J.create_job(prompt="probe", schedule="every 10m", no_agent=True, script="p.py")
        now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
        _wedge(
            job["id"],
            next_run_at=now + timedelta(minutes=5),
            last_run_at=now - timedelta(minutes=110),
        )
        with mock.patch.object(J, "_hermes_now", lambda: now):
            due = J.get_due_jobs()
        assert any(j["id"] == job["id"] for j in due), (
            "wedged interval job must become due immediately after re-arm"
        )
