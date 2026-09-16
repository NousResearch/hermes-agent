"""hermes doctor surfaces cron jobs whose successful runs were never delivered."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hermes_cli.doctor_cron import _check_cron_delivery


def _run_check(monkeypatch, tmp_path, jobs):
    """Point the cron store at a throwaway home and capture the check's rows."""
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    rows = []

    def fake_warn(text, detail=""):
        rows.append(("warn", text, detail))

    def fake_info(text):
        rows.append(("info", text, None))

    monkeypatch.setattr("hermes_cli.doctor_cron.check_warn", fake_warn)
    monkeypatch.setattr("hermes_cli.doctor_cron.check_info", fake_info)
    finding = _check_cron_delivery(False)
    return rows, finding


def _job(**overrides):
    job = {
        "id": "job-1",
        "name": "nightly report",
        "enabled": True,
        "state": "scheduled",
        "last_status": "ok",
        "last_delivery_error": None,
    }
    job.update(overrides)
    return job


class TestCronDeliveryDoctorCheck:
    def test_healthy_store_stays_silent(self, monkeypatch, tmp_path, capsys):
        jobs = [_job(), _job(id="job-2", name="daily digest")]
        rows, finding = _run_check(monkeypatch, tmp_path, jobs)

        assert rows == [("info", "last results delivered for all 2 job(s)", None)]
        assert finding.manual_issues == []
        assert capsys.readouterr().out == ""

    def test_delivery_failed_job_warns_once(self, monkeypatch, tmp_path):
        jobs = [
            _job(),
            _job(
                id="job-2",
                name="daily digest",
                last_status="delivery_failed",
                last_delivery_error="telegram: chat not found: Chat not found",
            ),
        ]
        rows, finding = _run_check(monkeypatch, tmp_path, jobs)

        warns = [r for r in rows if r[0] == "warn"]
        assert len(warns) == 1
        assert "daily digest" in warns[0][1]
        assert "not delivered" in warns[0][1]
        assert "chat not found" in warns[0][2]
        assert len(finding.manual_issues) == 1
        assert "cron doctor" in finding.manual_issues[0]

    def test_recovery_clears_the_warning(self, monkeypatch, tmp_path):
        jobs = [_job(last_status="delivery_failed", last_delivery_error="boom")]
        rows, finding = _run_check(monkeypatch, tmp_path, jobs)
        assert any(r[0] == "warn" for r in rows)

        jobs[0]["last_status"] = "ok"
        jobs[0]["last_delivery_error"] = None
        rows, finding = _run_check(monkeypatch, tmp_path, jobs)
        assert not any(r[0] == "warn" for r in rows)
        assert finding.manual_issues == []

    def test_error_delivery_and_unrelated_statuses_do_not_warn(
        self, monkeypatch, tmp_path
    ):
        jobs = [
            _job(id="err", last_status="error", last_error="agent crashed"),
            _job(id="q", last_status="delivery_queued"),
            _job(id="stale", last_status="delivery_failed", last_delivery_error=None),
        ]
        rows, finding = _run_check(monkeypatch, tmp_path, jobs)

        assert [r for r in rows if r[0] == "warn"] == [
            (
                "warn",
                "cron job 'nightly report' ran successfully but its result was not delivered",
                "(no details)",
            )
        ]
        assert len(finding.manual_issues) == 1
