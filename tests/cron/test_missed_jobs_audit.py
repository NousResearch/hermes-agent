"""Regression tests for the missed-jobs audit log write path (#54349).

Review point (teknium1): ``CRON_DIR`` is only the default-profile fallback
on current main. ``_log_missed_job`` is reachable inside a
``use_cron_store(home)`` context (profile-scoped dashboard operations route
``cron.jobs`` calls through it), so the audit event must resolve its output
directory via ``_current_cron_store()`` — otherwise a named profile's missed
jobs would be logged into the default profile's cron directory.
"""
import json

import cron.jobs as jobs


def _sample_job():
    return {
        "id": "job-nightly",
        "name": "nightly-report",
        "schedule": {"expr": "0 3 * * *", "display": "daily at 03:00"},
    }


def test_audit_line_lands_in_active_store(tmp_path, monkeypatch):
    """Inside ``use_cron_store(home)`` the audit line goes to ``<home>/cron``
    — never to the module-level default store."""
    default_dir = tmp_path / "default_home" / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", default_dir)

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)

    with jobs.use_cron_store(profile_home):
        jobs._log_missed_job(
            _sample_job(),
            "2026-07-29T03:00:00+00:00",
            600,
            "2026-07-30T03:00:00+00:00",
        )

    log_file = profile_home / "cron" / "missed_jobs.jsonl"
    assert log_file.exists()
    entry = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert entry["job_id"] == "job-nightly"
    assert entry["fast_forwarded_to"] == "2026-07-30T03:00:00+00:00"
    assert entry["schedule"] == "daily at 03:00"
    # The default store must stay untouched — writing there is exactly the
    # cross-profile leak the review flagged.
    assert not default_dir.exists()


def test_audit_line_falls_back_to_default_store(tmp_path, monkeypatch):
    """Outside any override, the audit line uses the module-level store."""
    default_dir = tmp_path / "default_home" / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", default_dir)

    jobs._log_missed_job(
        _sample_job(),
        "2026-07-29T03:00:00+00:00",
        600,
        "2026-07-30T03:00:00+00:00",
    )

    log_file = default_dir / "missed_jobs.jsonl"
    assert log_file.exists()
    entry = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert entry["job_id"] == "job-nightly"
