"""`hermes cron doctor` flags a fire that burned far more tokens than the job usually does."""

import json

from cron import jobs
from hermes_cli import cron


def _audit(tmp_path, monkeypatch, fires):
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    lines = [json.dumps({"job_id": job_id, "total_tokens": tokens}) for job_id, tokens in fires]
    (cron_dir / "usage_audit.jsonl").write_text("\n".join(lines + ["not json"]) + "\n", encoding="utf-8")


def test_last_fire_far_above_the_usual_is_reported(tmp_path, monkeypatch):
    # total_tokens=null = a run that failed before usage was known: out of the baseline, not zero.
    usual = [("runaway", 10_000_000)] * 4 + [("runaway", None)] * 6 + [("steady", 10_000_000)] * 6 + [("tiny", 50_000)] * 5
    _audit(tmp_path, monkeypatch, usual + [("runaway", 84_000_000), ("tiny", 400_000)])

    outliers = cron._cron_spend_outliers()

    assert set(outliers) == {"runaway"}  # steady is normal, tiny stays under the 1M floor
    assert "84.0M tokens, 8x its usual 10.0M" in outliers["runaway"]
    issues = cron._cron_doctor_issues_for_job({"id": "runaway", "next_run_at": None, "state": "paused"},
                                              outliers)
    assert issues == [outliers["runaway"]]


def test_warning_clears_once_the_next_fire_is_normal(tmp_path, monkeypatch):
    _audit(tmp_path, monkeypatch, [("job", 10_000_000)] * 5 + [("job", 84_000_000), ("job", 11_000_000)])

    assert cron._cron_spend_outliers() == {}
