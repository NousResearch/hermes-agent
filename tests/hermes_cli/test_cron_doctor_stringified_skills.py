"""Regression: doctor flags skills stored as a stringified list, even though the read
path heals the shape (so doctor must compare against the raw stored record)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hermes_cli.cron import _skills_repr_strings, _cron_doctor_issues_for_job


class TestDoctorFlagsStringifiedSkills:
    def test_repr_string_in_skills_flagged(self):
        job = {"id": "j1", "skills": ["['x']"], "skill": None}
        assert _skills_repr_strings(job) == ["['x']"]
        issues = _cron_doctor_issues_for_job(job, raw_job=job)
        assert any("stringified list" in i for i in issues)

    def test_repr_in_legacy_skill_flagged(self):
        job = {"id": "j1", "skills": [], "skill": "['x', 'y']"}
        assert _skills_repr_strings(job) == ["['x', 'y']"]

    def test_healed_view_alone_not_flagged(self):
        # list_jobs() heals on read; doctor only flags when the RAW record carries it.
        raw = {"id": "j1", "skills": ["['x']"], "skill": None}
        healed = {"id": "j1", "skills": ["x"], "skill": "x"}
        assert _skills_repr_strings(healed) == []
        issues_with_raw = _cron_doctor_issues_for_job(healed, raw_job=raw)
        assert any("stringified list" in i for i in issues_with_raw)
        assert not any("stringified list" in i for i in _cron_doctor_issues_for_job(healed, raw_job=healed))

    def test_normal_skills_not_flagged(self):
        job = {"id": "j1", "skills": ["newsy", "gif-search"], "skill": "newsy"}
        assert _skills_repr_strings(job) == []

    def test_e2e_doctor_reports_stored_corruption(self, tmp_path, monkeypatch, capsys):
        from argparse import Namespace
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
        from cron.jobs import create_job, load_jobs, save_jobs
        from hermes_cli.cron import cron_command

        create_job(prompt="Watch the news", schedule="every 1h")
        jobs = load_jobs()
        jobs[0]["skills"] = "['newsy']"
        jobs[0]["skill"] = "['newsy']"
        save_jobs(jobs)

        rc = cron_command(Namespace(cron_command="doctor"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "stringified list" in out
        assert jobs[0]["id"] in out