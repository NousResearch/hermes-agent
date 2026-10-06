"""Behavioral RED probe: on upstream/main, `cron list` cannot see another profile's job.

Imports only `cron_list`, which already exists on main, so this file collects against
unmodified source and fails on the BEHAVIOR rather than on a missing symbol. Kept as a
regression test for the cross-store read path the aggregate relies on.
"""

import json

import pytest

from cron.jobs import use_cron_store
from hermes_cli.cron import cron_list


def _store(home, jobs):
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    (cron_dir / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def _job(job_id, name):
    return {
        "id": job_id,
        "name": name,
        "prompt": f"run {name}",
        "enabled": True,
        "schedule": {"kind": "cron", "expr": "0 9 * * *", "value": "0 9 * * *"},
    }


def test_cron_list_shows_a_job_owned_by_another_profile(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("hermes_cli.cron._warn_if_gateway_not_running", lambda: None)

    default_home = tmp_path / "home"
    other_home = tmp_path / "profiles" / "research"
    _store(default_home, [_job("a1", "default-job")])
    _store(other_home, [_job("b2", "other-job")])

    from cron.jobs import _current_cron_store, list_jobs as real_list_jobs

    def _store_aware_list_jobs(include_disabled: bool = False):
        """Serve the fixture stores only; never touch the real active store."""
        jobs_file = str(_current_cron_store().jobs_file)
        if jobs_file == str(default_home / "cron" / "jobs.json"):
            return [_job("a1", "default-job")]
        if jobs_file == str(other_home / "cron" / "jobs.json"):
            return [_job("b2", "other-job")]
        return real_list_jobs(include_disabled=include_disabled)

    monkeypatch.setattr("cron.jobs.list_jobs", _store_aware_list_jobs)
    monkeypatch.setattr("hermes_cli.cron._cron_profile_stores",
                        lambda: [("default", default_home), ("research", other_home)],
                        raising=False)

    with use_cron_store(default_home):
        cron_list()

    out = capsys.readouterr().out
    assert "default-job" in out
    assert "other-job" in out, "cron list hid the other profile's job"
    assert "Profile:" in out, "cron list does not name the owning profile"