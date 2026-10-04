"""``hermes cron list`` aggregates every profile's cron store (#132920).

Each profile owns its own ``cron/jobs.json``, and ``cron_list`` reads the store
pinned to the active profile's HERMES_HOME — so a job created in another profile
was invisible, and two profiles could hold same-named jobs with no way to tell
them apart. The listing now walks every live profile's store, names the owner of
each job, and dedups by job id with default-profile precedence (a job copied into
a second profile during profile creation is one job, not two).
"""

import json

import pytest

from hermes_cli.cron import _aggregate_cron_jobs, _read_jobs_for_home, cron_list


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _store(home, jobs):
    """Write a jobs.json straight into ``home/cron`` — no scheduler involved."""
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    (cron_dir / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def _job(job_id, name, **extra):
    job = {
        "id": job_id,
        "name": name,
        "prompt": f"run {name}",
        "enabled": True,
        "schedule": {"kind": "cron", "expr": "0 9 * * *", "value": "0 9 * * *"},
    }
    job.update(extra)
    return job


class TestAggregateCronJobs:
    def test_reads_every_profile_store(self, tmp_path):
        default_home = tmp_path / "home"
        other_home = tmp_path / "profiles" / "research"
        _store(default_home, [_job("a1", "default-job")])
        _store(other_home, [_job("b2", "other-job")])

        rows = _aggregate_cron_jobs(
            [("default", default_home), ("research", other_home)])

        assert {row[0] for row in rows} == {"default", "research"}
        assert [row[1]["name"] for row in rows] == ["default-job", "other-job"]

    def test_same_named_jobs_in_two_profiles_both_survive(self, tmp_path):
        default_home = tmp_path / "home"
        other_home = tmp_path / "profiles" / "research"
        # Same name, different ids: the report's "all different but with same names" case.
        _store(default_home, [_job("a1", "shared-name")])
        _store(other_home, [_job("b2", "shared-name")])

        rows = _aggregate_cron_jobs(
            [("default", default_home), ("research", other_home)])

        assert [(row[0], row[1]["id"], row[1]["name"]) for row in rows] == [
            ("default", "a1", "shared-name"),
            ("research", "b2", "shared-name"),
        ]

    def test_duplicate_id_prefers_default_profile(self, tmp_path):
        default_home = tmp_path / "home"
        other_home = tmp_path / "profiles" / "research"
        # A job copied into a second profile during profile creation (#51721).
        _store(default_home, [_job("dup", "daily", prompt="from default")])
        _store(other_home, [_job("dup", "daily", prompt="from research")])

        rows = _aggregate_cron_jobs(
            [("default", default_home), ("research", other_home)])

        assert len(rows) == 1
        assert rows[0][0] == "default"
        assert rows[0][1]["prompt"] == "from default"

    def test_duplicate_id_default_wins_regardless_of_order(self, tmp_path):
        default_home = tmp_path / "home"
        other_home = tmp_path / "profiles" / "research"
        _store(default_home, [_job("dup", "daily", prompt="from default")])
        _store(other_home, [_job("dup", "daily", prompt="from research")])

        rows = _aggregate_cron_jobs(
            [("research", other_home), ("default", default_home)])

        assert len(rows) == 1
        assert rows[0][0] == "default"

    def test_missing_id_sentinel_never_dedups(self, tmp_path):
        default_home = tmp_path / "home"
        other_home = tmp_path / "profiles" / "research"
        # cron.jobs._normalize_job_record() fills a missing id with the literal
        # sentinel "unknown" — truthy, so a plain falsy check would collapse two
        # genuinely different id-less jobs from different profiles into one.
        _store(default_home, [_job("unknown", "alpha")])
        _store(other_home, [_job("unknown", "beta")])

        rows = _aggregate_cron_jobs(
            [("default", default_home), ("research", other_home)])

        assert len(rows) == 2
        assert {row[1]["name"] for row in rows} == {"alpha", "beta"}

    def test_unreadable_store_does_not_hide_the_others(self, tmp_path):
        default_home = tmp_path / "home"
        other_home = tmp_path / "profiles" / "research"
        _store(default_home, [_job("a1", "default-job")])
        _store(other_home, [_job("b2", "other-job")])

        def _selective(home):
            if home == other_home:
                raise OSError("permission denied")
            return _read_jobs_for_home(home)

        rows = _aggregate_cron_jobs(
            [("default", default_home), ("research", other_home)],
            read_jobs=_selective)

        assert [row[1]["name"] for row in rows] == ["default-job"]


class TestCronListRendersOwner:
    def test_multi_profile_listing_names_each_owner(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.cron._warn_if_gateway_not_running", lambda: None)

        def _profiles():
            default_home = tmp_path / "home"
            other_home = tmp_path / "profiles" / "research"
            _store(default_home, [_job("a1", "default-job")])
            _store(other_home, [_job("b2", "other-job")])
            return [("default", default_home), ("research", other_home)]

        monkeypatch.setattr("hermes_cli.cron._cron_profile_stores", _profiles)

        cron_list()

        out = capsys.readouterr().out
        assert "default-job" in out
        assert "other-job" in out
        assert "Profile:" in out
        assert "research" in out
        assert "default" in out