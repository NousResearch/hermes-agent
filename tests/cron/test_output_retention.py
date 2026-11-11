"""Run-output retention tiers: silent runs never evict the runs that reported something."""

from datetime import datetime, timedelta

import pytest


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    clock = iter(datetime(2026, 6, 25, 10, 0, 0) + timedelta(seconds=i) for i in range(10_000))
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: next(clock))  # one distinct file name per save
    cfg = {"cron": {}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
    return cfg["cron"]


def _names(job_id):
    from cron.jobs import _job_output_dir

    return sorted(p.name for p in _job_output_dir(job_id).glob("*.md"))


def test_silent_heartbeats_do_not_evict_reported_runs(cron_store):
    from cron.jobs import save_job_output

    cron_store.update(output_retention=3, output_retention_silent=2)
    reports = [save_job_output("watch", f"report {i}").name for i in range(2)]
    silents = [save_job_output("watch", "[SILENT]", silent=True).name for _ in range(10)]

    assert _names("watch") == sorted(reports + silents[-2:])
    assert all(name.endswith(".silent.md") for name in silents)


def test_silent_tier_defaults_to_twenty(cron_store):
    from cron.jobs import save_job_output

    for _ in range(25):
        save_job_output("watch", "[SILENT]", silent=True)
    assert len(_names("watch")) == 20


def test_per_job_override_caps_both_tiers(cron_store):
    from cron.jobs import save_job_output

    job = {"id": "watch", "output_retention": 1}
    for _ in range(3):
        save_job_output("watch", "report", job=job)
        save_job_output("watch", "[SILENT]", job=job, silent=True)

    names = _names("watch")
    assert len(names) == 2 and sum(n.endswith(".silent.md") for n in names) == 1


def test_non_positive_per_job_override_disables_pruning(cron_store):
    from cron.jobs import save_job_output

    cron_store.update(output_retention=1)
    for _ in range(4):
        save_job_output("keep-all", "report", job={"id": "keep-all", "output_retention": 0})
    assert len(_names("keep-all")) == 4


def test_output_retention_is_stored_validated_and_clearable(cron_store):
    from cron.jobs import create_job, get_job, update_job

    job = create_job(prompt="check", schedule="every 1h", output_retention="5")
    assert get_job(job["id"])["output_retention"] == 5
    for flag in (True, False):  # booleans are rejected, never coerced to 1/0
        with pytest.raises(ValueError):
            create_job(prompt="check", schedule="every 1h", output_retention=flag)
        with pytest.raises(ValueError):
            update_job(job["id"], {"output_retention": flag})
    with pytest.raises(ValueError):
        update_job(job["id"], {"output_retention": "many"})
    update_job(job["id"], {"output_retention": ""})
    assert get_job(job["id"]).get("output_retention") is None
    plain = create_job(prompt="check", schedule="every 1h")
    assert "output_retention" not in get_job(plain["id"])  # unset jobs stay byte-identical


@pytest.mark.parametrize("value", [0.5, 1.9, -0.5, "0.5", float("nan"), float("inf")])
def test_fractional_output_retention_is_rejected_not_truncated(cron_store, value):
    from cron.jobs import create_job, update_job

    # int(0.5) == 0 would silently disable pruning and int(1.9) == 1 would keep less than asked.
    with pytest.raises(ValueError, match="output_retention must be an integer or empty"):
        create_job(prompt="check", schedule="every 1h", output_retention=value)
    job = create_job(prompt="check", schedule="every 1h")
    with pytest.raises(ValueError, match="output_retention must be an integer or empty"):
        update_job(job["id"], {"output_retention": value})


def test_integral_float_output_retention_is_accepted(cron_store):
    from cron.jobs import create_job, get_job

    job = create_job(prompt="check", schedule="every 1h", output_retention=5.0)
    assert get_job(job["id"])["output_retention"] == 5


def test_fractional_global_caps_fall_back_to_the_defaults(cron_store):
    from cron.output_retention import output_keep

    cron_store.update(output_retention=0.5, output_retention_silent=1.9)
    assert output_keep(None, silent=False) == 50  # not 0 (= never prune)
    assert output_keep(None, silent=True) == 20  # not 1
    cron_store.update(output_retention=30.0, output_retention_silent="4")
    assert output_keep(None, silent=False) == 30
    assert output_keep(None, silent=True) == 4


def test_output_retention_is_an_authored_field():
    """Profile distributions must carry the field (job_definition.JOB_DEFINITION_FIELDS)."""
    from cron.job_definition import JOB_DEFINITION_FIELDS

    assert "output_retention" in JOB_DEFINITION_FIELDS


def test_dashboard_reads_the_timestamp_of_a_silent_run_document(tmp_path):
    from hermes_cli.web_routers.cron import _cron_output_run_timestamp

    plain = _cron_output_run_timestamp(tmp_path / "2026-06-25_10-00-00.md")
    silent = _cron_output_run_timestamp(tmp_path / "2026-06-25_10-00-00.silent.md")
    assert plain is not None and silent == plain


def test_cronjob_tool_sets_shows_and_keeps_zero(cron_store):
    import json

    from cron.jobs import load_jobs
    from tools.cronjob_tools import cronjob

    created = json.loads(cronjob(action="create", prompt="watch", schedule="every 1h", output_retention=0))
    assert created["success"] is True, created
    assert load_jobs()[0]["output_retention"] == 0  # 0 = never prune, not "unset"
    listed = json.loads(cronjob(action="list"))
    assert listed["jobs"][0]["output_retention"] == 0

    json.loads(cronjob(action="update", job_id=created["job_id"], output_retention=7))
    assert load_jobs()[0]["output_retention"] == 7
