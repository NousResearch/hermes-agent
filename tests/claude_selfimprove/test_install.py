from __future__ import annotations

import json

from claude_selfimprove import install, paths


def _seed_jobs_json(sandbox, job_names):
    cron_dir = sandbox.hermes_home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    jobs = [{"id": f"id-{i}", "name": name} for i, name in enumerate(job_names)]
    (cron_dir / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def _fake_creator(calls):
    def creator(name, schedule, script, *, hermes_bin="hermes"):
        calls.append((name, schedule, script))
        return True, f"Created job: fake-{name}"

    return creator


def test_deploys_package_and_both_entry_scripts(sandbox):
    calls = []
    result = install.run(job_creator=_fake_creator(calls))

    scripts_dir = sandbox.hermes_home / "scripts"
    assert (scripts_dir / "claude_selfimprove" / "__init__.py").exists()
    assert (scripts_dir / "claude_selfimprove" / "pipeline.py").exists()
    assert (scripts_dir / "claude_selfimprove_scan.py").exists()
    assert (scripts_dir / "claude_selfimprove_consolidate.py").exists()
    assert result.deployed_package
    assert result.deployed_scan_script
    assert result.deployed_consolidate_script


def test_deploy_excludes_pycache(sandbox):
    # Force a __pycache__ to exist in the real source package so the test
    # actually exercises the exclusion, not just its absence.
    pycache = install.PACKAGE_DIR / "__pycache__"
    pycache.mkdir(exist_ok=True)
    (pycache / "junk.pyc").write_bytes(b"junk")
    try:
        install.run(job_creator=_fake_creator([]))
        deployed_pycache = sandbox.hermes_home / "scripts" / "claude_selfimprove" / "__pycache__"
        assert not deployed_pycache.exists()
    finally:
        import shutil
        shutil.rmtree(pycache, ignore_errors=True)


def test_creates_both_jobs_on_a_fresh_profile(sandbox):
    calls = []
    result = install.run(job_creator=_fake_creator(calls))

    assert result.nightly_job_created
    assert result.weekly_job_created
    assert not result.nightly_job_already_existed
    assert not result.weekly_job_already_existed
    assert result.ok

    names = [c[0] for c in calls]
    assert names == [install.NIGHTLY_JOB_NAME, install.WEEKLY_JOB_NAME]
    schedules = {c[0]: c[1] for c in calls}
    assert schedules[install.NIGHTLY_JOB_NAME] == install.NIGHTLY_SCHEDULE
    assert schedules[install.WEEKLY_JOB_NAME] == install.WEEKLY_SCHEDULE
    scripts = {c[0]: c[2] for c in calls}
    assert scripts[install.NIGHTLY_JOB_NAME] == install.SCAN_SCRIPT_NAME
    assert scripts[install.WEEKLY_JOB_NAME] == install.CONSOLIDATE_SCRIPT_NAME


def test_skips_job_creation_when_already_registered(sandbox):
    _seed_jobs_json(sandbox, [install.NIGHTLY_JOB_NAME, install.WEEKLY_JOB_NAME])
    calls = []
    result = install.run(job_creator=_fake_creator(calls))

    assert calls == []  # neither job creator call happened
    assert result.nightly_job_already_existed
    assert result.weekly_job_already_existed
    assert not result.nightly_job_created
    assert not result.weekly_job_created
    assert result.ok
    # The deploy step still runs even when jobs already exist - a repeat
    # install must still push the latest code.
    assert (sandbox.hermes_home / "scripts" / "claude_selfimprove" / "pipeline.py").exists()


def test_creates_only_the_missing_job(sandbox):
    _seed_jobs_json(sandbox, [install.NIGHTLY_JOB_NAME])
    calls = []
    result = install.run(job_creator=_fake_creator(calls))

    assert result.nightly_job_already_existed
    assert result.weekly_job_created
    assert [c[0] for c in calls] == [install.WEEKLY_JOB_NAME]


def test_running_twice_never_duplicates(sandbox):
    calls = []
    install.run(job_creator=_fake_creator(calls))
    # Second run must see the first run's jobs via jobs.json. Simulate that
    # by seeding jobs.json the way a real `hermes cron create` would have
    # left it.
    _seed_jobs_json(sandbox, [install.NIGHTLY_JOB_NAME, install.WEEKLY_JOB_NAME])
    result2 = install.run(job_creator=_fake_creator(calls))

    assert len(calls) == 2  # only from the first run
    assert result2.nightly_job_already_existed
    assert result2.weekly_job_already_existed


def test_dry_run_creates_nothing_and_calls_no_creator(sandbox):
    calls = []
    result = install.run(dry_run=True, job_creator=_fake_creator(calls))

    assert calls == []
    assert not (sandbox.hermes_home / "scripts" / "claude_selfimprove").exists()
    assert result.nightly_job_created  # "would create"
    assert result.weekly_job_created


def test_job_creation_failure_is_recorded_as_an_error(sandbox):
    def failing_creator(name, schedule, script, *, hermes_bin="hermes"):
        return False, f"hermes cron create exited 1 for {name}"

    result = install.run(job_creator=failing_creator)
    assert not result.ok
    assert any(install.NIGHTLY_JOB_NAME in e or "nightly" in e for e in result.errors) or result.errors
    assert len(result.errors) == 2  # both jobs failed


def test_existing_job_names_handles_missing_and_corrupt_jobs_file(sandbox):
    assert install._existing_job_names(sandbox.hermes_home) == set()

    cron_dir = sandbox.hermes_home / "cron"
    cron_dir.mkdir(parents=True)
    (cron_dir / "jobs.json").write_text("not json", encoding="utf-8")
    assert install._existing_job_names(sandbox.hermes_home) == set()


def test_real_job_creator_missing_hermes_binary_fails_safe(sandbox, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    ok, msg = install._create_job_via_cli("name", "0 3 * * *", "script.py")
    assert ok is False
    assert "not found" in msg.lower() or "no such file" in msg.lower()
