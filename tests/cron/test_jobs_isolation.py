"""
Test for issue #60014: HERMES_HOME test isolation in cron.jobs module-level constants.

Tests that dynamically resolve HERMES_HOME respect test monkeypatch isolation,
preventing cron jobs from leaking into production storage during test runs.
"""

import tempfile
from pathlib import Path
import pytest


def test_jobs_file_path_respects_hermes_home_monkeypatch(tmp_path, monkeypatch):
    """Verify that _get_jobs_file() respects HERMES_HOME changes via monkeypatch.
    
    This is a regression test for issue #60014: module-level constants in
    cron/jobs.py were computed at import time, so tests that set
    monkeypatch.setenv("HERMES_HOME", tmp_path) would still write to the
    production jobs.json because the constant was frozen.
    
    The fix uses dynamic accessor functions (_get_jobs_file(), etc.) that
    call get_hermes_home() at runtime instead of at import time.
    """
    import cron.jobs
    
    # Set HERMES_HOME to a temporary directory
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    # The dynamic getter should now resolve to the temp path
    jobs_file = cron.jobs._get_jobs_file()
    
    # Verify it points to the temp directory, not the production one
    assert jobs_file == tmp_path / "cron" / "jobs.json"
    assert str(tmp_path) in str(jobs_file)


def test_cron_dir_respects_hermes_home_monkeypatch(tmp_path, monkeypatch):
    """Verify that _get_cron_dir() respects HERMES_HOME changes via monkeypatch."""
    import cron.jobs
    
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    cron_dir = cron.jobs._get_cron_dir()
    
    assert cron_dir == tmp_path / "cron"
    assert str(tmp_path) in str(cron_dir)


def test_output_dir_respects_hermes_home_monkeypatch(tmp_path, monkeypatch):
    """Verify that _get_output_dir() respects HERMES_HOME changes via monkeypatch."""
    import cron.jobs
    
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    output_dir = cron.jobs._get_output_dir()
    
    assert output_dir == tmp_path / "cron" / "output"
    assert str(tmp_path) in str(output_dir)


def test_load_jobs_uses_correct_isolated_home(tmp_path, monkeypatch):
    """Verify that load_jobs() uses the isolated HERMES_HOME, not production."""
    import cron.jobs
    import json
    
    # Set up a temp HERMES_HOME with a known jobs.json
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    # Create the cron directory and jobs.json in the temp home
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    
    jobs_file = cron_dir / "jobs.json"
    test_job = {
        "id": "test-job",
        "name": "test",
        "prompt": "echo test",
        "schedule": "every 5m",
        "paused": False,
    }
    with open(jobs_file, "w") as f:
        json.dump({"jobs": [test_job]}, f)
    
    # Load jobs should read from the temp directory, not production
    jobs = cron.jobs.load_jobs()
    
    assert len(jobs) == 1
    assert jobs[0]["id"] == "test-job"


def test_save_jobs_uses_correct_isolated_home(tmp_path, monkeypatch):
    """Verify that save_jobs() writes to the isolated HERMES_HOME."""
    import cron.jobs
    import json
    
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    # Create a test job
    test_job = {
        "id": "save-test",
        "name": "save_test",
        "prompt": "echo test",
        "schedule": "every 10m",
        "paused": False,
    }
    
    # Save the job
    cron.jobs.save_jobs([test_job])
    
    # Verify it was written to the temp directory
    jobs_file = tmp_path / "cron" / "jobs.json"
    assert jobs_file.exists(), f"jobs.json should exist at {jobs_file}"
    
    with open(jobs_file, "r") as f:
        data = json.load(f)
    
    assert len(data["jobs"]) == 1
    assert data["jobs"][0]["id"] == "save-test"


def test_hermes_home_override_module_level_var(tmp_path, monkeypatch):
    """Verify that setting _hermes_home module var overrides get_hermes_home()."""
    import cron.jobs
    
    # Override via the module-level _hermes_home variable
    cron.jobs._hermes_home = Path(str(tmp_path))
    
    try:
        jobs_file = cron.jobs._get_jobs_file()
        assert jobs_file == tmp_path / "cron" / "jobs.json"
    finally:
        # Clean up
        cron.jobs._hermes_home = None
