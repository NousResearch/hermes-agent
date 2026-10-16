"""Tests for the cron job ``category`` field (dashboard groups the job list by it)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


class TestJobCategoryField:
    def test_create_persists_trimmed_category(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(prompt="Piano practice", schedule="every 1d", category="  Family  ")

        assert job["category"] == "Family"
        loaded = get_job(job["id"])
        assert loaded is not None
        assert loaded["category"] == "Family"

    def test_unset_and_blank_categories_are_not_labels(self, cron_env):
        from cron.jobs import create_job, get_job

        unset = create_job(prompt="No label", schedule="every 1d")
        blank = create_job(prompt="Blank label", schedule="every 1d", category="   ")

        assert unset.get("category") is None
        assert blank.get("category") is None
        loaded = get_job(unset["id"])
        assert loaded is not None
        assert loaded.get("category") is None

    def test_update_sets_trims_and_clears(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="Task", schedule="every 1d")

        labelled = update_job(job["id"], {"category": " Tech "})
        assert labelled is not None
        assert labelled["category"] == "Tech"
        # An empty string clears the label instead of storing a blank bucket.
        cleared = update_job(job["id"], {"category": ""})
        assert cleared is not None
        assert cleared["category"] is None

    def test_category_is_inert_for_scheduling(self, cron_env):
        """A label is presentation-only: the job still has a next run."""
        from cron.jobs import create_job

        job = create_job(prompt="Task", schedule="every 2h", category="Backups")

        assert job["category"] == "Backups"
        assert job["next_run_at"]
        assert job["schedule"]["kind"] == "interval"
