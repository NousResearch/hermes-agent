"""Tests for repeat value normalization — string synonyms must never crash (t_cd6a18cc).

Regression suite: cronjob(action=create|update) with repeat="forever" (or
other string synonyms / zero / negative ints) must either create/update a
job cleanly or return a ValueError, NOT raise a raw TypeError on `<=` comparison.

Fix landed: _normalize_repeat_value() in cron/jobs.py, called from both
create_job entry point and the tool's update path.
"""

import json
import pytest

from cron.jobs import (
    _normalize_repeat_value,
    create_job,
    get_job,
    load_jobs,
)


# ===========================================================================
# Direct unit tests for _normalize_repeat_value
# ===========================================================================

class TestNormalizeRepeatValueDirect:
    """Unit tests for the normalization helper itself."""

    def test_none_passes_through(self):
        assert _normalize_repeat_value(None) is None

    def test_string_forever_becomes_none(self):
        assert _normalize_repeat_value("forever") is None

    def test_string_forever_case_insensitive(self):
        assert _normalize_repeat_value("FOREVER") is None
        assert _normalize_repeat_value("Forever") is None
        assert _normalize_repeat_value("  FOREVER  ") is None

    def test_string_infinite_becomes_none(self):
        assert _normalize_repeat_value("infinite") is None
        assert _normalize_repeat_value("Infinite") is None

    def test_string_inf_becomes_none(self):
        assert _normalize_repeat_value("inf") is None

    def test_empty_string_becomes_none(self):
        assert _normalize_repeat_value("") is None

    def test_zero_int_becomes_none(self):
        assert _normalize_repeat_value(0) is None

    def test_negative_int_becomes_none(self):
        assert _normalize_repeat_value(-1) is None
        assert _normalize_repeat_value(-42) is None

    def test_zero_string_becomes_none(self):
        assert _normalize_repeat_value("0") is None

    def test_positive_int_preserved(self):
        assert _normalize_repeat_value(1) == 1
        assert _normalize_repeat_value(99) == 99

    def test_numeric_string_converted_to_int(self):
        assert _normalize_repeat_value("3") == 3
        assert _normalize_repeat_value("50") == 50

    def test_invalid_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="repeat must be an integer"):
            _normalize_repeat_value("notanumber")

    def test_invalid_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="repeat must be an integer"):
            _normalize_repeat_value([])

    def test_invalid_type_dict_raises_valueerror(self):
        with pytest.raises(ValueError, match="repeat must be an integer"):
            _normalize_repeat_value({"key": "val"})


# ===========================================================================
# Integration tests: create_job with string repeat values
# ===========================================================================

class TestCreateJobWithStringRepeat:
    """create_job must accept repeat as string without raising TypeError."""

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        p = tmp_path / "cron"
        monkeypatch.setattr("cron.jobs.CRON_DIR", p)
        monkeypatch.setattr("cron.jobs.JOBS_FILE", p / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", p / "output")

    def test_create_with_repeat_forever_string(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 1h", repeat="forever")
        # The raw job record has {"times": None} which means infinite
        assert job["repeat"]["times"] is None
        assert job["state"] == "scheduled"

    def test_create_with_repeat_inf_string(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 2h", repeat="inf")
        assert job["repeat"]["times"] is None

    def test_create_with_repeat_zero_string(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 1h", repeat="0")
        assert job["repeat"]["times"] is None

    def test_create_with_repeat_integer_still_works(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 1h", repeat=5)
        assert job["repeat"]["times"] == 5

    def test_create_with_no_repeat_default(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 1h")
        assert job["repeat"]["times"] is None

    def test_create_interval_no_repeat_is_finite(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 1h")
        assert job["repeat"]["times"] is None


# ===========================================================================
# Integration tests: tool cronjob(action=create|update) with string repeat
# ===========================================================================

class TestCronjobToolWithStringRepeat:
    """The cronjob() tool must handle string repeat values without crashing.

    Note: the tool's _format_job() outputs "repeat" as a display string
    via _repeat_display(job), e.g. "forever", "once", "5 times".
    We assert on the formatted output string AND verify the stored record
    directly via load_jobs().
    """

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        p = tmp_path / "cron"
        monkeypatch.setattr("cron.jobs.CRON_DIR", p)
        monkeypatch.setattr("cron.jobs.JOBS_FILE", p / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", p / "output")

    def test_tool_create_with_repeat_forever(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                prompt="Check server status",
                schedule="every 1h",
                name="Server Check",
                repeat="forever",
            )
        )
        assert result["success"] is True
        # Formatted output shows "forever" (string display)
        assert result["job"]["repeat"] == "forever"
        # Stored record confirms times=None (infinite)
        jobs = load_jobs()
        assert len(jobs) == 1
        assert jobs[0]["repeat"]["times"] is None

    def test_tool_create_with_repeat_inf(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(action="create", prompt="test", schedule="every 2h", repeat="inf")
        )
        assert result["success"] is True
        assert result["job"]["repeat"] == "forever"
        jobs = load_jobs()
        assert jobs[0]["repeat"]["times"] is None

    def test_tool_update_with_repeat_forever(self, tmp_path, monkeypatch):
        """Original crash site: update via tool with repeat='forever'."""
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(action="create", prompt="Baseline", schedule="every 1h", repeat=5)
        )
        assert created["success"] is True
        job_id = created["job_id"]
        assert created["job"]["repeat"] == "5 times"

        updated = json.loads(
            cronjob(action="update", job_id=job_id, repeat="forever")
        )
        assert updated["success"] is True
        assert updated["job"]["repeat"] == "forever"
        # Verify stored record
        jobs = load_jobs()
        assert jobs[0]["repeat"]["times"] is None

    def test_tool_update_with_repeat_zero_string(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(action="create", prompt="test", schedule="every 1h", repeat=10)
        )
        assert created["success"] is True
        job_id = created["job_id"]

        updated = json.loads(
            cronjob(action="update", job_id=job_id, repeat="0")
        )
        assert updated["success"] is True
        assert updated["job"]["repeat"] == "forever"
        jobs = load_jobs()
        assert jobs[0]["repeat"]["times"] is None

    def test_tool_create_with_numeric_string_repeat(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(action="create", prompt="test", schedule="every 1h", repeat="3")
        )
        assert result["success"] is True
        assert result["job"]["repeat"] == "3 times"

    def test_tool_invalid_repeat_returns_clean_error(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(action="create", prompt="test", schedule="every 1h", repeat="invalidvalue")
        )
        assert result["success"] is False
        assert "repeat must be an integer" in result.get("error", "")
        # No job should have been persisted
        jobs = load_jobs()
        assert len(jobs) == 0


# ===========================================================================
# Backward-compatibility: existing int-repeat behaviour unchanged
# ===========================================================================

class TestRepeatIntBackwardCompatibility:
    """Existing callers passing int repeat values must behave identically."""

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        p = tmp_path / "cron"
        monkeypatch.setattr("cron.jobs.CRON_DIR", p)
        monkeypatch.setattr("cron.jobs.JOBS_FILE", p / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", p / "output")

    def test_create_repeat_1_sets_once(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 1h", repeat=1)
        assert job["repeat"]["times"] == 1

    def test_create_repeat_3_preserved(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 1h", repeat=3)
        assert job["repeat"]["times"] == 3

    def test_create_repeat_none_means_infinite(self, tmp_path, monkeypatch):
        job = create_job(prompt="test", schedule="every 1h", repeat=None)
        assert job["repeat"]["times"] is None

    def test_update_repeat_int_change(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(action="create", prompt="test", schedule="every 1h", repeat=3)
        )
        assert created["success"] is True
        job_id = created["job_id"]

        updated = json.loads(
            cronjob(action="update", job_id=job_id, repeat=10)
        )
        assert updated["success"] is True
        assert updated["job"]["repeat"] == "10 times"
        jobs = load_jobs()
        assert jobs[0]["repeat"]["times"] == 10
