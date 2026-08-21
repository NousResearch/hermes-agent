"""Regression tests for cron repeat string normalization (t_cd6a18cc / t_681ebac7).

Covers the defect where ``repeat='forever'`` passed to
``cronjob(action=create)`` or ``create_job()`` raised a raw TypeError::

    '<=' not supported between instances of 'str' and 'int'

This happened because two code paths compared the ``repeat`` parameter with
an integer using ``<=`` without first normalising string inputs::

    - cron/jobs.py::create_job  (~line 1315): ``if repeat <= 0:``
    - tools/cronjob_tools.py::cronjob update path (~line 1003):
      ``normalized_repeat = None if repeat <= 0 else repeat``

Expected normalisation contract (what both sites must implement)::

    None         -> None   (infinite)
    <int <=0>    -> None   (zero/negative treated as infinite)
    "forever"    -> None   (synonym: forever, infinite, inf, "")
    >0 int       -> int    (finite recurrence count preserved)
    numeric str  -> int    ("3" -> 3)
    garbage str  -> ValueError  (clean error message, NOT TypeError)

Tests are written to the RED-GREEN paradigm: on origin/main the unit
tests for ``_normalize_repeat_value`` will FAIL RED because the helper
hasn't landed yet; integration tests that call ``cronjob()`` will also
FAIL because create/update will raise TypeError — once the builder's
implementation lands, every test should go GREEN.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


# ===========================================================================
# Direct unit tests for _normalize_repeat_value
# ===========================================================================

class TestNormalizeRepeatValueDirect:
    """Unit tests for the shared normalisation helper.

    After the fix lands these exercises ``cron.jobs._normalize_repeat_value``
    directly. Before landing they are written to document the intended contract
    and will FAIL RED — confirming the bug still exists.
    """

    def test_none_passes_through(self):
        """None maps to None (infinite recurrence)."""
        from cron.jobs import _normalize_repeat_value
        assert _normalize_repeat_value(None) is None

    def test_forever_becomes_none(self):
        """'forever' normalises to None."""
        from cron.jobs import _normalize_repeat_value
        assert _normalize_repeat_value("forever") is None

    @pytest.mark.parametrize(
        "input_val",
        [
            "FOREVER",
            "Forever",
            "  FOREVER  ",
            "infinite",
            "Infinite",
            "INF",
            "inf",
            "",
            "  ",
        ],
    )
    def test_all_synonyms_map_to_none(self, input_val):
        """All recognised synonyms (case-insensitive, stripped) yield None."""
        from cron.jobs import _normalize_repeat_value
        result = _normalize_repeat_value(input_val)
        assert result is None, f"{input_val!r} did not map to None"

    def test_zero_int_becomes_none(self):
        """Integer zero treats as infinite (same old semantics)."""
        from cron.jobs import _normalize_repeat_value
        assert _normalize_repeat_value(0) is None

    def test_negative_int_becomes_none(self):
        """Negative ints treat as infinite."""
        from cron.jobs import _normalize_repeat_value
        assert _normalize_repeat_value(-1) is None
        assert _normalize_repeat_value(-999) is None

    def test_positive_int_preserved(self):
        """Positive integers pass through unchanged."""
        from cron.jobs import _normalize_repeat_value
        assert _normalize_repeat_value(1) == 1
        assert _normalize_repeat_value(42) == 42
        assert _normalize_repeat_value(9999) == 9999

    def test_numeric_string_converted_to_int(self):
        """Numeric strings get converted to int."""
        from cron.jobs import _normalize_repeat_value
        assert _normalize_repeat_value("3") == 3
        assert _normalize_repeat_value("100") == 100
        assert _normalize_repeat_value("-5") is None  # negative parsed out

    def test_invalid_string_raises_valueerror_not_typeerror(self):
        """Bogus strings produce ValueError with a helpful message."""
        from cron.jobs import _normalize_repeat_value
        with pytest.raises(ValueError, match="repeat must be an integer"):
            _normalize_repeat_value("bogus")
        with pytest.raises(ValueError, match="repeat must be an integer"):
            _normalize_repeat_value("not-a-number")

    def test_float_numeric_coercion(self):
        """Floats get converted to int (e.g. 3.14 -> 3)."""
        from cron.jobs import _normalize_repeat_value
        result = _normalize_repeat_value(3.14)
        assert result == 3  # numeric coercion accepted


# ===========================================================================
# Integration tests: create_job data-layer with string repeat values
# ===========================================================================

class TestCreateJobWithStringRepeat:
    """Data-layer: create_job must accept string repeat without crashing."""

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        p = tmp_path / "cron"
        monkeypatch.setattr("cron.jobs.CRON_DIR", p)
        monkeypatch.setattr("cron.jobs.JOBS_FILE", p / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", p / "output")

    def test_create_with_repeat_forever_string_no_crash(self, tmp_path, monkeypatch):
        """The original crash vector: repeat='forever' must not raise TypeError."""
        from cron.jobs import create_job

        job = create_job(prompt="test", schedule="every 1h", repeat="forever")
        assert job["repeat"]["times"] is None
        assert job["state"] == "scheduled"

    def test_create_with_repeat_inf(self, tmp_path, monkeypatch):
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 2h", repeat="inf")
        assert job["repeat"]["times"] is None

    def test_create_with_repeat_infinite(self, tmp_path, monkeypatch):
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 30m", repeat="infinite")
        assert job["repeat"]["times"] is None

    def test_create_with_repeat_zero_string(self, tmp_path, monkeypatch):
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 1h", repeat="0")
        assert job["repeat"]["times"] is None

    def test_create_with_repeat_integer_still_works(self, tmp_path, monkeypatch):
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 1h", repeat=5)
        assert job["repeat"]["times"] == 5

    def test_create_with_no_repeat_defaults_to_none(self, tmp_path, monkeypatch):
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 1h")
        assert job["repeat"]["times"] is None

    def test_create_with_numeric_string_repeat(self, tmp_path, monkeypatch):
        """String '7' should become int 7 in the stored record."""
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 1h", repeat="7")
        assert job["repeat"]["times"] == 7

    def test_create_with_invalid_repeat_raises_valueerror(self, tmp_path, monkeypatch):
        """Invalid strings should raise ValueError, not TypeError."""
        from cron.jobs import create_job
        with pytest.raises(ValueError, match="repeat must be an integer"):
            create_job(prompt="test", schedule="every 1h", repeat="garbagevalue")


# ===========================================================================
# Integration tests: tool cronjob(action=create|update) with string repeat
# ===========================================================================

class TestCronjobToolWithStringRepeat:
    """API-layer: cronjob() tool must handle string repeat cleanly.

    Both action='create' and action='update' touch the repeat field and must
    never expose a raw TypeError to the caller.
    """

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        p = tmp_path / "cron"
        monkeypatch.setattr("cron.jobs.CRON_DIR", p)
        monkeypatch.setattr("cron.jobs.JOBS_FILE", p / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", p / "output")

    def test_tool_create_repeat_forever_succeeds(self, tmp_path, monkeypatch):
        """The original crash site via the tool's create path."""
        from tools.cronjob_tools import cronjob
        from cron.jobs import load_jobs

        result = json.loads(cronjob(
            action="create",
            prompt="Check server status",
            schedule="every 1h",
            name="Server Check",
            repeat="forever",
        ))
        assert result["success"] is True, f"Unexpected error: {result}"
        # Formatted display shows "forever"
        assert result["job"]["repeat"] == "forever"
        # Stored record confirms times=None (infinite)
        jobs = load_jobs()
        assert len(jobs) == 1
        assert jobs[0]["repeat"]["times"] is None

    def test_tool_create_repeat_inf(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create", prompt="x", schedule="every 2h", repeat="inf"
        ))
        assert result["success"] is True
        assert result["job"]["repeat"] == "forever"

    def test_tool_update_repeat_forever_on_existing_job(self, tmp_path, monkeypatch):
        """Original crash: update path with repeat='forever'.

        This was the second crash site at line ~1003 in cronjob_tools.py.
        """
        from tools.cronjob_tools import cronjob
        from cron.jobs import load_jobs

        # Create with int repeat
        created = json.loads(cronjob(
            action="create", prompt="Baseline", schedule="every 1h", repeat=5
        ))
        assert created["success"] is True
        job_id = created["job_id"]
        assert created["job"]["repeat"] == "5 times"

        # Update to 'forever' — this was the crash site
        updated = json.loads(cronjob(
            action="update", job_id=job_id, repeat="forever"
        ))
        assert updated["success"] is True
        assert updated["job"]["repeat"] == "forever"
        jobs = load_jobs()
        assert jobs[0]["repeat"]["times"] is None

    def test_tool_update_repeat_to_zero_string(self, tmp_path, monkeypatch):
        """Update: change finite repeat to '0' (should become infinite)."""
        from tools.cronjob_tools import cronjob

        created = json.loads(cronjob(
            action="create", prompt="test", schedule="every 1h", repeat=10
        ))
        assert created["success"] is True
        job_id = created["job_id"]

        updated = json.loads(cronjob(
            action="update", job_id=job_id, repeat="0"
        ))
        assert updated["success"] is True
        assert updated["job"]["repeat"] == "forever"

    def test_tool_update_repeat_numeric_string(self, tmp_path, monkeypatch):
        """Update: change repeat from one int to another via numeric string."""
        from tools.cronjob_tools import cronjob

        created = json.loads(cronjob(
            action="create", prompt="test", schedule="every 1h", repeat=3
        ))
        assert created["success"] is True
        job_id = created["job_id"]

        updated = json.loads(cronjob(
            action="update", job_id=job_id, repeat="50"
        ))
        assert updated["success"] is True
        assert updated["job"]["repeat"] == "50 times"

    def test_tool_create_invalid_repeat_returns_error_not_crash(self, tmp_path, monkeypatch):
        """Invalid repeat must return a JSON error object with success=False, not crash."""
        from tools.cronjob_tools import cronjob
        from cron.jobs import load_jobs

        result = json.loads(cronjob(
            action="create", prompt="test", schedule="every 1h", repeat="invalidvalue"
        ))
        assert result["success"] is False
        assert "repeat" in result.get("error", "").lower()
        # No job persisted
        jobs = load_jobs()
        assert len(jobs) == 0

    def test_tool_update_invalid_repeat_returns_error_not_crash(self, tmp_path, monkeypatch):
        """Update with invalid repeat must also return a clean error."""
        from tools.cronjob_tools import cronjob
        from cron.jobs import get_job

        created = json.loads(cronjob(
            action="create", prompt="test", schedule="every 1h", repeat=3
        ))
        assert created["success"] is True
        job_id = created["job_id"]

        updated = json.loads(cronjob(
            action="update", job_id=job_id, repeat="badrepeat"
        ))
        assert updated["success"] is False
        # Original job must be unchanged
        stored = get_job(job_id)
        assert stored is not None
        assert stored["repeat"]["times"] == 3


# ===========================================================================
# Backward compatibility: existing int-repeat behaviour unchanged
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
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 1h", repeat=1)
        assert job["repeat"]["times"] == 1

    def test_create_repeat_3_preserved(self, tmp_path, monkeypatch):
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 1h", repeat=3)
        assert job["repeat"]["times"] == 3

    def test_create_repeat_none_means_infinite(self, tmp_path, monkeypatch):
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 1h", repeat=None)
        assert job["repeat"]["times"] is None

    def test_create_interval_without_repeat_is_finite(self, tmp_path, monkeypatch):
        """Interval schedules with no explicit repeat default to None (infinite).
        One-shot schedules auto-set repeat=1 via the parsed_schedule check.
        """
        from cron.jobs import create_job
        job = create_job(prompt="test", schedule="every 1h")
        assert job["repeat"]["times"] is None

    def test_update_repeat_int_change(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import cronjob
        from cron.jobs import load_jobs

        created = json.loads(cronjob(
            action="create", prompt="test", schedule="every 1h", repeat=3
        ))
        assert created["success"] is True
        job_id = created["job_id"]

        updated = json.loads(cronjob(
            action="update", job_id=job_id, repeat=10
        ))
        assert updated["success"] is True
        assert updated["job"]["repeat"] == "10 times"
        jobs = load_jobs()
        assert jobs[0]["repeat"]["times"] == 10

    def test_tool_create_repeat_as_string_digit(self, tmp_path, monkeypatch):
        """When agents pass repeat='3' (string digit), it works like int 3."""
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create", prompt="test", schedule="every 1h", repeat="3"
        ))
        assert result["success"] is True
        assert result["job"]["repeat"] == "3 times"
