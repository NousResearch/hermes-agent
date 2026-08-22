"""
Tests for timezone support (hermes_time module + integration points).

Covers:
  - Valid timezone applies correctly
  - Invalid timezone falls back safely (no crash, warning logged)
  - execute_code child env receives TZ
  - Cron uses timezone-aware now()
  - Backward compatibility with naive timestamps
"""

import os
import logging
import sys
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import hermes_time


def _reset_hermes_time_cache():
    """Reset the hermes_time module cache (replacement for removed reset_cache)."""
    hermes_time._cached_tz = None
    hermes_time._cached_tz_name = None
    hermes_time._cache_resolved = False
    hermes_time._cache_identity = None


# =========================================================================
# hermes_time.now() — core helper
# =========================================================================

class TestHermesTimeNow:
    """Test the timezone-aware now() helper."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_valid_timezone_applies(self):
        """With a valid IANA timezone, now() returns time in that zone."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        result = hermes_time.now()
        assert result.tzinfo is not None
        # IST is UTC+5:30
        offset = result.utcoffset()
        assert offset == timedelta(hours=5, minutes=30)

    def test_utc_timezone(self):
        """UTC timezone works."""
        os.environ["HERMES_TIMEZONE"] = "UTC"
        result = hermes_time.now()
        assert result.utcoffset() == timedelta(0)

    def test_us_eastern(self):
        """US/Eastern timezone works (DST-aware zone)."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        result = hermes_time.now()
        assert result.tzinfo is not None
        # Offset is -5h or -4h depending on DST
        offset_hours = result.utcoffset().total_seconds() / 3600
        assert offset_hours in {-5, -4}






class TestGetTimezone:
    """Test get_timezone()."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_returns_zoneinfo_for_valid(self):
        os.environ["HERMES_TIMEZONE"] = "Europe/London"
        tz = hermes_time.get_timezone()
        assert isinstance(tz, ZoneInfo)
        assert str(tz) == "Europe/London"

    def test_cache_isolated_by_active_profile_config(self, tmp_path, monkeypatch):
        """Switching HERMES_HOME must not reuse another profile's timezone."""
        first_home = tmp_path / "first"
        second_home = tmp_path / "second"
        first_home.mkdir()
        second_home.mkdir()
        (first_home / "config.yaml").write_text("timezone: Asia/Tokyo\n", encoding="utf-8")
        (second_home / "config.yaml").write_text(
            "timezone: America/New_York\n", encoding="utf-8"
        )
        monkeypatch.delenv("HERMES_TIMEZONE", raising=False)

        monkeypatch.setenv("HERMES_HOME", str(first_home))
        assert str(hermes_time.get_timezone()) == "Asia/Tokyo"

        # Multiplexed profile runtime scopes switch HERMES_HOME in one process.
        monkeypatch.setenv("HERMES_HOME", str(second_home))
        assert str(hermes_time.get_timezone()) == "America/New_York"


# =========================================================================

# execute_code child env — TZ injection
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="UDS not available on Windows")
class TestCodeExecutionTZ:
    """Verify TZ env var is passed to sandboxed child process via real execute_code."""

    @pytest.fixture(autouse=True)
    def _import_execute_code(self, monkeypatch):
        """Lazy-import execute_code to avoid pulling in firecrawl at collection time."""
        # Force local backend — other tests in the same xdist worker may leak
        # TERMINAL_ENV=modal/docker which causes modal.exception.AuthError.
        monkeypatch.setenv("TERMINAL_ENV", "local")
        try:
            from tools.code_execution_tool import execute_code
            self._execute_code = execute_code
        except ImportError:
            pytest.skip("tools.code_execution_tool not importable (missing deps)")

    def teardown_method(self):
        os.environ.pop("HERMES_TIMEZONE", None)

    def _mock_handle(self, function_name, function_args, task_id=None, user_task=None):
        import json as _json
        return _json.dumps({"error": f"unexpected tool call: {function_name}"})

    def test_tz_injected_when_configured(self):
        """When HERMES_TIMEZONE is set, child process sees TZ env var.

        Verified alongside leak-prevention + empty-TZ handling in one
        subprocess call so we don't pay 3x the subprocess startup cost
        (each execute_code spawns a real Python subprocess ~3s).
        """
        import json as _json
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"

        # One subprocess, three things checked:
        #   1) TZ is injected as "Asia/Kolkata"
        #   2) HERMES_TIMEZONE itself does NOT leak into the child env
        probe = (
            'import os; '
            'print("TZ=" + os.environ.get("TZ", "NOT_SET")); '
            'print("HERMES_TIMEZONE=" + os.environ.get("HERMES_TIMEZONE", "NOT_SET"))'
        )
        with patch("model_tools.handle_function_call", side_effect=self._mock_handle):
            result = _json.loads(self._execute_code(
                code=probe,
                task_id="tz-combined-test",
                enabled_tools=[],
            ))
        assert result["status"] == "success"
        assert "TZ=Asia/Kolkata" in result["output"]
        assert "HERMES_TIMEZONE=NOT_SET" in result["output"], (
            "HERMES_TIMEZONE should not leak into child env (only TZ)"
        )

    def test_tz_not_injected_when_empty(self):
        """When HERMES_TIMEZONE is not set, child process has no TZ."""
        import json as _json
        os.environ.pop("HERMES_TIMEZONE", None)

        with patch("model_tools.handle_function_call", side_effect=self._mock_handle):
            result = _json.loads(self._execute_code(
                code='import os; print(os.environ.get("TZ", "NOT_SET"))',
                task_id="tz-test-empty",
                enabled_tools=[],
            ))
        assert result["status"] == "success"
        assert "NOT_SET" in result["output"]


# =========================================================================
# Cron timezone-aware scheduling
# =========================================================================

class TestCronTimezone:
    """Verify cron paths use timezone-aware now()."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_parse_schedule_one_shot_duration_uses_tz_aware_now(self):
        """parse_schedule('in 30m') should produce a tz-aware run_at."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        from cron.jobs import parse_schedule
        result = parse_schedule("in 30m")
        run_at = datetime.fromisoformat(result["run_at"])
        # The stored timestamp should be tz-aware
        assert run_at.tzinfo is not None

    def test_compute_next_run_tz_aware(self):
        """compute_next_run returns tz-aware timestamps."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        from cron.jobs import compute_next_run
        schedule = {"kind": "interval", "minutes": 60}
        result = compute_next_run(schedule)
        next_dt = datetime.fromisoformat(result)
        assert next_dt.tzinfo is not None


    def test_ensure_aware_naive_preserves_absolute_time(self):
        """_ensure_aware must preserve the absolute instant for naive datetimes.

        Regression: the old code used replace(tzinfo=hermes_tz) which shifted
        absolute time when system-local tz != Hermes tz.  The fix interprets
        naive values as system-local wall time, then converts.
        """
        from cron.jobs import _ensure_aware

        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        _reset_hermes_time_cache()

        # Create a naive datetime — will be interpreted as system-local time
        naive_dt = datetime(2026, 3, 11, 12, 0, 0)

        result = _ensure_aware(naive_dt)

        # The result should be in Kolkata tz
        assert result.tzinfo is not None

        # The UTC equivalent must match what we'd get by correctly interpreting
        # the naive dt as system-local time first, then converting
        system_tz = datetime.now().astimezone().tzinfo
        expected_utc = naive_dt.replace(tzinfo=system_tz).astimezone(timezone.utc)
        actual_utc = result.astimezone(timezone.utc)
        assert actual_utc == expected_utc, (
            f"Absolute time shifted: expected {expected_utc}, got {actual_utc}"
        )



    def test_get_due_jobs_naive_cross_timezone(self, tmp_path, monkeypatch):
        """Naive past timestamps must be detected as due even when Hermes tz
        is behind system local tz — the scenario that triggered #806."""
        import cron.jobs as jobs_module
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")

        # Use a Hermes timezone far behind UTC so that the numeric wall time
        # of the naive timestamp exceeds _hermes_now's wall time — this would
        # have caused a false "not due" with the old replace(tzinfo=...) approach.
        os.environ["HERMES_TIMEZONE"] = "Pacific/Midway"  # UTC-11
        _reset_hermes_time_cache()

        from cron.jobs import create_job, load_jobs, save_jobs, get_due_jobs
        create_job(prompt="Cross-tz job", schedule="every 1h")
        jobs = load_jobs()

        # Force a naive past timestamp (system-local wall time, 10 min ago)
        naive_past = (datetime.now() - timedelta(seconds=30)).isoformat()
        jobs[0]["next_run_at"] = naive_past
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 1, (
            "Naive past timestamp should be due regardless of Hermes timezone"
        )

    def test_create_job_stores_tz_aware_timestamps(self, tmp_path, monkeypatch):
        """New jobs store timezone-aware created_at and next_run_at."""
        import cron.jobs as jobs_module
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")

        os.environ["HERMES_TIMEZONE"] = "US/Eastern"
        _reset_hermes_time_cache()

        from cron.jobs import create_job
        job = create_job(prompt="TZ test", schedule="every 2h")

        created = datetime.fromisoformat(job["created_at"])
        assert created.tzinfo is not None

        next_run = datetime.fromisoformat(job["next_run_at"])
        assert next_run.tzinfo is not None
