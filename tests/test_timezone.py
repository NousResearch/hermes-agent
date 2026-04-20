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
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import hermes_time


def _reset_hermes_time_cache():
    """Reset the hermes_time module cache (replacement for removed reset_cache)."""
    hermes_time._cached_tz = None
    hermes_time._cached_tz_name = None
    hermes_time._cache_resolved = False


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
        assert offset_hours in (-5, -4)

    def test_invalid_timezone_falls_back(self, caplog):
        """Invalid timezone logs warning and falls back to server-local."""
        os.environ["HERMES_TIMEZONE"] = "Mars/Olympus_Mons"
        with caplog.at_level(logging.WARNING, logger="hermes_time"):
            result = hermes_time.now()
        assert result.tzinfo is not None  # Still tz-aware (server-local)
        assert "Invalid timezone" in caplog.text
        assert "Mars/Olympus_Mons" in caplog.text

    def test_empty_timezone_uses_local(self):
        """No timezone configured → server-local time (still tz-aware)."""
        os.environ.pop("HERMES_TIMEZONE", None)
        result = hermes_time.now()
        assert result.tzinfo is not None

    def test_format_unchanged(self):
        """Timestamp formatting matches original strftime pattern."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        result = hermes_time.now()
        formatted = result.strftime("%A, %B %d, %Y %I:%M %p")
        # Should produce something like "Monday, March 03, 2026 05:30 PM"
        assert len(formatted) > 10
        # No timezone abbreviation in the format (matching original behavior)
        assert "+" not in formatted

    def test_cache_invalidation(self):
        """Changing env var + reset_cache picks up new timezone."""
        os.environ["HERMES_TIMEZONE"] = "UTC"
        _reset_hermes_time_cache()
        r1 = hermes_time.now()
        assert r1.utcoffset() == timedelta(0)

        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        _reset_hermes_time_cache()
        r2 = hermes_time.now()
        assert r2.utcoffset() == timedelta(hours=5, minutes=30)


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

    def test_returns_none_for_empty(self):
        os.environ.pop("HERMES_TIMEZONE", None)
        tz = hermes_time.get_timezone()
        assert tz is None

    def test_returns_none_for_invalid(self):
        os.environ["HERMES_TIMEZONE"] = "Not/A/Timezone"
        tz = hermes_time.get_timezone()
        assert tz is None



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

    def test_parse_schedule_duration_uses_tz_aware_now(self):
        """parse_schedule('30m') should produce a tz-aware run_at."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        from cron.jobs import parse_schedule
        result = parse_schedule("30m")
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

    def test_get_due_jobs_handles_naive_timestamps(self, tmp_path, monkeypatch):
        """Backward compat: naive timestamps from before tz support don't crash."""
        import cron.jobs as jobs_module
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")

        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        _reset_hermes_time_cache()

        # Create a job with a NAIVE past timestamp (simulating pre-tz data)
        from cron.jobs import create_job, load_jobs, save_jobs, get_due_jobs
        job = create_job(prompt="Test job", schedule="every 1h")
        jobs = load_jobs()
        # Force a naive (no timezone) past timestamp
        naive_past = (datetime.now() - timedelta(seconds=30)).isoformat()
        jobs[0]["next_run_at"] = naive_past
        save_jobs(jobs)

        # Should not crash — _ensure_aware handles the naive timestamp
        due = get_due_jobs()
        assert len(due) == 1

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

    def test_ensure_aware_normalizes_aware_to_hermes_tz(self):
        """Already-aware datetimes should be normalized to Hermes tz."""
        from cron.jobs import _ensure_aware

        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        _reset_hermes_time_cache()

        # Create an aware datetime in UTC
        utc_dt = datetime(2026, 3, 11, 15, 0, 0, tzinfo=timezone.utc)
        result = _ensure_aware(utc_dt)

        # Must be in Hermes tz (Kolkata) but same absolute instant
        kolkata = ZoneInfo("Asia/Kolkata")
        assert result.utctimetuple()[:5] == (2026, 3, 11, 15, 0)
        expected_local = utc_dt.astimezone(kolkata)
        assert result == expected_local

    def test_ensure_aware_due_job_not_skipped_when_system_ahead(self, tmp_path, monkeypatch):
        """Reproduce the actual bug: system tz ahead of Hermes tz caused
        overdue jobs to appear as not-yet-due.

        Scenario: system is Asia/Kolkata (UTC+5:30), Hermes is UTC.
        A naive timestamp from 5 minutes ago (local time) should still
        be recognized as due after conversion.
        """
        import cron.jobs as jobs_module
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")

        os.environ["HERMES_TIMEZONE"] = "UTC"
        _reset_hermes_time_cache()

        from cron.jobs import create_job, load_jobs, save_jobs, get_due_jobs

        job = create_job(prompt="Bug repro", schedule="every 1h")
        jobs = load_jobs()

        # Simulate a naive timestamp that was written by datetime.now() on a
        # system running in UTC+5:30 — 5 minutes in the past (local time)
        naive_past = (datetime.now() - timedelta(seconds=30)).isoformat()
        jobs[0]["next_run_at"] = naive_past
        save_jobs(jobs)

        # Must be recognized as due regardless of tz mismatch
        due = get_due_jobs()
        assert len(due) == 1, (
            "Overdue job was skipped — _ensure_aware likely shifted absolute time"
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
