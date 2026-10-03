"""Parametrized unit tests for google_api._normalize_event_time."""

import sys
from pathlib import Path

import pytest

# Make the google-workspace scripts directory importable without installing the
# package. _hermes_home is a sibling module that google_api imports at load time;
# stub it out before the import so the test works without a real hermes home.
_SCRIPTS = Path(__file__).resolve().parents[1] / "skills/productivity/google-workspace/scripts"
sys.path.insert(0, str(_SCRIPTS))

# Stub _hermes_home so google_api can be imported without a live Hermes install.
import types

_stub = types.ModuleType("_hermes_home")
_stub.get_hermes_home = lambda: Path("/tmp/fake-hermes")  # type: ignore[attr-defined]
sys.modules["_hermes_home"] = _stub

from google_api import _normalize_event_time  # noqa: E402


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestNormalizeEventTimeAllDay:
    """All-day events have only a 'date' key and no 'dateTime'."""

    def test_returns_raw_date_string(self):
        dt_obj = {"date": "2026-07-21"}
        assert _normalize_event_time(dt_obj, user_tz=None) == "2026-07-21"

    def test_returns_raw_date_string_even_when_user_tz_provided(self):
        dt_obj = {"date": "2026-07-21"}
        assert _normalize_event_time(dt_obj, user_tz="America/New_York") == "2026-07-21"


class TestNormalizeEventTimeTimed:
    """Timed events have a 'dateTime' key."""

    def test_returns_raw_when_no_user_tz(self):
        dt_obj = {"dateTime": "2026-07-21T15:00:00+02:00", "timeZone": "Europe/Prague"}
        result = _normalize_event_time(dt_obj, user_tz=None)
        assert result == "2026-07-21T15:00:00+02:00"

    def test_converts_to_user_tz(self):
        # 15:00 Prague (CEST = UTC+2) → 09:00 New York (EDT = UTC-4)
        dt_obj = {"dateTime": "2026-07-21T15:00:00+02:00", "timeZone": "Europe/Prague"}
        result = _normalize_event_time(dt_obj, user_tz="America/New_York")
        assert result == "2026-07-21 09:00 EDT"

    def test_z_suffix_normalized_to_utc_offset(self):
        # UTC event expressed with Z suffix must be parsed as +00:00 and converted.
        dt_obj = {"dateTime": "2026-07-21T13:00:00Z"}
        result = _normalize_event_time(dt_obj, user_tz="Europe/London")
        # BST = UTC+1 in July
        assert result == "2026-07-21 14:00 BST"

    def test_utc_event_no_tz_conversion_returns_raw(self):
        dt_obj = {"dateTime": "2026-07-21T13:00:00Z"}
        result = _normalize_event_time(dt_obj, user_tz=None)
        assert result == "2026-07-21T13:00:00Z"

    def test_offset_already_present_no_user_tz(self):
        dt_obj = {"dateTime": "2026-07-21T09:00:00-05:00"}
        result = _normalize_event_time(dt_obj, user_tz=None)
        assert result == "2026-07-21T09:00:00-05:00"


class TestNormalizeEventTimeErrorHandling:
    """Malformed or unconvertible input must fall back gracefully."""

    def test_malformed_datetime_falls_back_to_raw_with_stderr_log(self, capsys):
        dt_obj = {"dateTime": "not-a-date"}
        result = _normalize_event_time(dt_obj, user_tz="America/New_York")
        assert result == "not-a-date"
        captured = capsys.readouterr()
        assert "[timezone] parse error:" in captured.err

    def test_invalid_user_tz_falls_back_to_raw_with_warning(self, capsys):
        dt_obj = {"dateTime": "2026-07-21T15:00:00+02:00"}
        result = _normalize_event_time(dt_obj, user_tz="Not/AReal_Timezone")
        assert result == "2026-07-21T15:00:00+02:00"
        captured = capsys.readouterr()
        assert "[calendar] Warning:" in captured.err

    def test_empty_dt_obj_returns_empty_string(self):
        assert _normalize_event_time({}, user_tz=None) == ""
        assert _normalize_event_time({}, user_tz="UTC") == ""

    def test_empty_datetime_value_returns_empty_string(self):
        dt_obj = {"dateTime": ""}
        assert _normalize_event_time(dt_obj, user_tz=None) == ""

    def test_both_date_and_datetime_prefers_datetime(self):
        # When both keys exist, the timed path is taken (dateTime takes priority).
        dt_obj = {"date": "2026-07-21", "dateTime": "2026-07-21T15:00:00+02:00"}
        result = _normalize_event_time(dt_obj, user_tz=None)
        assert result == "2026-07-21T15:00:00+02:00"
