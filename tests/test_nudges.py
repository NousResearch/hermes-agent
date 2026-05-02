"""Tests for nudge functionality."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_hermes_home(tmp_path):
    """Provide a temporary HERMES_HOME directory."""
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()
    
    with patch("hermes_constants.get_hermes_home") as mock_home:
        mock_home.return_value = hermes_dir
        import nudges
        nudges.HERMES_DIR = hermes_dir
        nudges.NUDGES_FILE = hermes_dir / "nudges.json"
        yield hermes_dir


class TestParseSchedule:
    """Test schedule parsing functionality."""
    
    def test_parse_one_time_relative_minutes(self, temp_hermes_home):
        from nudges import _parse_schedule, _hermes_now
        result = _parse_schedule("5m")
        assert result is not None
        assert result.is_recurring is False
        assert result.interval_seconds is None
        expected = _hermes_now() + timedelta(minutes=5)
        assert abs((result.fire_at - expected).total_seconds()) < 1
    
    def test_parse_one_time_relative_seconds(self, temp_hermes_home):
        from nudges import _parse_schedule
        result = _parse_schedule("30s")
        assert result is not None
        assert result.is_recurring is False
    
    def test_parse_one_time_relative_hours(self, temp_hermes_home):
        from nudges import _parse_schedule
        result = _parse_schedule("2h")
        assert result is not None
        assert result.is_recurring is False
    
    def test_parse_one_time_relative_days(self, temp_hermes_home):
        from nudges import _parse_schedule
        result = _parse_schedule("1d")
        assert result is not None
        assert result.is_recurring is False
    
    def test_parse_one_time_absolute_iso(self, temp_hermes_home):
        from nudges import _parse_schedule
        result = _parse_schedule("2025-06-02T14:30:00")
        assert result is not None
        assert result.is_recurring is False
        assert result.fire_at == datetime(2025, 6, 2, 14, 30, 0)
    
    def test_parse_recurring_minutes(self, temp_hermes_home):
        from nudges import _parse_schedule, _hermes_now
        result = _parse_schedule("every 5m")
        assert result is not None
        assert result.is_recurring is True
        assert result.interval_seconds == 300
        assert result.raw_schedule == "every 5m"
        expected = _hermes_now() + timedelta(minutes=5)
        assert abs((result.fire_at - expected).total_seconds()) < 1
    
    def test_parse_recurring_seconds(self, temp_hermes_home):
        from nudges import _parse_schedule
        result = _parse_schedule("every 30s")
        assert result is not None
        assert result.is_recurring is True
        assert result.interval_seconds == 30
    
    def test_parse_recurring_hours(self, temp_hermes_home):
        from nudges import _parse_schedule
        result = _parse_schedule("every 2h")
        assert result is not None
        assert result.is_recurring is True
        assert result.interval_seconds == 7200
    
    def test_parse_recurring_days(self, temp_hermes_home):
        from nudges import _parse_schedule
        result = _parse_schedule("every 1d")
        assert result is not None
        assert result.is_recurring is True
        assert result.interval_seconds == 86400
    
    def test_parse_invalid_schedule(self, temp_hermes_home):
        from nudges import _parse_schedule
        result = _parse_schedule("invalid schedule")
        assert result is None


class TestCreateNudge:
    """Test nudge creation."""
    
    def test_create_one_time_nudge(self, temp_hermes_home):
        from nudges import create_nudge, load_nudges
        nudge = create_nudge(
            session_id="session_123",
            session_key="key_123",
            schedule="5m",
            context="Test context",
            name="test_nudge"
        )
        assert nudge is not None
        assert nudge["session_id"] == "session_123"
        assert nudge["session_key"] == "key_123"
        assert nudge["context"] == "Test context"
        assert nudge["name"] == "test_nudge"
        assert nudge["is_recurring"] is False
        assert nudge["interval_seconds"] is None
        assert nudge["fired"] is False
        nudges = load_nudges()
        assert len(nudges) == 1
    
    def test_create_recurring_nudge(self, temp_hermes_home):
        from nudges import create_nudge
        nudge = create_nudge(
            session_id="session_123",
            session_key="key_123",
            schedule="every 5m",
            context="Test context"
        )
        assert nudge is not None
        assert nudge["is_recurring"] is True
        assert nudge["interval_seconds"] == 300
        assert nudge["schedule"] == "every 5m"
    
    def test_create_nudge_invalid_schedule(self, temp_hermes_home):
        from nudges import create_nudge
        nudge = create_nudge(
            session_id="session_123",
            session_key="key_123",
            schedule="invalid"
        )
        assert nudge is None
    
    def test_create_nudge_default_name(self, temp_hermes_home):
        from nudges import create_nudge
        nudge = create_nudge(
            session_id="session_123",
            session_key="key_123",
            schedule="5m"
        )
        assert nudge is not None
        assert nudge["name"].startswith("nudge_")


class TestGetDueNudges:
    """Test getting due nudges."""
    
    def test_get_due_nudges_empty(self, temp_hermes_home):
        from nudges import get_due_nudges
        due = get_due_nudges()
        assert due == []
    
    def test_get_due_nudges_one_due(self, temp_hermes_home):
        from nudges import create_nudge, get_due_nudges, _hermes_now
        create_nudge(session_id="session_123", session_key="key_123", schedule="1m", context="Test")
        future = _hermes_now() + timedelta(minutes=2)
        due = get_due_nudges(now=future)
        assert len(due) == 1
        assert due[0]["context"] == "Test"
    
    def test_get_due_nudges_not_yet_due(self, temp_hermes_home):
        from nudges import create_nudge, get_due_nudges, _hermes_now
        create_nudge(session_id="session_123", session_key="key_123", schedule="1h", context="Test")
        due = get_due_nudges(now=_hermes_now())
        assert len(due) == 0
    
    def test_get_due_nudges_already_fired(self, temp_hermes_home):
        from nudges import create_nudge, get_due_nudges, mark_nudge_fired
        nudge = create_nudge(session_id="session_123", session_key="key_123", schedule="1m", context="Test")
        mark_nudge_fired(nudge["id"])
        due = get_due_nudges()
        assert len(due) == 0


class TestFireNudge:
    """Test firing nudges."""
    
    def test_fire_one_time_nudge(self, temp_hermes_home):
        from nudges import create_nudge, fire_nudge, get_nudge
        nudge = create_nudge(session_id="session_123", session_key="key_123", schedule="5m", context="Test")
        result = fire_nudge(nudge)
        assert result is True
        fired_nudge = get_nudge(nudge["id"])
        assert fired_nudge["fired"] is True
        assert fired_nudge["fired_at"] is not None
    
    def test_fire_recurring_nudge_reschedules(self, temp_hermes_home):
        from nudges import create_nudge, fire_nudge, get_nudge
        nudge = create_nudge(session_id="session_123", session_key="key_123", schedule="every 5m", context="Test")
        original_fire_at = nudge["fire_at"]
        result = fire_nudge(nudge)
        assert result is True
        rescheduled = get_nudge(nudge["id"])
        assert rescheduled["fired"] is False
        assert rescheduled["fired_at"] is not None
        assert rescheduled["fire_count"] == 1
        new_fire_at = datetime.fromisoformat(rescheduled["fire_at"])
        assert new_fire_at > datetime.fromisoformat(original_fire_at)
    
    def test_fire_recurring_nudge_multiple_times(self, temp_hermes_home):
        from nudges import create_nudge, fire_nudge, get_nudge
        nudge = create_nudge(session_id="session_123", session_key="key_123", schedule="every 5m", context="Test")
        fire_nudge(nudge)
        fire_nudge(get_nudge(nudge["id"]))
        fire_nudge(get_nudge(nudge["id"]))
        final = get_nudge(nudge["id"])
        assert final["fire_count"] == 3


class TestDeleteNudge:
    """Test deleting nudges."""
    
    def test_delete_existing_nudge(self, temp_hermes_home):
        from nudges import create_nudge, delete_nudge, get_nudge
        nudge = create_nudge(session_id="session_123", session_key="key_123", schedule="5m")
        result = delete_nudge(nudge["id"])
        assert result is True
        assert get_nudge(nudge["id"]) is None
    
    def test_delete_nonexistent_nudge(self, temp_hermes_home):
        from nudges import delete_nudge
        result = delete_nudge("nonexistent_id")
        assert result is False
    
    def test_delete_nudges_for_session(self, temp_hermes_home):
        from nudges import create_nudge, delete_nudges_for_session, list_nudges
        create_nudge(session_id="session_1", session_key="key_1", schedule="5m")
        create_nudge(session_id="session_1", session_key="key_1", schedule="10m")
        create_nudge(session_id="session_2", session_key="key_2", schedule="5m")
        count = delete_nudges_for_session("session_1")
        assert count == 2
        remaining = list_nudges()
        assert len(remaining) == 1
        assert remaining[0]["session_id"] == "session_2"


class TestListNudges:
    """Test listing nudges."""
    
    def test_list_all_nudges(self, temp_hermes_home):
        from nudges import create_nudge, list_nudges
        create_nudge(session_id="s1", session_key="k1", schedule="5m")
        create_nudge(session_id="s2", session_key="k2", schedule="10m")
        nudges = list_nudges()
        assert len(nudges) == 2
    
    def test_list_by_session(self, temp_hermes_home):
        from nudges import create_nudge, list_nudges
        create_nudge(session_id="session_1", session_key="k1", schedule="5m")
        create_nudge(session_id="session_2", session_key="k2", schedule="10m")
        nudges = list_nudges(session_id="session_1")
        assert len(nudges) == 1
        assert nudges[0]["session_id"] == "session_1"
    
    def test_list_exclude_fired(self, temp_hermes_home):
        from nudges import create_nudge, list_nudges, mark_nudge_fired
        nudge1 = create_nudge(session_id="s1", session_key="k1", schedule="1m")
        nudge2 = create_nudge(session_id="s1", session_key="k1", schedule="2m")
        mark_nudge_fired(nudge1["id"])
        nudges = list_nudges(include_fired=False)
        assert len(nudges) == 1
        assert nudges[0]["id"] == nudge2["id"]


class TestCleanup:
    """Test cleanup of old nudges."""
    
    def test_cleanup_old_fired_nudges(self, temp_hermes_home):
        from nudges import create_nudge, cleanup_old_nudges, fire_nudge, _hermes_now
        import nudges
        nudge = create_nudge(session_id="s1", session_key="k1", schedule="1m")
        fire_nudge(nudge)
        nudges_list = nudges.load_nudges()
        for n in nudges_list:
            if n["id"] == nudge["id"]:
                past = _hermes_now() - timedelta(hours=25)
                n["fired_at"] = past.isoformat()
        nudges.save_nudges(nudges_list)
        removed = cleanup_old_nudges(max_age_hours=24)
        assert removed == 1
        remaining = nudges.load_nudges()
        assert len(remaining) == 0
    
    def test_cleanup_does_not_remove_recent(self, temp_hermes_home):
        from nudges import create_nudge, cleanup_old_nudges, fire_nudge
        nudge = create_nudge(session_id="s1", session_key="k1", schedule="1m")
        fire_nudge(nudge)
        removed = cleanup_old_nudges(max_age_hours=24)
        assert removed == 0


class TestNudgePersistence:
    """Test that nudges persist across operations."""
    
    def test_nudges_persist_to_disk(self, temp_hermes_home):
        from nudges import create_nudge, load_nudges
        nudge = create_nudge(session_id="s1", session_key="k1", schedule="5m", context="Test persistence")
        fresh_nudges = load_nudges()
        assert len(fresh_nudges) == 1
        assert fresh_nudges[0]["context"] == "Test persistence"
        assert fresh_nudges[0]["id"] == nudge["id"]
    
    def test_nudges_file_format(self, temp_hermes_home):
        from nudges import create_nudge, NUDGES_FILE
        create_nudge(session_id="s1", session_key="k1", schedule="every 5m")
        with open(NUDGES_FILE) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "id" in data[0]
        assert "session_id" in data[0]
        assert "is_recurring" in data[0]
