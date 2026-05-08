import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from agent import restart_awareness as ra


class TestComputeStaleness:
    def test_fresh_data(self):
        now = datetime.now(timezone.utc).isoformat()
        is_stale, age_text = ra._compute_staleness(now)
        assert is_stale is False
        assert "m old" in age_text or age_text == "0m old"

    def test_stale_data(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        is_stale, age_text = ra._compute_staleness(old)
        assert is_stale is True
        assert "h" in age_text

    def test_missing_timestamp(self):
        is_stale, age_text = ra._compute_staleness(None)
        assert is_stale is False
        assert age_text == ""

    def test_malformed_timestamp(self):
        is_stale, age_text = ra._compute_staleness("not-a-date")
        assert is_stale is False
        assert age_text == ""


class TestBuildHandoff:
    def test_simple_fresh(self):
        activity = {
            "current_task": "Test task",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "simple",
        }
        handoff = ra.build_handoff(activity)
        assert "[Restart handoff \u2014 FRESH]" in handoff
        assert "Back after gateway restart." in handoff
        assert "Task:" not in handoff

    def test_simple_stale(self):
        activity = {
            "current_task": "Test task",
            "updated_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "mode": "simple",
        }
        handoff = ra.build_handoff(activity)
        assert "[Restart handoff \u2014 STALE" in handoff
        assert "old" in handoff

    def test_verbose_fresh(self):
        activity = {
            "current_task": "Test task",
            "files_modified": ["a.py", "b.py"],
            "last_action": "Did something",
            "next_expected_step": "Do next thing",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "verbose",
        }
        handoff = ra.build_handoff(activity)
        assert "[Restart handoff \u2014 FRESH]" in handoff
        assert "Task: Test task" in handoff
        assert "Files touched: a.py, b.py" in handoff
        assert "Last action: Did something" in handoff
        assert "Next step: Do next thing" in handoff

    def test_verbose_stale(self):
        activity = {
            "current_task": "Test task",
            "updated_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "mode": "verbose",
        }
        handoff = ra.build_handoff(activity)
        assert "[Restart handoff \u2014 STALE" in handoff
        assert "Do NOT auto-execute" in handoff
        assert "Task: Test task" in handoff

    def test_verbose_no_optional_fields(self):
        activity = {
            "current_task": "Test task",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "verbose",
        }
        handoff = ra.build_handoff(activity)
        assert "Task: Test task" in handoff
        assert "Files touched" not in handoff
        assert "Last action" not in handoff
        assert "Next step" not in handoff


class TestReadWriteClear:
    @patch.object(ra, "_activity_path")
    def test_round_trip(self, mock_path):
        tmp = MagicMock()
        mock_path.return_value = tmp
        tmp.parent = MagicMock()
        tmp.exists.return_value = True

        ra.update_activity(
            current_task="Round trip",
            files_modified=["x.py"],
            last_action="wrote",
            next_expected_step="read",
            mode="verbose",
        )
        written = json.loads(tmp.write_text.call_args[0][0])
        assert written["current_task"] == "Round trip"
        assert written["files_modified"] == ["x.py"]
        assert written["mode"] == "verbose"
        assert "updated_at" in written

        tmp.read_text.return_value = json.dumps(written)
        result = ra.read_activity()
        assert result["current_task"] == "Round trip"

        ra.clear_activity()
        assert tmp.unlink.called

    @patch.object(ra, "_activity_path")
    def test_read_missing(self, mock_path):
        tmp = MagicMock()
        mock_path.return_value = tmp
        tmp.exists.return_value = False
        assert ra.read_activity() is None

    @patch.object(ra, "_activity_path")
    def test_read_corrupt(self, mock_path):
        tmp = MagicMock()
        mock_path.return_value = tmp
        tmp.exists.return_value = True
        tmp.read_text.return_value = "not json"
        assert ra.read_activity() is None
