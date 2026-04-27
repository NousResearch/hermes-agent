"""Tests for GET /api/summary/daily aggregation methods in hermes_state."""

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_state import SessionDB, TaskDB, ApprovalDB, _date_to_timestamp_range


class TestDateToTimestampRange:
    def test_converts_date_to_start_and_end(self):
        start_ts, end_ts = _date_to_timestamp_range("2026-04-23")
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        assert start_dt.year == 2026
        assert start_dt.month == 4
        assert start_dt.day == 23
        assert start_dt.hour == 0
        assert start_dt.minute == 0
        assert end_dt.day == 24
        assert end_dt.hour == 0
        assert end_ts == start_ts + 86400

    def test_different_dates_produce_different_ranges(self):
        r1 = _date_to_timestamp_range("2026-04-01")
        r2 = _date_to_timestamp_range("2026-04-23")
        assert r1[0] != r2[0]
        assert r1[1] != r2[1]


class TestSessionDBAggregations:
    @pytest.fixture
    def db(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "test.db")
        yield db
        db.close()

    def test_sessions_today_count_empty(self, db):
        assert db.sessions_today_count("2026-04-23") == 0

    def test_sessions_today_count_includes_today(self, db):
        ts_today = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        db.ensure_session("session-1", source="web", model="gpt-4")
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (ts_today, "session-1"),
            )
        assert db.sessions_today_count("2026-04-23") == 1

    def test_sessions_today_count_excludes_other_days(self, db):
        ts_yesterday = datetime(
            2026, 4, 22, 23, 59, 59, tzinfo=timezone.utc
        ).timestamp()
        db.ensure_session("session-1", source="web", model="gpt-4")
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (ts_yesterday, "session-1"),
            )
        assert db.sessions_today_count("2026-04-23") == 0

    def test_active_sessions_count_none_active(self, db):
        assert db.active_sessions_count() == 0


class TestTaskDBAggregations:
    @pytest.fixture
    def db(self, tmp_path):
        db = TaskDB(db_path=tmp_path / "test_tasks.db")
        yield db
        db.close()

    def test_count_by_status_empty(self, db):
        assert db.count_by_status() == {}

    def test_count_by_status_groups_correctly(self, db):
        db.create_task("task-1", title="One", priority="high")
        db.create_task("task-2", title="Two", priority="medium")
        db.create_task("task-3", title="Three", priority="low")
        db.update_task("task-1", {"status": "done"})
        db.update_task("task-2", {"status": "in_progress"})
        counts = db.count_by_status()
        assert counts.get("todo", 0) == 1
        assert counts.get("in_progress", 0) == 1
        assert counts.get("done", 0) == 1

    def test_completed_today_count_none(self, db):
        assert db.completed_today_count("2026-04-23") == 0

    def test_completed_today_count_filters_by_date(self, db):
        db.create_task("task-1", title="One", priority="high")
        db.update_task("task-1", {"status": "done"})
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert db.completed_today_count(today_prefix) == 1
        assert db.completed_today_count("2020-01-01") == 0


class TestApprovalDBAggregations:
    @pytest.fixture
    def db(self, tmp_path):
        db = ApprovalDB(db_path=tmp_path / "test_approvals.db")
        yield db
        db.close()

    def test_pending_count_empty(self, db):
        assert db.get_pending_count() == 0

    def test_pending_count_increments(self, db):
        db.create_approval(
            approval_id="approval-1",
            session_id="session-1",
            agent_id="agent-1",
            title="test",
            command="ls",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        assert db.get_pending_count() == 1

    def test_resolved_today_count_empty(self, db):
        assert db.resolved_today_count("approved", "2026-04-23") == 0

    def test_resolved_today_count_filters_by_status(self, db):
        now = datetime.now(timezone.utc)
        today_iso = now.isoformat()
        yesterday_iso = (now.replace(day=now.day - 1)).isoformat()
        db.create_approval(
            approval_id="approval-1",
            session_id="session-1",
            agent_id="agent-1",
            title="test",
            command="ls",
            created_at=today_iso,
        )
        db.resolve_approval(
            "approval-1", "approved", resolved_by="user", choice="session"
        )
        today_prefix = now.strftime("%Y-%m-%d")
        assert db.resolved_today_count("approved", today_prefix) == 1
        assert db.resolved_today_count("rejected", today_prefix) == 0
        assert db.resolved_today_count("approved", "2020-01-01") == 0
