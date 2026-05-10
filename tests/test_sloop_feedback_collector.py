"""Tests for the Sloop feedback collector (tools/feedback_collector.py)."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def tmp_feedback_db(tmp_path, monkeypatch):
    """Redirect the feedback DB to a temp directory."""
    db_path = tmp_path / "sloop_feedback.db"
    monkeypatch.setattr(
        "tools.feedback_collector.get_feedback_db_path", lambda: db_path
    )
    return db_path


class TestRecordFeedback:
    def test_record_and_retrieve(self, tmp_feedback_db):
        from tools.feedback_collector import (
            record_feedback,
            get_recent_feedback,
        )

        record_feedback(
            job_id="test001",
            job_name="Test Job",
            success=True,
            duration_ms=1234.5,
        )

        entries = get_recent_feedback(limit=10)
        assert len(entries) == 1
        assert entries[0]["job_id"] == "test001"
        assert entries[0]["job_name"] == "Test Job"
        assert entries[0]["success"] == 1
        assert entries[0]["duration_ms"] == 1234.5

    def test_record_failure(self, tmp_feedback_db):
        from tools.feedback_collector import (
            record_feedback,
            get_recent_feedback,
        )

        record_feedback(
            job_id="test002",
            job_name="Failing Job",
            success=False,
            error_type="timeout",
            error_msg="Request timed out after 30s",
        )

        entries = get_recent_feedback(limit=10)
        assert len(entries) == 1
        assert entries[0]["success"] == 0
        assert entries[0]["error_type"] == "timeout"

    def test_record_multiple(self, tmp_feedback_db):
        from tools.feedback_collector import (
            record_feedback,
            get_recent_feedback,
        )

        for i in range(5):
            record_feedback(
                job_id=f"job_{i}",
                job_name=f"Job {i}",
                success=i % 2 == 0,
                duration_ms=i * 100.0,
            )

        entries = get_recent_feedback(limit=10)
        assert len(entries) == 5

    def test_record_with_metadata(self, tmp_feedback_db):
        from tools.feedback_collector import (
            record_feedback,
            get_recent_feedback,
        )

        record_feedback(
            job_id="meta001",
            job_name="Meta Job",
            success=True,
            metadata={"model": "gpt-4", "tokens": 500},
        )

        entries = get_recent_feedback(limit=1)
        assert entries[0]["metadata"] is not None
        meta = json.loads(entries[0]["metadata"])
        assert meta["model"] == "gpt-4"

    def test_record_delivery_failure(self, tmp_feedback_db):
        from tools.feedback_collector import (
            record_feedback,
            get_recent_feedback,
        )

        record_feedback(
            job_id="del001",
            job_name="Delivery Fail Job",
            success=True,
            delivery_ok=False,
        )

        entries = get_recent_feedback(limit=1)
        assert entries[0]["delivery_ok"] == 0


class TestSuccessRate:
    def test_all_success(self, tmp_feedback_db):
        from tools.feedback_collector import record_feedback, get_success_rate

        for i in range(10):
            record_feedback(job_id="ok_job", success=True)

        rate = get_success_rate(hours=1)
        assert rate["total"] == 10
        assert rate["successes"] == 10
        assert rate["failures"] == 0
        assert rate["rate_percent"] == 100.0

    def test_mixed_results(self, tmp_feedback_db):
        from tools.feedback_collector import record_feedback, get_success_rate

        for i in range(10):
            record_feedback(job_id="mixed_job", success=i < 7)

        rate = get_success_rate(hours=1)
        assert rate["total"] == 10
        assert rate["successes"] == 7
        assert rate["failures"] == 3
        assert rate["rate_percent"] == 70.0

    def test_filtered_by_job(self, tmp_feedback_db):
        from tools.feedback_collector import record_feedback, get_success_rate

        record_feedback(job_id="job_a", success=True)
        record_feedback(job_id="job_a", success=False)
        record_feedback(job_id="job_b", success=True)

        rate = get_success_rate(job_id="job_a", hours=1)
        assert rate["total"] == 2
        assert rate["successes"] == 1

    def test_empty(self, tmp_feedback_db):
        from tools.feedback_collector import get_success_rate

        rate = get_success_rate(hours=1)
        assert rate["total"] == 0
        assert rate["rate_percent"] == 0.0


class TestAvgDuration:
    def test_avg_duration(self, tmp_feedback_db):
        from tools.feedback_collector import record_feedback, get_avg_duration

        record_feedback(job_id="dur_job", success=True, duration_ms=1000)
        record_feedback(job_id="dur_job", success=True, duration_ms=2000)
        record_feedback(job_id="dur_job", success=True, duration_ms=3000)

        avg = get_avg_duration(hours=1)
        assert avg == 2000.0

    def test_avg_duration_no_data(self, tmp_feedback_db):
        from tools.feedback_collector import get_avg_duration

        avg = get_avg_duration(hours=1)
        assert avg is None


class TestFailureTrend:
    def test_trend_returns_buckets(self, tmp_feedback_db):
        from tools.feedback_collector import record_feedback, get_failure_trend

        record_feedback(job_id="trend_job", success=True)
        record_feedback(job_id="trend_job", success=False)

        trend = get_failure_trend(buckets=3, bucket_minutes=60)
        assert len(trend) == 3
        # At least the most recent bucket should have data
        total_in_trend = sum(b["total"] for b in trend)
        assert total_in_trend >= 2

    def test_trend_empty(self, tmp_feedback_db):
        from tools.feedback_collector import get_failure_trend

        trend = get_failure_trend(buckets=5, bucket_minutes=30)
        assert len(trend) == 5
        assert all(b["total"] == 0 for b in trend)


class TestConsecutiveFailures:
    def test_consecutive_failures_counting(self, tmp_feedback_db):
        from tools.feedback_collector import (
            record_feedback,
            get_consecutive_failures,
        )

        # 3 failures then a success
        record_feedback(job_id="cf_job", success=False)
        record_feedback(job_id="cf_job", success=False)
        record_feedback(job_id="cf_job", success=False)
        record_feedback(job_id="cf_job", success=True)

        assert get_consecutive_failures("cf_job") == 0

    def test_consecutive_failures_active(self, tmp_feedback_db):
        from tools.feedback_collector import (
            record_feedback,
            get_consecutive_failures,
        )

        record_feedback(job_id="cf_job2", success=True)
        record_feedback(job_id="cf_job2", success=False)
        record_feedback(job_id="cf_job2", success=False)

        assert get_consecutive_failures("cf_job2") == 2


class TestErrorTypes:
    def test_error_types(self, tmp_feedback_db):
        from tools.feedback_collector import record_feedback, get_error_types

        for _ in range(3):
            record_feedback(job_id="err_job", success=False, error_type="timeout")
        for _ in range(2):
            record_feedback(job_id="err_job", success=False, error_type="api_error")

        errors = get_error_types(hours=1)
        assert len(errors) == 2
        assert errors[0]["error_type"] == "timeout"
        assert errors[0]["count"] == 3


class TestCleanup:
    def test_cleanup_returns_count(self, tmp_feedback_db):
        from tools.feedback_collector import cleanup_old_feedback

        deleted = cleanup_old_feedback(days=90)
        assert isinstance(deleted, int)


class TestFeedbackTool:
    def test_tool_record_action(self, tmp_feedback_db):
        from tools.feedback_collector import sloop_feedback

        result = json.loads(sloop_feedback(
            action="record",
            job_id="tool001",
            job_name="Tool Job",
            success=True,
            duration_ms=500,
        ))
        assert result["success"] is True

    def test_tool_recent_action(self, tmp_feedback_db):
        from tools.feedback_collector import sloop_feedback, record_feedback

        record_feedback(job_id="tool002", success=True)
        result = json.loads(sloop_feedback(action="recent"))
        assert result["success"] is True
        assert len(result["entries"]) >= 1

    def test_tool_stats_action(self, tmp_feedback_db):
        from tools.feedback_collector import sloop_feedback

        result = json.loads(sloop_feedback(action="stats", hours=1))
        assert result["success"] is True
        assert "success_rate" in result

    def test_tool_unknown_action(self, tmp_feedback_db):
        from tools.feedback_collector import sloop_feedback

        result = json.loads(sloop_feedback(action="nonexistent"))
        assert result["success"] is False


class TestSchemaRegistration:
    """Verify the tool is registered in the registry."""

    def test_registry_has_sloop_feedback(self):
        from tools.registry import registry
        # The tool may not be imported yet in test context,
        # but after import it should be registered
        try:
            import tools.feedback_collector  # noqa: F401
        except ImportError:
            pytest.skip("feedback_collector not importable in test env")
        entry = registry.get_entry("sloop_feedback")
        assert entry is not None
        assert entry.toolset == "sloop"
        assert entry.emoji == "📊"
