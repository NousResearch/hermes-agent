"""Tests for agent/audit.py — structured audit event logger."""

import time
from pathlib import Path

import pytest

from agent.audit import AuditLogger, EVENT_TOOL_CALL, EVENT_TOOL_ERROR, EVENT_API_CALL, EVENT_API_ERROR


@pytest.fixture
def audit_db(tmp_path):
    """Create an AuditLogger backed by a temp database."""
    db = AuditLogger(db_path=tmp_path / "test_audit.db")
    yield db
    db.close()


class TestAuditLoggerWrite:

    def test_log_tool_call(self, audit_db):
        audit_db.log_tool_call(
            tool_name="terminal",
            args={"command": "ls -la"},
            result="file1.txt\nfile2.txt",
            duration_ms=120.5,
            success=True,
            session_id="sess-1",
        )
        events = audit_db.query(event_type=EVENT_TOOL_CALL)
        assert len(events) == 1
        assert events[0]["tool_name"] == "terminal"
        assert events[0]["duration_ms"] == 120.5
        assert events[0]["success"] == 1
        assert "ls -la" in events[0]["tool_args"]

    def test_log_tool_error(self, audit_db):
        audit_db.log_tool_error(
            tool_name="terminal",
            args={"command": "rm -rf /"},
            error="Permission denied",
            error_type="PermissionError",
            duration_ms=5.0,
        )
        events = audit_db.query(event_type=EVENT_TOOL_ERROR)
        assert len(events) == 1
        assert events[0]["success"] == 0
        assert events[0]["error_type"] == "PermissionError"
        assert "Permission denied" in events[0]["error_message"]

    def test_log_api_call(self, audit_db):
        audit_db.log_api_call(
            model="claude-sonnet-4-6",
            provider="anthropic",
            status_code=200,
            duration_ms=1500.0,
            input_tokens=5000,
            output_tokens=200,
            session_id="sess-1",
        )
        events = audit_db.query(event_type=EVENT_API_CALL)
        assert len(events) == 1
        assert events[0]["model"] == "claude-sonnet-4-6"
        assert events[0]["input_tokens"] == 5000

    def test_log_api_error(self, audit_db):
        audit_db.log_api_error(
            model="claude-sonnet-4-6",
            provider="anthropic",
            status_code=500,
            error="Internal server error",
            error_type="InternalServerError",
            retry_count=2,
        )
        events = audit_db.query(event_type=EVENT_API_ERROR)
        assert len(events) == 1
        assert events[0]["status_code"] == 500
        assert events[0]["retry_count"] == 2
        assert events[0]["severity"] == "error"

    def test_429_is_warning_not_error(self, audit_db):
        audit_db.log_api_error(status_code=429, error="Rate limited")
        events = audit_db.query(event_type=EVENT_API_ERROR)
        assert events[0]["severity"] == "warning"

    def test_result_preview_truncated(self, audit_db):
        long_result = "x" * 1000
        audit_db.log_tool_call(tool_name="read_file", result=long_result)
        events = audit_db.query()
        assert len(events[0]["tool_result_preview"]) == 500

    def test_args_truncated(self, audit_db):
        long_args = {"data": "y" * 5000}
        audit_db.log_tool_call(tool_name="write_file", args=long_args)
        events = audit_db.query()
        assert len(events[0]["tool_args"]) == 2000


class TestAuditLoggerQuery:

    def test_filter_by_session(self, audit_db):
        audit_db.log_tool_call(tool_name="t1", session_id="sess-A")
        audit_db.log_tool_call(tool_name="t2", session_id="sess-B")
        events = audit_db.query(session_id="sess-A")
        assert len(events) == 1
        assert events[0]["tool_name"] == "t1"

    def test_filter_by_tool_name(self, audit_db):
        audit_db.log_tool_call(tool_name="terminal")
        audit_db.log_tool_call(tool_name="read_file")
        audit_db.log_tool_call(tool_name="terminal")
        events = audit_db.query(tool_name="terminal")
        assert len(events) == 2

    def test_filter_errors_only(self, audit_db):
        audit_db.log_tool_call(tool_name="ok", success=True)
        audit_db.log_tool_error(tool_name="fail", error="boom")
        audit_db.log_api_error(status_code=500, error="down")
        events = audit_db.query(errors_only=True)
        assert len(events) == 2

    def test_filter_by_last_hours(self, audit_db):
        # Insert old event
        audit_db._insert(
            timestamp=time.time() - 7200,  # 2 hours ago
            event_type=EVENT_TOOL_CALL,
            severity="info",
            tool_name="old",
        )
        audit_db.log_tool_call(tool_name="recent")
        events = audit_db.query(last_hours=1)
        assert len(events) == 1
        assert events[0]["tool_name"] == "recent"

    def test_limit(self, audit_db):
        for i in range(10):
            audit_db.log_tool_call(tool_name=f"t{i}")
        events = audit_db.query(limit=3)
        assert len(events) == 3


class TestAuditLoggerSummary:

    def test_summary_basic(self, audit_db):
        audit_db.log_tool_call(tool_name="terminal", duration_ms=100)
        audit_db.log_tool_call(tool_name="terminal", duration_ms=200)
        audit_db.log_tool_call(tool_name="read_file", duration_ms=50)
        audit_db.log_api_call(model="claude")
        audit_db.log_api_error(status_code=500, error="down", error_type="ServerError")

        summary = audit_db.summary(last_hours=1)
        assert summary["total_events"] == 5
        assert summary["errors"] == 1
        assert summary["tool_calls"] == 3
        assert summary["api_calls"] == 1
        assert summary["api_errors"] == 1
        assert len(summary["top_tools"]) == 2
        assert summary["top_tools"][0]["name"] == "terminal"
        assert summary["top_tools"][0]["calls"] == 2

    def test_summary_empty(self, audit_db):
        summary = audit_db.summary(last_hours=1)
        assert summary["total_events"] == 0
        assert summary["errors"] == 0


class TestAuditLoggerResilience:

    def test_closed_logger_does_not_crash(self, audit_db):
        audit_db.close()
        # Should silently no-op
        audit_db.log_tool_call(tool_name="after_close")
        assert audit_db.query() == []

    def test_invalid_db_path_does_not_crash(self):
        audit = AuditLogger(db_path=Path("/nonexistent/path/audit.db"))
        audit.log_tool_call(tool_name="noop")
        assert audit.query() == []
        audit.close()

    def test_concurrent_writes(self, audit_db):
        import threading
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    audit_db.log_tool_call(tool_name=f"thread-{n}", session_id=f"s-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"
        events = audit_db.query(limit=200)
        assert len(events) == 100  # 5 threads * 20 writes
