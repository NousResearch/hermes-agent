"""Tests for the structured audit log module."""

import json
import tempfile
import time
import os
from pathlib import Path

from agent.audit import (
    configure,
    start_session,
    end_session,
    log_api_request,
    log_tool_call,
    log_honcho_operation,
    log_cron_run,
    is_enabled,
    list_sessions,
    query_events,
    export_events,
    emit,
    _sanitize_args,
    _rotate_logs,
)


class TestAuditSanitize:
    def test_redacts_sensitive_keys(self):
        args = {"query": "hello", "password": "secret123", "api_key": "sk-12345"}
        result = _sanitize_args(args)
        assert result["query"] == "hello"
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"

    def test_truncates_long_values(self):
        args = {"content": "x" * 3000}
        result = _sanitize_args(args)
        assert len(result["content"]) < 3000
        assert "3000 chars" in result["content"]

    def test_passthrough_when_redact_disabled(self):
        import agent.audit as audit
        old = audit._redact
        audit._redact = False
        try:
            args = {"password": "secret123"}
            result = _sanitize_args(args)
            assert result["password"] == "secret123"
        finally:
            audit._redact = old


def _setup_tmpdir(tmpdir):
    """Patch audit module to use a temp directory."""
    import agent.audit as audit
    old_dir = audit._AUDIT_DIR
    old_link = audit._LATEST_LINK
    audit._AUDIT_DIR = Path(tmpdir)
    audit._LATEST_LINK = Path(tmpdir) / "latest.jsonl"
    return old_dir, old_link


def _teardown(old_dir, old_link):
    import agent.audit as audit
    end_session()
    configure(enabled=False)
    audit._AUDIT_DIR = old_dir
    audit._LATEST_LINK = old_link


class TestAuditLifecycle:
    def test_disabled_by_default(self):
        configure(enabled=False)
        assert not is_enabled()

    def test_writes_all_event_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True, redact=True)
                start_session("test-001", model="test-model", provider="test")

                log_api_request(
                    model="test-model", provider="test",
                    prompt_tokens=100, completion_tokens=50, total_tokens=150,
                    cost_usd=0.001, duration_ms=1234.5, finish_reason="stop",
                )
                log_tool_call(
                    tool_name="read_file",
                    args={"path": "/tmp/test.txt"},
                    result='{"content": "hello"}',
                    duration_ms=12.3,
                )
                log_tool_call(
                    tool_name="execute_code",
                    args={"code": "print('hi')"},
                    error="SandboxError: denied",
                )
                log_honcho_operation(
                    operation="honcho_conclude",
                    payload={"conclusion": "user prefers dark mode"},
                    result="ok",
                )
                log_cron_run(
                    job_id="cron-001", job_name="daily-report",
                    success=True, duration_s=45.2,
                )
                end_session(duration_s=60.0, total_tokens=500)

                log_path = Path(tmpdir) / "test-001.jsonl"
                assert log_path.exists()
                lines = log_path.read_text().strip().split("\n")
                events = [json.loads(line) for line in lines]

                assert len(events) == 7
                types = [e["type"] for e in events]
                assert types == [
                    "session.start", "api.request", "tool.call",
                    "tool.error", "honcho.operation", "cron.run", "session.end",
                ]

                # Verify specific fields
                assert events[1]["prompt_tokens"] == 100
                assert events[1]["cost_usd"] == 0.001
                assert events[2]["tool"] == "read_file"
                assert events[3]["error"] == "SandboxError: denied"
                assert events[4]["operation"] == "honcho_conclude"
                assert events[5]["job_name"] == "daily-report"
                assert events[5]["success"] is True

                # Symlink created
                latest = Path(tmpdir) / "latest.jsonl"
                assert latest.is_symlink()
            finally:
                _teardown(old_dir, old_link)


class TestAuditQuery:
    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("sess-a", model="gpt-4", platform="cli")
                log_tool_call(tool_name="read_file", args={}, result="ok")
                end_session()

                sessions = list_sessions()
                assert len(sessions) >= 1
                # Find our session in the results
                our = [s for s in sessions if s["session_id"] == "sess-a"]
                assert len(our) == 1
                assert our[0]["session_id"] == "sess-a"
            finally:
                _teardown(old_dir, old_link)

    def test_query_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("sess-b", model="test")
                log_api_request(model="test", total_tokens=100)
                log_tool_call(tool_name="read_file", args={}, result="ok")
                log_tool_call(tool_name="write_file", args={}, result="ok")
                end_session()

                api_events = query_events(event_type="api.request")
                assert len(api_events) == 1
                tool_events = query_events(event_type="tool.call")
                assert len(tool_events) == 2
            finally:
                _teardown(old_dir, old_link)

    def test_query_by_tool_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("sess-c", model="test")
                log_tool_call(tool_name="read_file", args={}, result="ok")
                log_tool_call(tool_name="write_file", args={}, result="ok")
                end_session()

                events = query_events(tool_name="write_file")
                assert len(events) == 1
                assert events[0]["tool"] == "write_file"
            finally:
                _teardown(old_dir, old_link)

    def test_query_keyword(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("sess-d", model="test")
                log_tool_call(tool_name="search_files", args={"query": "foobar"}, result="found")
                log_tool_call(tool_name="read_file", args={"path": "/tmp/x"}, result="ok")
                end_session()

                events = query_events(keyword="foobar")
                assert len(events) == 1
                assert events[0]["tool"] == "search_files"
            finally:
                _teardown(old_dir, old_link)

    def test_query_by_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("sess-source", model="test")
                log_tool_call(tool_name="read_file", args={}, result="ok")
                emit("honcho.operation", operation="search", payload={"query": "q"}, result_summary="ok")
                emit("mcp.tool_call", mcp_tool="obsidian.search", query="vault")
                end_session()

                core_events = query_events(source="core")
                assert len(core_events) >= 3
                assert all(e["type"].startswith(("session.", "tool.")) for e in core_events)

                honcho_events = query_events(source="honcho")
                assert len(honcho_events) == 1
                assert honcho_events[0]["type"] == "honcho.operation"
                assert honcho_events[0]["source"] == "honcho"

                mcp_events = query_events(source="mcp")
                assert len(mcp_events) == 1
                assert mcp_events[0]["type"] == "mcp.tool_call"
                assert mcp_events[0]["source"] == "mcp"
            finally:
                _teardown(old_dir, old_link)

    def test_export_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("sess-e", model="test")
                log_tool_call(tool_name="read_file", args={}, result="ok")
                end_session()

                csv_str = export_events(session_id="sess-e", format="csv")
                assert "tool" in csv_str or "type" in csv_str
                assert "read_file" in csv_str
            finally:
                _teardown(old_dir, old_link)


class TestAuditRotation:
    def test_rotation_removes_old_files(self):
        import agent.audit as audit
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            old_retention = audit._retention_days
            try:
                audit._retention_days = 1
                # Create an old file
                old_file = Path(tmpdir) / "old-session.jsonl"
                old_file.write_text('{"type":"session.start"}\n')
                # Set mtime to 3 days ago
                old_time = time.time() - (3 * 86400)
                os.utime(old_file, (old_time, old_time))

                # Create a recent file
                new_file = Path(tmpdir) / "new-session.jsonl"
                new_file.write_text('{"type":"session.start"}\n')

                _rotate_logs()

                assert not old_file.exists()
                assert new_file.exists()
            finally:
                audit._retention_days = old_retention
                _teardown(old_dir, old_link)


# =========================================================================
# SQLite audit integration tests
# =========================================================================

class TestAuditSQLite:
    """Integration tests for SQLite-backed audit storage."""

    def test_dual_write_and_query(self):
        """Events written via audit.py appear in both JSONL and SQLite."""
        import agent.audit as audit
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True, user_id="alice", platform="cli")
                start_session("dual-001", model="test", user_id="alice", platform="cli")
                log_tool_call(tool_name="terminal", args={"command": "ls"}, result='{"output":"ok"}', duration_ms=50.0)
                emit("custom.event", data="test-data")
                end_session()

                # JSONL exists
                jsonl_path = Path(tmpdir) / "dual-001.jsonl"
                assert jsonl_path.exists()
                lines = jsonl_path.read_text().strip().split("\n")
                assert len(lines) >= 4  # start, tool, custom, end

                # SQLite query by session
                events = query_events(session_id="dual-001")
                types = {e.get("type") for e in events}
                assert "session.start" in types
                assert "tool.call" in types
                assert "custom.event" in types

                # Query by user
                assert len(query_events(user_id="alice")) >= 1
                assert len(query_events(user_id="bob")) == 0

                # Query by tool
                assert len(query_events(tool_name="terminal")) >= 1

                # Keyword search
                assert len(query_events(keyword="ls")) >= 1
            finally:
                _teardown(old_dir, old_link)

    def test_date_range_filter(self):
        """Date range filtering works through query_events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                before_ts = time.time()
                configure(enabled=True)
                start_session("range-001", model="test")
                log_tool_call(tool_name="read_file", args={}, result="ok")
                end_session()

                # Nothing before session start
                assert len(query_events(before=before_ts)) == 0
                # Everything after session start
                assert len(query_events(after=before_ts)) >= 1
            finally:
                _teardown(old_dir, old_link)

    def test_graceful_degradation(self):
        """JSONL still works if SQLite is unavailable."""
        import agent.audit as audit
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                audit._db = None
                start_session("fallback-001", model="test")
                audit._db = None  # clear again after start_session creates it
                log_tool_call(tool_name="read_file", args={}, result="ok")
                end_session()

                jsonl_path = Path(tmpdir) / "fallback-001.jsonl"
                assert jsonl_path.exists()
                lines = jsonl_path.read_text().strip().split("\n")
                assert len(lines) >= 2
            finally:
                _teardown(old_dir, old_link)

    def test_query_by_source_in_jsonl_fallback(self):
        """Source filtering works in JSONL fallback mode too."""
        import agent.audit as audit
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("fallback-source", model="test")
                audit._db = None  # force query_events to use JSONL fallback
                log_tool_call(tool_name="read_file", args={}, result="ok")
                emit("plugin.action", plugin="radio", action="play")
                end_session()

                plugin_events = query_events(session_id="fallback-source", source="plugin")
                assert len(plugin_events) == 1
                assert plugin_events[0]["type"] == "plugin.action"
                assert plugin_events[0]["source"] == "plugin"
            finally:
                _teardown(old_dir, old_link)

    def test_skill_manage_detail_extraction(self):
        """skill_manage tool calls get rich detail extraction."""
        from agent.audit import _extract_tool_detail
        detail = _extract_tool_detail(
            "skill_manage",
            {"action": "create", "name": "my-skill", "category": "devops"},
            '{"success": true}',
        )
        assert detail["action"] == "create"
        assert detail["skill"] == "my-skill"


class TestAuditSummary:
    def test_summary_aggregates_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("sum-001", model="test")
                from agent.audit import log_api_request, audit_summary
                log_api_request(model="test", total_tokens=500, prompt_tokens=300,
                                completion_tokens=200, cost_usd=0.01)
                log_api_request(model="test", total_tokens=800, prompt_tokens=500,
                                completion_tokens=300, cost_usd=0.02)
                log_tool_call(tool_name="terminal", args={"command": "ls"}, result="ok", duration_ms=50)
                log_tool_call(tool_name="terminal", args={"command": "pwd"}, result="ok", duration_ms=30)
                log_tool_call(tool_name="read_file", args={}, result="ok", duration_ms=10)
                end_session()

                s = audit_summary(session_id="sum-001")
                assert s["total"] >= 5
                assert s["api_calls"] == 2
                assert s["tool_calls"] == 3
                assert s["total_tokens"] == 1300
                assert s["total_cost_usd"] == 0.03
                assert len(s["top_tools"]) >= 1
                assert s["top_tools"][0]["tool"] == "terminal"
                assert s["top_tools"][0]["count"] == 2
            finally:
                _teardown(old_dir, old_link)

    def test_summary_empty_returns_zero(self):
        from agent.audit import audit_summary
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                s = audit_summary()
                assert s["total"] == 0
            finally:
                _teardown(old_dir, old_link)


class TestAuditProblems:
    def test_detects_repeated_tool_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("prob-001", model="test")
                for _ in range(4):
                    log_tool_call(tool_name="terminal", args={}, error="SandboxError: denied")
                end_session()

                from agent.audit import audit_problems
                findings = audit_problems(session_id="prob-001")
                rules = [f["rule"] for f in findings]
                assert "repeated_tool_failure" in rules
            finally:
                _teardown(old_dir, old_link)

    def test_no_problems_when_clean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("prob-002", model="test")
                log_tool_call(tool_name="read_file", args={}, result="ok", duration_ms=50)
                end_session()

                from agent.audit import audit_problems
                findings = audit_problems(session_id="prob-002")
                assert len(findings) == 0
            finally:
                _teardown(old_dir, old_link)

    def test_detects_high_error_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_link = _setup_tmpdir(tmpdir)
            try:
                configure(enabled=True)
                start_session("prob-003", model="test")
                log_tool_call(tool_name="read_file", args={}, result="ok")
                for _ in range(5):
                    log_tool_call(tool_name="terminal", args={}, error="fail")
                end_session()

                from agent.audit import audit_problems
                findings = audit_problems(session_id="prob-003")
                rules = [f["rule"] for f in findings]
                assert "high_error_rate" in rules
            finally:
                _teardown(old_dir, old_link)
