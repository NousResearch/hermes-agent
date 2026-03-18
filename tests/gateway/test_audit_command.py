"""Tests for gateway /audit command filtering behavior."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/audit", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner with minimal mocks."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.session_store = SimpleNamespace()
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    return runner


class TestHandleAuditCommand:
    """Tests for GatewayRunner._handle_audit_command."""

    @pytest.mark.asyncio
    async def test_parses_after_before_and_passes_timestamps(self, monkeypatch):
        runner = _make_runner()
        event = _make_event(
            text="/audit --after 2026-03-17T12:00:00 --before 2026-03-17T18:00:00 --source core 5"
        )

        captured = {}

        def fake_query_events(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr("agent.audit.query_events", fake_query_events)
        monkeypatch.setattr("agent.audit.list_sessions", lambda: [])

        result = await runner._handle_audit_command(event)

        assert result == "No audit events found. Enable with `audit.enabled: true` in config.yaml"
        assert captured["source"] == "core"
        assert captured["limit"] == 5
        # fromisoformat on naive string is local-tz-dependent; just verify
        # the gateway parsed both timestamps and they're 6 hours apart.
        expected_after = datetime.fromisoformat("2026-03-17T12:00:00").timestamp()
        expected_before = datetime.fromisoformat("2026-03-17T18:00:00").timestamp()
        assert captured["after"] == pytest.approx(expected_after)
        assert captured["before"] == pytest.approx(expected_before)

    @pytest.mark.asyncio
    async def test_invalid_after_date_returns_error(self, monkeypatch):
        runner = _make_runner()
        event = _make_event(text="/audit --after not-a-date")

        monkeypatch.setattr("agent.audit.query_events", lambda **kwargs: [])
        monkeypatch.setattr("agent.audit.list_sessions", lambda: [])

        result = await runner._handle_audit_command(event)

        assert result == "Cannot parse date: not-a-date"

    @pytest.mark.asyncio
    async def test_sessions_lists_recent_sessions_and_caps_at_20(self, monkeypatch):
        runner = _make_runner()
        event = _make_event(text="/audit sessions")

        sessions = [
            {
                "session_id": f"sess-{i:02d}",
                "size_kb": i + 1,
                "modified": f"2026-03-17 1{i % 10}:00",
            }
            for i in range(25)
        ]

        monkeypatch.setattr("agent.audit.list_sessions", lambda: sessions)
        monkeypatch.setattr("agent.audit.query_events", lambda **kwargs: [])

        result = await runner._handle_audit_command(event)

        assert "**Audit Sessions**" in result
        assert "`sess-00` 1kb" in result
        assert "`sess-19` 20kb" in result
        assert "sess-20" not in result
        assert "sess-24" not in result

    @pytest.mark.asyncio
    async def test_formats_source_tags_and_tool_detail_previews(self, monkeypatch):
        runner = _make_runner()
        event = _make_event(text="/audit 3")

        events = [
            {
                "iso": "2026-03-17T12:00:00+0000",
                "type": "tool.call",
                "source": "core",
                "tool": "terminal",
                "detail": {"command": "echo hello from terminal"},
                "duration_ms": 12.0,
            },
            {
                "iso": "2026-03-17T12:01:00+0000",
                "type": "tool.call",
                "source": "core",
                "tool": "read_file",
                "detail": {"path": "/tmp/example.txt"},
                "duration_ms": 8.0,
            },
            {
                "iso": "2026-03-17T12:02:00+0000",
                "type": "honcho.operation",
                "source": "honcho",
                "operation": "search",
            },
        ]

        monkeypatch.setattr("agent.audit.query_events", lambda **kwargs: events)
        monkeypatch.setattr("agent.audit.list_sessions", lambda: [])

        result = await runner._handle_audit_command(event)

        assert "[core] **tool.call** terminal 12ms echo hello from terminal" in result
        assert "[core] **tool.call** read_file 8ms /tmp/example.txt" in result
        assert "[honcho] **honcho.operation** search" in result

    @pytest.mark.asyncio
    async def test_mixed_flags_unknown_tokens_and_repeated_source_parse_cleanly(self, monkeypatch):
        runner = _make_runner()
        event = _make_event(
            text=(
                "/audit --type tool.call junk --tool terminal --source honcho "
                "--source core --session sess-1 --keyword foo "
                "--after 2026-03-17T12:00:00 --before 2026-03-17T18:00:00 7 extra"
            )
        )

        captured = {}

        def fake_query_events(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr("agent.audit.query_events", fake_query_events)
        monkeypatch.setattr("agent.audit.list_sessions", lambda: [])

        result = await runner._handle_audit_command(event)

        assert result == "No audit events found. Enable with `audit.enabled: true` in config.yaml"
        assert captured["event_type"] == "tool.call"
        assert captured["tool_name"] == "terminal"
        assert captured["source"] == "core"
        assert captured["session_id"] == "sess-1"
        assert captured["keyword"] == "foo"
        assert captured["limit"] == 7
        expected_after = datetime.fromisoformat("2026-03-17T12:00:00").timestamp()
        expected_before = datetime.fromisoformat("2026-03-17T18:00:00").timestamp()
        assert captured["after"] == pytest.approx(expected_after)
        assert captured["before"] == pytest.approx(expected_before)
