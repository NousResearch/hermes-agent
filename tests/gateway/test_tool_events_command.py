"""Tests for gateway /tool-events command parsing and output."""

import json
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str = "/tool-events") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(session_db=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_db = session_db
    entry = MagicMock()
    entry.session_id = "session-1"
    entry.session_key = "telegram:u1:c1"
    store = MagicMock()
    store.get_or_create_session.return_value = entry
    runner.session_store = store
    return runner


class TestGatewayToolEventsCommand:
    @pytest.mark.asyncio
    async def test_unavailable_without_session_db(self):
        runner = _make_runner(session_db=None)
        result = await runner._handle_tool_events_command(_make_event("/tool-events"))
        assert "unavailable" in result.lower()

    @pytest.mark.asyncio
    async def test_passes_filters_to_session_db(self):
        db = MagicMock()
        db.get_recent_tool_execution_events.return_value = [
            {
                "tool_name": "browser_click",
                "tool_call_id": "c2",
                "duration_ms": 45,
                "success": 0,
                "retryable": 1,
                "error_summary": "missing ref",
            }
        ]
        runner = _make_runner(session_db=db)

        result = await runner._handle_tool_events_command(
            _make_event("/tool-events 7 errors retryable tool=browser_click")
        )

        db.get_recent_tool_execution_events.assert_called_once_with(
            "session-1",
            limit=7,
            tool_name="browser_click",
            success=False,
            retryable=True,
        )
        assert "browser_click" in result
        assert "missing ref" in result

    @pytest.mark.asyncio
    async def test_json_mode_returns_json_payload(self):
        db = MagicMock()
        db.get_recent_tool_execution_events.return_value = [
            {
                "timestamp": 1710000000.0,
                "tool_name": "web_search",
                "tool_call_id": "c1",
                "duration_ms": 12,
                "success": 1,
                "retryable": 0,
                "envelope": 1,
                "error_type": None,
                "error_summary": "",
            }
        ]
        runner = _make_runner(session_db=db)

        result = await runner._handle_tool_events_command(_make_event("/tool-events json 1"))
        parsed = json.loads(result)
        assert parsed[0]["tool"] == "web_search"
        assert parsed[0]["ok"] is True

    @pytest.mark.asyncio
    async def test_stats_mode_shows_percentiles_and_error_rate(self):
        db = MagicMock()
        db.get_recent_tool_execution_events.return_value = [
            {"timestamp": 1710000005.0, "tool_name": "browser_click", "duration_ms": 10, "success": 0},
            {"timestamp": 1710000004.0, "tool_name": "browser_click", "duration_ms": 40, "success": 1},
            {"timestamp": 1710000003.0, "tool_name": "web_search", "duration_ms": 50, "success": 1},
            {"timestamp": 1710000002.0, "tool_name": "web_search", "duration_ms": 30, "success": 1},
            {"timestamp": 1710000001.0, "tool_name": "web_search", "duration_ms": 20, "success": 0},
        ]
        runner = _make_runner(session_db=db)

        result = await runner._handle_tool_events_command(_make_event("/tool-events stats 20"))

        assert "Tool Event Stats" in result
        assert "web_search" in result
        assert "browser_click" in result
        assert "33.3%" in result
        assert "50.0%" in result
        assert "p50=30ms" in result
        assert "p95=50ms" in result

    @pytest.mark.asyncio
    async def test_stats_json_mode_returns_machine_readable_payload(self):
        db = MagicMock()
        db.get_recent_tool_execution_events.return_value = [
            {"timestamp": 1710000005.0, "tool_name": "browser_click", "duration_ms": 10, "success": 0},
            {"timestamp": 1710000004.0, "tool_name": "browser_click", "duration_ms": 40, "success": 1},
            {"timestamp": 1710000003.0, "tool_name": "web_search", "duration_ms": 50, "success": 1},
        ]
        runner = _make_runner(session_db=db)

        result = await runner._handle_tool_events_command(_make_event("/tool-events stats json 20"))
        parsed = json.loads(result)
        assert parsed["mode"] == "stats"
        assert parsed["bucket_by"] == "tool"
        assert parsed["total_events"] == 3
        assert parsed["buckets"][0]["count"] >= 1

    @pytest.mark.asyncio
    async def test_stats_mode_supports_window_bucketing(self):
        db = MagicMock()
        db.get_recent_tool_execution_events.return_value = [
            {"timestamp": 1710000006.0, "tool_name": "browser_click", "duration_ms": 12, "success": 1},
            {"timestamp": 1710000005.0, "tool_name": "browser_click", "duration_ms": 10, "success": 0},
            {"timestamp": 1710000004.0, "tool_name": "browser_click", "duration_ms": 40, "success": 1},
            {"timestamp": 1710000003.0, "tool_name": "web_search", "duration_ms": 50, "success": 1},
            {"timestamp": 1710000002.0, "tool_name": "web_search", "duration_ms": 30, "success": 1},
            {"timestamp": 1710000001.0, "tool_name": "web_search", "duration_ms": 20, "success": 0},
        ]
        runner = _make_runner(session_db=db)

        result = await runner._handle_tool_events_command(_make_event("/tool-events stats bucket=window window=2 20"))

        assert "bucket=latest_1_2" in result
        assert "bucket=latest_3_4" in result
        assert "bucket=latest_5_6" in result

    @pytest.mark.asyncio
    async def test_text_mode_includes_last_failure_replay_hint(self):
        db = MagicMock()
        db.get_recent_tool_execution_events.return_value = [
            {
                "timestamp": 1710000002.0,
                "tool_name": "web_search",
                "tool_call_id": "ok-1",
                "duration_ms": 8,
                "success": 1,
                "args_preview": '{"query":"ok"}',
                "error_summary": "",
            },
            {
                "timestamp": 1710000001.0,
                "tool_name": "browser_click",
                "tool_call_id": "err-1",
                "duration_ms": 12,
                "success": 0,
                "args_preview": '{"ref":"@e9"}',
                "error_summary": "missing ref",
            },
        ]
        runner = _make_runner(session_db=db)

        result = await runner._handle_tool_events_command(_make_event("/tool-events 5"))

        assert "Last failure replay hint" in result
        assert "browser_click" in result
        assert "tool=browser_click" in result
        assert 'args={"ref":"@e9"}' in result
