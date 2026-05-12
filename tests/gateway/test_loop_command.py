"""Tests for /loop gateway command.

Verifies that the gateway handler correctly dispatches, parses args, and
returns markdown-formatted responses.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )


def _make_runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    return runner


def _mock_cron_api(success=True, **extra):
    """Return a mock cronjob_tool that returns JSON with given fields."""
    def _cron_tool(**kwargs):
        result = {"success": success, **extra}
        return json.dumps(result)
    return _cron_tool


# ------------------------------------------------------------------
# Gateway dispatch tests
# ------------------------------------------------------------------

class TestLoopCommandGateway:

    @pytest.mark.asyncio
    async def test_no_args_shows_usage_and_list(self):
        """/loop with no args returns usage + job list."""
        runner = _make_runner()

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(jobs=[])):
            result = await runner._handle_loop_command(_make_event("/loop"))

        assert "*/loop <schedule> <prompt>*" in result
        assert "Subcommands:" in result
        assert "No loop jobs" in result

    @pytest.mark.asyncio
    async def test_no_args_with_jobs(self):
        """/loop with no args lists existing loop jobs."""
        runner = _make_runner()

        jobs = [
            {"job_id": "loop_abc123", "name": "loop: check deployment", "schedule": "5m", "state": "active", "prompt_preview": "check deployment"},
        ]

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(jobs=jobs)):
            result = await runner._handle_loop_command(_make_event("/loop"))

        assert "1 loop job(s)" in result
        assert "loop_abc123" in result
        assert "▶" in result  # active icon

    @pytest.mark.asyncio
    async def test_create_parses_schedule_and_prompt(self):
        """/loop 5m check deployment creates a job."""
        runner = _make_runner()

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({
                "success": True,
                "job_id": "loop_abc123",
                "schedule": "5m",
                "next_run_at": "2026-05-09T17:00:00Z",
            })

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            result = await runner._handle_loop_command(_make_event("/loop 5m check deployment"))

        assert len(calls) == 1
        assert calls[0]["action"] == "create"
        assert calls[0]["schedule"] == "5m"
        assert calls[0]["prompt"] == "check deployment"
        assert calls[0]["name"].startswith("loop:")
        assert calls[0]["deliver"] == "origin"
        assert "✅ Loop job created" in result
        assert "`loop_abc123`" in result

    @pytest.mark.asyncio
    async def test_create_empty_prompt_error(self):
        """/loop 5m with no prompt returns usage error."""
        runner = _make_runner()

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api()):
            result = await runner._handle_loop_command(_make_event("/loop 5m"))

        assert "Usage:" in result
        assert "Example:" in result

    @pytest.mark.asyncio
    async def test_list_subcommand(self):
        """/loop list filters by loop: prefix."""
        runner = _make_runner()

        jobs = [
            {"job_id": "j1", "name": "loop: check deployment", "schedule": "5m", "state": "active", "prompt_preview": "check deployment"},
            {"job_id": "j2", "name": "regular job", "schedule": "1h", "state": "active"},
        ]

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(jobs=jobs)):
            result = await runner._handle_loop_command(_make_event("/loop list"))

        assert "*Loop Jobs:*" in result
        assert "j1" in result
        assert "regular job" not in result

    @pytest.mark.asyncio
    async def test_list_no_jobs(self):
        """/loop list with no loop jobs returns friendly message."""
        runner = _make_runner()

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(jobs=[])):
            result = await runner._handle_loop_command(_make_event("/loop list"))

        assert "No loop jobs found" in result

    @pytest.mark.asyncio
    async def test_pause_subcommand(self):
        """/loop pause <id> pauses the job."""
        runner = _make_runner()

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "job": {"name": "loop: test"}})

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            result = await runner._handle_loop_command(_make_event("/loop pause abc123"))

        assert len(calls) == 1
        assert calls[0]["action"] == "pause"
        assert calls[0]["job_id"] == "abc123"
        assert calls[0]["reason"] == "paused from /loop"
        assert "⏸ Paused loop job" in result

    @pytest.mark.asyncio
    async def test_pause_missing_id(self):
        """/loop pause without id returns usage."""
        runner = _make_runner()

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api()):
            result = await runner._handle_loop_command(_make_event("/loop pause"))

        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_resume_subcommand(self):
        """/loop resume <id> resumes the job."""
        runner = _make_runner()

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "job": {"name": "loop: test", "next_run_at": "2026-05-09T18:00:00Z"}})

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            result = await runner._handle_loop_command(_make_event("/loop resume abc123"))

        assert len(calls) == 1
        assert calls[0]["action"] == "resume"
        assert calls[0]["job_id"] == "abc123"
        assert "▶ Resumed loop job" in result
        assert "next run" in result.lower()

    @pytest.mark.asyncio
    async def test_resume_missing_id(self):
        """/loop resume without id returns usage."""
        runner = _make_runner()

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api()):
            result = await runner._handle_loop_command(_make_event("/loop resume"))

        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_remove_subcommand(self):
        """/loop remove <id> deletes the job."""
        runner = _make_runner()

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "removed_job": {"name": "loop: test"}})

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            result = await runner._handle_loop_command(_make_event("/loop remove abc123"))

        assert len(calls) == 1
        assert calls[0]["action"] == "remove"
        assert calls[0]["job_id"] == "abc123"
        assert "🗑 Removed loop job" in result

    @pytest.mark.asyncio
    async def test_remove_missing_id(self):
        """/loop remove without id returns usage."""
        runner = _make_runner()

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api()):
            result = await runner._handle_loop_command(_make_event("/loop remove"))

        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_create_failure(self):
        """Failed create returns error markdown."""
        runner = _make_runner()

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(success=False, error="Invalid schedule")):
            result = await runner._handle_loop_command(_make_event("/loop bad check deployment"))

        assert "⚠ Failed to create loop" in result
        assert "Invalid schedule" in result

    @pytest.mark.asyncio
    async def test_strip_loop_prefix_variants(self):
        """Handler strips /loop, loop, and leading slash correctly."""
        runner = _make_runner()

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "job_id": "loop_xyz", "schedule": "1h", "next_run_at": "2026-05-09T18:00:00Z"})

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            result = await runner._handle_loop_command(_make_event("loop 1h test prompt"))

        assert calls[0]["schedule"] == "1h"
        assert calls[0]["prompt"] == "test prompt"

    @pytest.mark.asyncio
    async def test_gateway_known_command_auto_registered(self):
        """Adding CommandDef to registry auto-registers loop as gateway-known."""
        from hermes_cli.commands import is_gateway_known_command
        assert is_gateway_known_command("loop") is True
        assert is_gateway_known_command("repeat") is True  # alias
