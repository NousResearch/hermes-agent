"""Tests for the gateway /debug command."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/debug", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from unittest.mock import AsyncMock, MagicMock
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._update_prompt_pending = {}
    runner.session_store = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit_collect = AsyncMock(return_value=[])
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._check_slash_access = lambda _source, _command: None
    runner._handle_message_with_agent = AsyncMock(return_value="agent response")
    return runner


class TestHandleDebugCommand:
    @pytest.mark.asyncio
    async def test_debug_sweeps_expired_pastes_before_upload(self):
        runner = _make_runner()
        event = _make_event()

        with patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)) as mock_sweep, \
             patch("hermes_cli.debug._capture_dump", return_value="dump"), \
             patch("hermes_cli.debug.collect_debug_report", return_value="report"), \
             patch("hermes_cli.debug.upload_to_pastebin", return_value="https://paste.rs/report"), \
             patch("hermes_cli.debug._schedule_auto_delete"):
            result = await runner._handle_debug_command(event)

        mock_sweep.assert_called_once()
        assert "https://paste.rs/report" in result

    @pytest.mark.asyncio
    async def test_debug_survives_sweep_failure(self):
        runner = _make_runner()
        event = _make_event()

        with patch("hermes_cli.debug._sweep_expired_pastes", side_effect=RuntimeError("offline")), \
             patch("hermes_cli.debug._capture_dump", return_value="dump"), \
             patch("hermes_cli.debug.collect_debug_report", return_value="report"), \
             patch("hermes_cli.debug.upload_to_pastebin", return_value="https://paste.rs/report"), \
             patch("hermes_cli.debug._schedule_auto_delete"):
            result = await runner._handle_debug_command(event)

        assert "https://paste.rs/report" in result


class TestDebugCommandGatewayExposure:
    def test_debug_command_is_cli_only(self):
        from hermes_cli.commands import (
            GATEWAY_KNOWN_COMMANDS,
            gateway_help_lines,
            resolve_command,
            telegram_bot_commands,
        )

        cmd = resolve_command("debug")

        assert cmd is not None
        assert cmd.cli_only is True
        assert "debug" not in GATEWAY_KNOWN_COMMANDS
        assert all("`/debug" not in line for line in gateway_help_lines())
        assert "debug" not in {name for name, _desc in telegram_bot_commands()}

    @pytest.mark.asyncio
    async def test_gateway_debug_request_is_rejected_before_upload(self):
        runner = _make_runner()
        event = _make_event()

        with patch("hermes_cli.debug.upload_to_pastebin") as mock_upload:
            result = await runner._handle_message(event)

        assert result == "Command `/debug` is only available in the local CLI."
        mock_upload.assert_not_called()
        runner._handle_message_with_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_active_gateway_debug_request_is_rejected_before_upload(self):
        runner = _make_runner()
        event = _make_event()
        runner._running_agents[runner._session_key_for_source(event.source)] = object()

        with patch("hermes_cli.debug.upload_to_pastebin") as mock_upload:
            result = await runner._handle_message(event)

        assert result == "Command `/debug` is only available in the local CLI."
        mock_upload.assert_not_called()
        runner._handle_message_with_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_hook_rewrite_to_debug_is_rejected_before_upload(self):
        runner = _make_runner()
        runner.hooks.emit_collect = AsyncMock(
            return_value=[{"decision": "rewrite", "command_name": "debug"}]
        )
        event = _make_event(text="/help")

        with patch("hermes_cli.debug.upload_to_pastebin") as mock_upload:
            result = await runner._handle_message(event)

        assert result == "Command `/debug` is only available in the local CLI."
        mock_upload.assert_not_called()
        runner._handle_message_with_agent.assert_not_called()
