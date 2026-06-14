"""Tests for the gateway /debug command."""

from unittest.mock import patch

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
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
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
