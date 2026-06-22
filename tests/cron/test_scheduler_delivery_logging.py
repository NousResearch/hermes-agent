"""Tests for cron/scheduler.py _deliver_result — live adapter delivery logging (A.1 & A.2).

Covers the warning-logging behaviour added in the dual-bug fix:

A.1 — safe_schedule_threadsafe returns None → WARNING logged, falls back to standalone
A.2 — runtime_adapter exists but loop is None / not running → WARNING logged
Also verifies the normal path (no adapter) does NOT emit spurious warnings.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _deliver_result


def _make_job(platform="telegram", chat_id="12345"):
    return {
        "id": "test-job-abc",
        "name": "test-job",
        "deliver": "origin",
        "origin": {"platform": platform, "chat_id": chat_id},
    }


def _make_pconfig():
    pconfig = MagicMock()
    pconfig.enabled = True
    return pconfig


def _make_gateway_config(platform, pconfig):
    from gateway.config import Platform
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform(platform): pconfig}
    return mock_cfg


class TestLiveAdapterDeliveryLogging:
    """A.1 & A.2: Verify warning logs when live adapter delivery can't proceed."""

    def test_safe_schedule_threadsafe_returns_none_logs_warning(self, caplog):
        """A.1: safe_schedule_threadsafe returns None → WARNING with 'live adapter schedule failed'."""
        from gateway.config import Platform

        adapter = MagicMock()
        adapter.send.return_value = MagicMock(success=True)

        loop = MagicMock()
        loop.is_running.return_value = True

        pconfig = _make_pconfig()
        mock_cfg = _make_gateway_config("telegram", pconfig)
        job = _make_job()

        mock_send = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=mock_send), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("agent.async_utils.safe_schedule_threadsafe", return_value=None):
            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                result = _deliver_result(
                    job, "Test content",
                    adapters={Platform.TELEGRAM: adapter},
                    loop=loop,
                )

        assert result is None
        assert any(
            "live adapter schedule failed" in r.message
            for r in caplog.records
        ), f"Expected 'live adapter schedule failed' warning, got: {[r.message for r in caplog.records]}"
        mock_send.assert_called_once()

    def test_live_adapter_loop_none_logs_warning(self, caplog):
        """A.2: runtime_adapter exists but loop is None → WARNING with 'loop is None'."""
        from gateway.config import Platform

        adapter = AsyncMock()
        pconfig = _make_pconfig()
        mock_cfg = _make_gateway_config("telegram", pconfig)
        job = _make_job()

        mock_send = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=mock_send), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                result = _deliver_result(
                    job, "Test content",
                    adapters={Platform.TELEGRAM: adapter},
                    loop=None,
                )

        assert result is None
        assert any(
            "loop is None" in r.message
            for r in caplog.records
        ), f"Expected 'loop is None' warning, got: {[r.message for r in caplog.records]}"
        mock_send.assert_called_once()
        adapter.send.assert_not_called()

    def test_live_adapter_loop_not_running_logs_warning(self, caplog):
        """A.2: runtime_adapter exists but loop.is_running() is False → WARNING with 'loop is not running'."""
        from gateway.config import Platform

        adapter = AsyncMock()
        loop = MagicMock()
        loop.is_running.return_value = False

        pconfig = _make_pconfig()
        mock_cfg = _make_gateway_config("telegram", pconfig)
        job = _make_job()

        mock_send = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=mock_send), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                result = _deliver_result(
                    job, "Test content",
                    adapters={Platform.TELEGRAM: adapter},
                    loop=loop,
                )

        assert result is None
        assert any(
            "loop is not running" in r.message
            for r in caplog.records
        ), f"Expected 'loop is not running' warning, got: {[r.message for r in caplog.records]}"
        mock_send.assert_called_once()
        adapter.send.assert_not_called()

    def test_no_runtime_adapter_no_warning(self, caplog):
        """Normal path: no adapter for the platform → no loop-related warning logged."""
        pconfig = _make_pconfig()
        mock_cfg = _make_gateway_config("telegram", pconfig)
        job = _make_job()

        mock_send = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=mock_send), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                result = _deliver_result(
                    job, "Test content",
                    adapters={},  # no adapter for telegram
                    loop=None,
                )

        assert result is None
        loop_warnings = [
            r.message for r in caplog.records
            if "loop" in r.message.lower() and "live adapter" in r.message.lower()
        ]
        assert not loop_warnings, (
            f"Expected no loop-related live-adapter warnings when no adapter is present, "
            f"got: {loop_warnings}"
        )
        mock_send.assert_called_once()
