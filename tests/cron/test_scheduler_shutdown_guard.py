"""Regression coverage for #58720 / #55924 — cron scheduling races
interpreter finalization.

When the gateway tears down (SIGTERM from ``hermes update`` /
``hermes gateway stop`` / systemd restart, or an OOM-kill), a cron tick can
still fire. Once the Python interpreter is finalizing, ``concurrent.futures``
refuses new work with ``RuntimeError: cannot schedule new futures after
interpreter shutdown`` and asyncio's default executor is gone. The cron
delivery + dispatch paths used to hit that unguarded, crashing the tick and
spraying a traceback into ``errors.log`` on every restart-race.

The fix adds ``_interpreter_shutting_down()`` and guards the scheduling
sites so they skip gracefully with a warning instead of raising.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

class TestInterpreterShuttingDownHelper:

    def test_matches_shutdown_error_text_as_fallback(self):
        """The concurrent.futures module-global flag can be set a hair before
        ``sys.is_finalizing()`` flips — matching the error text catches that
        race so a shutdown RuntimeError isn't misread as a real failure."""
        from cron.scheduler import _interpreter_shutting_down

        exc = RuntimeError("cannot schedule new futures after interpreter shutdown")
        with patch("sys.is_finalizing", return_value=False):
            assert _interpreter_shutting_down(exc) is True

    def test_unrelated_error_is_not_shutdown(self):
        from cron.scheduler import _interpreter_shutting_down

        exc = RuntimeError("some other problem")
        with patch("sys.is_finalizing", return_value=False):
            assert _interpreter_shutting_down(exc) is False

class TestStandaloneDeliverySkipsDuringShutdown:
    def _telegram_cfg(self):
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        return mock_cfg

    def test_standalone_path_skips_without_scheduling(self):
        """With the interpreter finalizing, the standalone delivery path must
        skip BEFORE attempting to schedule the send — no ``_send_to_platform``
        call, a graceful warning-level skip, and an error string returned
        (not a raised exception)."""
        from cron.scheduler import _deliver_result

        job = {
            "id": "gov-job",
            "name": "model-governor",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        send_mock = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=self._telegram_cfg()), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("sys.is_finalizing", return_value=True):
            result = _deliver_result(job, "daily report body")

        send_mock.assert_not_called()
        assert result is not None
