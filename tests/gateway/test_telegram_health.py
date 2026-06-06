"""Tests for TelegramAdapter.get_health() observability.

This is the foundation of the second-Telegram-provider design: the
dispatch layer (and the watchdog cron job) needs observable health to
decide when to fail over.  Counters are updated by the polling loop,
outbound send retries, and the connect-attempt path.
"""

import os
import time
from unittest.mock import patch, MagicMock

import pytest

from gateway.platforms.telegram import TelegramAdapter


def _build_adapter(api_mode="polling", name="telegram-primary"):
    """Build an adapter without triggering the real PTB init."""
    a = TelegramAdapter.__new__(TelegramAdapter)
    # The name property is derived from self.platform, so we need to
    # set that for get_health() to work in tests.
    from gateway.config import Platform
    a.platform = Platform.TELEGRAM
    a._webhook_mode = api_mode == "webhook"
    a._health_poll_success_at = 0.0
    a._health_poll_failure_streak = 0
    a._health_send_success_at = 0.0
    a._health_send_failure_streak = 0
    a._health_connect_failure_streak = 0
    a._health_first_init_at = 0.0
    return a


class TestGetHealth:
    def test_unhealthy_when_never_polled(self):
        a = _build_adapter()
        h = a.get_health()
        assert h["healthy"] is False
        # name comes from the platform property, not our test param
        assert h["name"] == "Telegram"
        assert h["mode"] == "polling"
        assert h["last_poll_success_age"] is None
        assert h["last_send_success_age"] is None
        assert h["poll_failure_streak"] == 0
        assert h["send_failure_streak"] == 0
        assert h["connect_failure_streak"] == 0

    def test_healthy_after_successful_poll(self):
        a = _build_adapter()
        a._health_poll_success_at = time.time() - 5  # 5s ago
        a._health_send_success_at = time.time() - 2
        a._health_first_init_at = time.time() - 100
        h = a.get_health()
        assert h["healthy"] is True
        assert h["last_poll_success_age"] == 5.0
        assert h["last_send_success_age"] == 2.0
        assert h["init_age_seconds"] == 100.0

    def test_unhealthy_above_failure_streak(self):
        a = _build_adapter()
        a._health_poll_success_at = time.time() - 5
        a._health_poll_failure_streak = 5  # above default threshold 3
        h = a.get_health()
        assert h["healthy"] is False
        assert h["poll_failure_streak"] == 5

    def test_unhealthy_above_send_failure_streak(self):
        a = _build_adapter()
        a._health_poll_success_at = time.time() - 5
        a._health_send_failure_streak = 3
        h = a.get_health()
        assert h["healthy"] is False

    def test_unhealthy_above_connect_failure_streak(self):
        a = _build_adapter()
        a._health_poll_success_at = time.time() - 5
        a._health_connect_failure_streak = 3
        h = a.get_health()
        assert h["healthy"] is False

    def test_webhook_mode_reported(self):
        a = _build_adapter(api_mode="webhook")
        a._health_poll_success_at = time.time() - 5
        h = a.get_health()
        assert h["mode"] == "webhook"
        assert h["healthy"] is True

    def test_custom_thresholds_via_env(self):
        a = _build_adapter()
        a._health_poll_success_at = time.time() - 5
        a._health_poll_failure_streak = 2
        # Default threshold is 3 → 2 failures should still be healthy
        h_default = a.get_health()
        assert h_default["healthy"] is True
        # Now bump threshold to 1 via env → 2 failures should be unhealthy
        with patch.dict(
            os.environ, {"HERMES_TELEGRAM_HEALTH_DEGRADE": "1"}
        ):
            h_strict = a.get_health()
            assert h_strict["healthy"] is False
            assert h_strict["degrade_threshold"] == 1

    def test_stale_threshold_included(self):
        a = _build_adapter()
        a._health_poll_success_at = time.time() - 5
        with patch.dict(
            os.environ, {"HERMES_TELEGRAM_HEALTH_STALE": "120"}
        ):
            h = a.get_health()
            assert h["stale_threshold_seconds"] == 120.0

    def test_ages_reported_as_none_when_never_set(self):
        a = _build_adapter()
        h = a.get_health()
        assert h["last_poll_success_age"] is None
        assert h["last_send_success_age"] is None
        assert h["init_age_seconds"] is None
