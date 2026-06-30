"""Tests for gateway.display_helpers — display / resolution helper functions."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from gateway.config import Platform
from gateway.display_helpers import (
    _AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT,
    _coerce_gateway_timestamp,
    _float_env,
    _has_platform_display_override,
    _resolve_progress_thread_id,
)


# ---------------------------------------------------------------------------
# _resolve_progress_thread_id
# ---------------------------------------------------------------------------


class TestResolveProgressThreadId:
    """Slack/Mattermost use event_message_id; others prefer source_thread_id."""

    def test_source_thread_id_preferred_over_event_message_id(self):
        """When source_thread_id is present, it wins on all platforms."""
        for platform in ["slack", "mattermost", "telegram", "discord"]:
            assert _resolve_progress_thread_id(platform, "src_123", "evt_456") == "src_123"

    def test_slack_falls_back_to_event_message_id(self):
        assert _resolve_progress_thread_id("slack", None, "evt_999") == "evt_999"

    def test_mattermost_falls_back_to_event_message_id(self):
        assert _resolve_progress_thread_id("mattermost", None, "evt_888") == "evt_888"

    def test_telegram_returns_none_when_no_source_thread_id(self):
        assert _resolve_progress_thread_id("telegram", None, "evt_777") is None

    def test_discord_returns_none_when_no_source_thread_id(self):
        assert _resolve_progress_thread_id("discord", None, "evt_666") is None

    def test_enum_values_work(self):
        """Platform enum objects should work via getattr(platform, 'value', platform)."""
        assert _resolve_progress_thread_id(Platform.SLACK, None, "evt_111") == "evt_111"
        assert _resolve_progress_thread_id(Platform.TELEGRAM, None, "evt_222") is None

    def test_empty_source_returns_none_for_non_slack(self):
        assert _resolve_progress_thread_id("telegram", "", "evt_333") is None

    def test_empty_event_returns_none_for_slack(self):
        assert _resolve_progress_thread_id("slack", None, "") is None


# ---------------------------------------------------------------------------
# _has_platform_display_override
# ---------------------------------------------------------------------------


class TestHasPlatformDisplayOverride:
    """Return True when display.platforms.<platform> explicitly sets setting."""

    def test_returns_true_when_setting_exists(self):
        config = {"display": {"platforms": {"telegram": {"show_reasoning": True}}}}
        assert _has_platform_display_override(config, "telegram", "show_reasoning") is True

    def test_returns_false_when_setting_missing(self):
        config = {"display": {"platforms": {"telegram": {"show_reasoning": True}}}}
        assert _has_platform_display_override(config, "telegram", "tool_progress") is False

    def test_returns_false_when_no_display_key(self):
        assert _has_platform_display_override({}, "telegram", "show_reasoning") is False

    def test_returns_false_when_display_not_dict(self):
        config = {"display": "invalid"}
        assert _has_platform_display_override(config, "telegram", "show_reasoning") is False

    def test_returns_false_when_platforms_not_dict(self):
        config = {"display": {"platforms": "invalid"}}
        assert _has_platform_display_override(config, "telegram", "show_reasoning") is False

    def test_returns_false_when_platform_not_in_platforms(self):
        config = {"display": {"platforms": {"slack": {"show_reasoning": True}}}}
        assert _has_platform_display_override(config, "telegram", "show_reasoning") is False

    def test_returns_false_for_non_dict_user_config(self):
        assert _has_platform_display_override(None, "telegram", "show_reasoning") is False


# ---------------------------------------------------------------------------
# _float_env
# ---------------------------------------------------------------------------


class TestFloatEnv:
    """Parse float from env var, return default on missing/invalid."""

    def test_parses_valid_float(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAL", "42.5")
        assert _float_env("TEST_FLOAT_VAL", 1.0) == 42.5

    def test_parses_valid_int_string(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAL", "100")
        assert _float_env("TEST_FLOAT_VAL", 1.0) == 100.0

    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_FLOAT_VAL", raising=False)
        assert _float_env("TEST_FLOAT_VAL", 7.5) == 7.5

    def test_returns_default_when_empty(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAL", "")
        assert _float_env("TEST_FLOAT_VAL", 7.5) == 7.5

    def test_returns_default_when_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAL", "not-a-number")
        assert _float_env("TEST_FLOAT_VAL", 7.5) == 7.5

    def test_returns_default_as_float(self, monkeypatch):
        monkeypatch.delenv("TEST_FLOAT_VAL", raising=False)
        result = _float_env("TEST_FLOAT_VAL", 3)
        assert result == 3.0
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _coerce_gateway_timestamp
# ---------------------------------------------------------------------------


class TestCoerceGatewayTimestamp:
    """Best-effort conversion of stored gateway timestamps to epoch seconds."""

    def test_returns_none_for_none(self):
        assert _coerce_gateway_timestamp(None) is None

    def test_handles_float_epoch(self):
        assert _coerce_gateway_timestamp(1_700_000_000.5) == 1_700_000_000.5

    def test_handles_int_epoch(self):
        assert _coerce_gateway_timestamp(1_700_000_000) == 1_700_000_000.0

    def test_converts_milliseconds_to_seconds(self):
        """Values > 10 billion are treated as milliseconds."""
        assert _coerce_gateway_timestamp(1_700_000_000_000) == 1_700_000_000.0

    def test_handles_iso_string(self):
        iso = "2023-11-14T22:13:20+00:00"
        result = _coerce_gateway_timestamp(iso)
        assert result is not None
        expected = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc).timestamp()
        assert result == pytest.approx(expected, abs=1e-3)

    def test_handles_iso_string_with_z_suffix(self):
        iso_z = "2023-11-14T22:13:20Z"
        result = _coerce_gateway_timestamp(iso_z)
        assert result is not None
        expected = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc).timestamp()
        assert result == pytest.approx(expected, abs=1e-3)

    def test_handles_numeric_string(self):
        assert _coerce_gateway_timestamp("1700000000") == 1_700_000_000.0

    def test_returns_none_for_empty_string(self):
        assert _coerce_gateway_timestamp("") is None

    def test_returns_none_for_non_numeric_string(self):
        assert _coerce_gateway_timestamp("not-a-timestamp") is None

    def test_returns_none_for_bool_true(self):
        """bool is a subclass of int — must be explicitly rejected."""
        assert _coerce_gateway_timestamp(True) is None

    def test_returns_none_for_bool_false(self):
        assert _coerce_gateway_timestamp(False) is None

    def test_returns_none_for_unsupported_type(self):
        assert _coerce_gateway_timestamp([1, 2, 3]) is None

    def test_handles_datetime_object(self):
        now = datetime.now(tz=timezone.utc)
        result = _coerce_gateway_timestamp(now)
        assert result == pytest.approx(now.timestamp(), abs=1e-3)
