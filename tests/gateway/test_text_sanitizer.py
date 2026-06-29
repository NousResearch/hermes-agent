"""Tests for gateway.text_sanitizer — text formatting and redaction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_gw_text_sanitizer",
        Path(__file__).resolve().parents[2] / "gateway" / "text_sanitizer.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGatewayPlatformValue:
    def test_string(self):
        m = _load_module()
        assert m._gateway_platform_value("Telegram") == "telegram"

    def test_enum_value(self):
        m = _load_module()

        class Fake:
            value = "Discord"

        assert m._gateway_platform_value(Fake()) == "discord"

    def test_none(self):
        m = _load_module()
        assert m._gateway_platform_value(None) == ""


class TestGatewaySurfacePassesRawText:
    def test_local_passes(self):
        m = _load_module()
        assert m._gateway_surface_passes_raw_text("local") is True

    def test_api_server_passes(self):
        m = _load_module()
        assert m._gateway_surface_passes_raw_text("api_server") is True

    def test_telegram_blocked(self):
        m = _load_module()
        assert m._gateway_surface_passes_raw_text("telegram") is False

    def test_none_blocked(self):
        m = _load_module()
        assert m._gateway_surface_passes_raw_text(None) is False


class TestLooksLikeGatewayProviderError:
    def test_short_error_detected(self):
        m = _load_module()
        assert m._looks_like_gateway_provider_error("API call failed: 500") is True

    def test_auth_error_detected(self):
        m = _load_module()
        assert m._looks_like_gateway_provider_error("Incorrect API key provided") is True

    def test_long_text_rejected(self):
        m = _load_module()
        assert m._looks_like_gateway_provider_error("A" * 500) is False

    def test_empty_rejected(self):
        m = _load_module()
        assert m._looks_like_gateway_provider_error("") is False


class TestGatewayProviderErrorReply:
    def test_auth_error(self):
        m = _load_module()
        reply = m._gateway_provider_error_reply("Provider authentication failed")
        assert "credentials" in reply.lower() or "check" in reply.lower()

    def test_rate_limit(self):
        m = _load_module()
        reply = m._gateway_provider_error_reply("Rate limited after 3 retries")
        assert "rate" in reply.lower() or "wait" in reply.lower()

    def test_generic_fallback(self):
        m = _load_module()
        reply = m._gateway_provider_error_reply("Something else went wrong")
        assert reply  # should return something non-empty


class TestRenderNoticeLine:
    def test_normal_notice(self):
        m = _load_module()

        class Notice:
            text = "Credits 90% used"

        assert m.render_notice_line(Notice()) == "Credits 90% used"

    def test_empty_notice(self):
        m = _load_module()

        class Notice:
            text = ""

        assert m.render_notice_line(Notice()) == ""

    def test_none_text(self):
        m = _load_module()

        class Notice:
            text = None

        assert m.render_notice_line(Notice()) == ""


class TestIsTransientNetworkError:
    def test_timed_out(self):
        m = _load_module()

        class TimedOut(Exception):
            pass

        assert m._is_transient_network_error(TimedOut("timed out")) is True

    def test_regular_error(self):
        m = _load_module()
        assert m._is_transient_network_error(ValueError("bad value")) is False

    def test_chained_transient(self):
        m = _load_module()

        class ConnectError(Exception):
            pass

        class NetworkError(Exception):
            pass

        outer = NetworkError("network down")
        inner = ConnectError("refused")
        outer.__cause__ = inner
        assert m._is_transient_network_error(outer) is True


class TestNonConversationalMetadata:
    def test_discord_gets_flag(self):
        m = _load_module()
        result = m._non_conversational_metadata({"key": "val"}, platform="discord")
        assert result["non_conversational"] is True
        assert result["key"] == "val"

    def test_other_platform_unchanged(self):
        m = _load_module()
        result = m._non_conversational_metadata({"key": "val"}, platform="telegram")
        assert result == {"key": "val"}

    def test_none_metadata(self):
        m = _load_module()
        result = m._non_conversational_metadata(None, platform="discord")
        assert result["non_conversational"] is True
