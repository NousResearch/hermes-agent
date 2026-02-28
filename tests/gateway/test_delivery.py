"""Tests for the delivery routing module."""

from gateway.config import Platform, GatewayConfig, PlatformConfig, HomeChannel
from gateway.delivery import DeliveryTarget, parse_deliver_spec
from gateway.session import SessionSource


class TestParseTargetPlatformChat:
    def test_explicit_telegram_chat(self):
        target = DeliveryTarget.parse("telegram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
        assert target.is_explicit is True

    def test_platform_only_no_chat_id(self):
        target = DeliveryTarget.parse("discord")
        assert target.platform == Platform.DISCORD
        assert target.chat_id is None
        assert target.is_explicit is False

    def test_local_target(self):
        target = DeliveryTarget.parse("local")
        assert target.platform == Platform.LOCAL
        assert target.chat_id is None

    def test_origin_with_source(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="789")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "789"
        assert target.is_origin is True

    def test_origin_without_source(self):
        target = DeliveryTarget.parse("origin")
        assert target.platform == Platform.LOCAL
        assert target.is_origin is True

    def test_unknown_platform(self):
        target = DeliveryTarget.parse("unknown_platform")
        assert target.platform == Platform.LOCAL


class TestParseDeliverSpec:
    def test_none_returns_default(self):
        result = parse_deliver_spec(None)
        assert result == "origin"

    def test_empty_string_returns_default(self):
        result = parse_deliver_spec("")
        assert result == "origin"

    def test_custom_default(self):
        result = parse_deliver_spec(None, default="local")
        assert result == "local"

    def test_passthrough_string(self):
        result = parse_deliver_spec("telegram")
        assert result == "telegram"

    def test_passthrough_list(self):
        result = parse_deliver_spec(["local", "telegram"])
        assert result == ["local", "telegram"]


class TestTargetToStringRoundtrip:
    def test_origin_roundtrip(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="111")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.to_string() == "origin"

    def test_local_roundtrip(self):
        target = DeliveryTarget.parse("local")
        assert target.to_string() == "local"

    def test_platform_only_roundtrip(self):
        target = DeliveryTarget.parse("discord")
        assert target.to_string() == "discord"

    def test_explicit_chat_roundtrip(self):
        target = DeliveryTarget.parse("telegram:999")
        s = target.to_string()
        assert s == "telegram:999"

        reparsed = DeliveryTarget.parse(s)
        assert reparsed.platform == Platform.TELEGRAM
        assert reparsed.chat_id == "999"
