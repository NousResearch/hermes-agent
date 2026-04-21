"""Tests for the delivery routing module."""

import asyncio

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget
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
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="789", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "789"
        assert target.thread_id == "42"
        assert target.is_origin is True

    def test_origin_without_source(self):
        target = DeliveryTarget.parse("origin")
        assert target.platform == Platform.LOCAL
        assert target.is_origin is True

    def test_unknown_platform(self):
        target = DeliveryTarget.parse("unknown_platform")
        assert target.platform == Platform.LOCAL


class TestTargetToStringRoundtrip:
    def test_origin_roundtrip(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="111", thread_id="42")
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


class _RecordingAdapter:
    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"ok": True, "chat_id": chat_id}


class TestDeliverToPlatformHomeChannel:
    def test_bare_platform_target_resolves_configured_home_channel(self):
        # Bare "telegram" target (chat_id=None) must resolve the configured
        # home_channel instead of raising "No chat ID for telegram delivery".
        # Regression for gh-13704.
        cfg = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="x",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="home123",
                    name="Home",
                ),
            )
        })
        adapter = _RecordingAdapter()
        router = DeliveryRouter(cfg, {Platform.TELEGRAM: adapter})

        result = asyncio.run(router.deliver("hello", [DeliveryTarget.parse("telegram")]))

        assert result["telegram"]["success"] is True
        assert adapter.calls == [
            {"chat_id": "home123", "content": "hello", "metadata": None},
        ]

    def test_bare_platform_target_without_home_channel_raises(self):
        cfg = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="x"),
        })
        adapter = _RecordingAdapter()
        router = DeliveryRouter(cfg, {Platform.TELEGRAM: adapter})

        result = asyncio.run(router.deliver("hello", [DeliveryTarget.parse("telegram")]))

        assert result["telegram"]["success"] is False
        assert "No chat ID" in result["telegram"]["error"]
        assert adapter.calls == []

    def test_explicit_chat_id_takes_precedence_over_home_channel(self):
        cfg = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="x",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="home123",
                    name="Home",
                ),
            )
        })
        adapter = _RecordingAdapter()
        router = DeliveryRouter(cfg, {Platform.TELEGRAM: adapter})

        result = asyncio.run(router.deliver("hi", [DeliveryTarget.parse("telegram:555")]))

        assert result["telegram:555"]["success"] is True
        assert adapter.calls == [
            {"chat_id": "555", "content": "hi", "metadata": None},
        ]



