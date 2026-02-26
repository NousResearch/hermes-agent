"""Tests for gateway session management."""

from gateway.config import Platform, HomeChannel, GatewayConfig, PlatformConfig
from gateway.session import (
    SessionSource,
    build_session_context,
    build_session_context_prompt,
)


class TestSessionSourceRoundtrip:
    def test_to_dict_from_dict(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_name="My Group",
            chat_type="group",
            user_id="99",
            user_name="alice",
            thread_id="t1",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)

        assert restored.platform == Platform.TELEGRAM
        assert restored.chat_id == "12345"
        assert restored.chat_name == "My Group"
        assert restored.chat_type == "group"
        assert restored.user_id == "99"
        assert restored.user_name == "alice"
        assert restored.thread_id == "t1"

    def test_minimal_roundtrip(self):
        source = SessionSource(platform=Platform.LOCAL, chat_id="cli")
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.platform == Platform.LOCAL
        assert restored.chat_id == "cli"


class TestLocalCliSource:
    def test_local_cli(self):
        source = SessionSource.local_cli()
        assert source.platform == Platform.LOCAL
        assert source.chat_id == "cli"
        assert source.chat_type == "dm"

    def test_description_local(self):
        source = SessionSource.local_cli()
        assert source.description == "CLI terminal"


class TestBuildSessionContextPrompt:
    def test_contains_platform_info(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="fake-token",
                    home_channel=HomeChannel(
                        platform=Platform.TELEGRAM,
                        chat_id="111",
                        name="Home Chat",
                    ),
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="111",
            chat_name="Home Chat",
            chat_type="dm",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Telegram" in prompt
        assert "Home Chat" in prompt
        assert "Session Context" in prompt

    def test_local_source_prompt(self):
        config = GatewayConfig()
        source = SessionSource.local_cli()
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Local" in prompt
        assert "machine running this agent" in prompt
