"""Tests for gateway session management."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from gateway.config import Platform, HomeChannel, GatewayConfig, PlatformConfig
from gateway.session import (
    SessionSource,
    SessionStore,
    build_session_context,
    build_session_context_prompt,
)


class TestSessionSourceRoundtrip:
    def test_full_roundtrip(self):
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

    def test_full_roundtrip_with_chat_topic(self):
        """chat_topic should survive to_dict/from_dict roundtrip."""
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="789",
            chat_name="Server / #project-planning",
            chat_type="group",
            user_id="42",
            user_name="bob",
            chat_topic="Planning and coordination for Project X",
        )
        d = source.to_dict()
        assert d["chat_topic"] == "Planning and coordination for Project X"

        restored = SessionSource.from_dict(d)
        assert restored.chat_topic == "Planning and coordination for Project X"
        assert restored.chat_name == "Server / #project-planning"

    def test_minimal_roundtrip(self):
        source = SessionSource(platform=Platform.LOCAL, chat_id="cli")
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.platform == Platform.LOCAL
        assert restored.chat_id == "cli"
        assert restored.chat_type == "dm"  # default value preserved

    def test_chat_id_coerced_to_string(self):
        """from_dict should handle numeric chat_id (common from Telegram)."""
        restored = SessionSource.from_dict({
            "platform": "telegram",
            "chat_id": 12345,
        })
        assert restored.chat_id == "12345"
        assert isinstance(restored.chat_id, str)

    def test_missing_optional_fields(self):
        restored = SessionSource.from_dict({
            "platform": "discord",
            "chat_id": "abc",
        })
        assert restored.chat_name is None
        assert restored.user_id is None
        assert restored.user_name is None
        assert restored.thread_id is None
        assert restored.chat_topic is None
        assert restored.chat_type == "dm"

    def test_invalid_platform_raises(self):
        with pytest.raises((ValueError, KeyError)):
            SessionSource.from_dict({"platform": "nonexistent", "chat_id": "1"})


class TestSessionSourceDescription:
    def test_local_cli(self):
        source = SessionSource.local_cli()
        assert source.description == "CLI terminal"

    def test_dm_with_username(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="123",
            chat_type="dm", user_name="bob",
        )
        assert "DM" in source.description
        assert "bob" in source.description

    def test_dm_without_username_falls_back_to_user_id(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="123",
            chat_type="dm", user_id="456",
        )
        assert "456" in source.description

    def test_group_shows_chat_name(self):
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="789",
            chat_type="group", chat_name="Dev Chat",
        )
        assert "group" in source.description
        assert "Dev Chat" in source.description

    def test_channel_type(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="100",
            chat_type="channel", chat_name="Announcements",
        )
        assert "channel" in source.description
        assert "Announcements" in source.description

    def test_thread_id_appended(self):
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="789",
            chat_type="group", chat_name="General",
            thread_id="thread-42",
        )
        assert "thread" in source.description
        assert "thread-42" in source.description

    def test_unknown_chat_type_uses_name(self):
        source = SessionSource(
            platform=Platform.SLACK, chat_id="C01",
            chat_type="forum", chat_name="Questions",
        )
        assert "Questions" in source.description


class TestLocalCliFactory:
    def test_local_cli_defaults(self):
        source = SessionSource.local_cli()
        assert source.platform == Platform.LOCAL
        assert source.chat_id == "cli"
        assert source.chat_type == "dm"
        assert source.chat_name == "CLI terminal"


class TestBuildSessionContextPrompt:
    def test_telegram_prompt_contains_platform_and_chat(self):
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

    def test_discord_prompt(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    token="fake-discord-token",
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_name="Server",
            chat_type="group",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Discord" in prompt

    def test_discord_prompt_with_channel_topic(self):
        """Channel topic should appear in the session context prompt."""
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    token="fake-discord-token",
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_name="Server / #project-planning",
            chat_type="group",
            user_name="alice",
            chat_topic="Planning and coordination for Project X",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Discord" in prompt
        assert "**Channel Topic:** Planning and coordination for Project X" in prompt

    def test_prompt_omits_channel_topic_when_none(self):
        """Channel Topic line should NOT appear when chat_topic is None."""
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    token="fake-discord-token",
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_name="Server / #general",
            chat_type="group",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Channel Topic" not in prompt

    def test_local_prompt_mentions_machine(self):
        config = GatewayConfig()
        source = SessionSource.local_cli()
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Local" in prompt
        assert "machine running this agent" in prompt

    def test_whatsapp_prompt(self):
        config = GatewayConfig(
            platforms={
                Platform.WHATSAPP: PlatformConfig(enabled=True, token=""),
            },
        )
        source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="15551234567@s.whatsapp.net",
            chat_type="dm",
            user_name="Phone User",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "WhatsApp" in prompt or "whatsapp" in prompt.lower()


class TestSessionStoreRewriteTranscript:
    """Regression: /retry and /undo must persist truncated history to disk."""

    @pytest.fixture()
    def store(self, tmp_path):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None  # no SQLite for these tests
        s._loaded = True
        return s

    def test_rewrite_replaces_jsonl(self, store, tmp_path):
        session_id = "test_session_1"
        # Write initial transcript
        for msg in [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "undo this"},
            {"role": "assistant", "content": "ok"},
        ]:
            store.append_to_transcript(session_id, msg)

        # Rewrite with truncated history
        store.rewrite_transcript(session_id, [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])

        reloaded = store.load_transcript(session_id)
        assert len(reloaded) == 2
        assert reloaded[0]["content"] == "hello"
        assert reloaded[1]["content"] == "hi"

    def test_rewrite_with_empty_list(self, store):
        session_id = "test_session_2"
        store.append_to_transcript(session_id, {"role": "user", "content": "hi"})

        store.rewrite_transcript(session_id, [])

        reloaded = store.load_transcript(session_id)
        assert reloaded == []


class TestSessionStoreEntriesAttribute:
    """Regression: /reset must access _entries, not _sessions."""

    def test_entries_attribute_exists(self):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=Path("/tmp"), config=config)
        store._loaded = True
        assert hasattr(store, "_entries")
        assert not hasattr(store, "_sessions")
