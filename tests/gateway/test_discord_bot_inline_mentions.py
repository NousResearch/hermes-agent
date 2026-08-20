"""Tests for hard-inline mention gating on bot-authored Discord messages."""

from types import SimpleNamespace
from unittest.mock import Mock

import discord
import pytest

from gateway.platforms.helpers import MessageDeduplicator
from hermes_cli.config_defaults import DEFAULT_CONFIG
from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter(*, allow_bots: str = "mentions", extra=None) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    setattr(adapter, "config", SimpleNamespace(extra=extra or {}))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._get_allow_bots = Mock(return_value=allow_bots)
    adapter._is_allowed_user = Mock(return_value=True)
    return adapter


def _bot_message(adapter: DiscordAdapter, *, content: str, message_type):
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=42, bot=True),
        channel=SimpleNamespace(id=7),
        content=content,
        mentions=[getattr(adapter._client, "user")],
        type=message_type,
    )


def test_bot_inline_mentions_are_required_by_default(monkeypatch):
    monkeypatch.delenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", raising=False)
    adapter = _adapter()

    assert adapter._discord_bots_require_inline_mention() is True
    assert DEFAULT_CONFIG["discord"]["bots_require_inline_mention"] is True


@pytest.mark.parametrize("allow_bots", ["mentions", "all"])
def test_reply_ping_does_not_trigger_bot_by_default(monkeypatch, allow_bots):
    monkeypatch.delenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", raising=False)
    adapter = _adapter(allow_bots=allow_bots)
    message = _bot_message(
        adapter,
        content="reply without a hard mention",
        message_type=discord.MessageType.reply,
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is False


def test_hard_inline_mention_triggers_bot_by_default(monkeypatch):
    monkeypatch.delenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", raising=False)
    adapter = _adapter(allow_bots="mentions")
    message = _bot_message(
        adapter,
        content="<@99> intentional handoff",
        message_type=discord.MessageType.default,
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is True


def test_human_messages_are_unaffected_by_inline_mention_gate(monkeypatch):
    monkeypatch.delenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", raising=False)
    adapter = _adapter(allow_bots="none")
    message = _bot_message(
        adapter,
        content="human reply without a hard mention",
        message_type=discord.MessageType.reply,
    )
    message.author.bot = False
    message.mentions = []

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is True


def test_explicit_false_restores_reply_ping_compatibility(monkeypatch):
    monkeypatch.delenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", raising=False)
    adapter = _adapter(
        allow_bots="mentions",
        extra={"bots_require_inline_mention": False},
    )
    message = _bot_message(
        adapter,
        content="legacy reply handoff",
        message_type=discord.MessageType.reply,
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is True
