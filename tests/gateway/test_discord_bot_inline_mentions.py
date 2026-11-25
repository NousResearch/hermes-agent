"""Bot-authored Discord messages need a typed ``<@bot>`` by default; a reply ping alone
no longer wakes the bot (two bots otherwise volley replies forever). Salvage of #106874."""

from types import SimpleNamespace
from unittest.mock import Mock

import discord
import pytest

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter(*, allow_bots: str = "mentions", extra=None) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra=extra or {})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._get_allow_bots = Mock(return_value=allow_bots)
    adapter._is_allowed_user = Mock(return_value=True)
    adapter._text_batch_delay_seconds = 0.6
    adapter._text_batch_split_delay_seconds = 2.0
    adapter._bot_tag_debounce_until = {}
    return adapter


def _reply_ping_from_bot(adapter: DiscordAdapter, content: str):
    # Discord populates ``mentions`` with the replied-to author even though the body
    # carries no ``<@99>`` token — that metadata is what used to count as a summons.
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=42, bot=True),
        channel=SimpleNamespace(id=7),
        content=content,
        mentions=[adapter._client.user],
        type=discord.MessageType.reply,
    )


@pytest.mark.parametrize("allow_bots", ["mentions", "all"])
def test_reply_ping_alone_does_not_trigger_bot_by_default(monkeypatch, allow_bots):
    monkeypatch.delenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", raising=False)
    adapter = _adapter(allow_bots=allow_bots)

    admitted, _ = adapter._discord_message_admission(
        _reply_ping_from_bot(adapter, "reply without a typed mention"), claim=False)
    assert admitted is False

    admitted, _ = adapter._discord_message_admission(
        _reply_ping_from_bot(adapter, "<@99> intentional handoff"), claim=False)
    assert admitted is True


def test_explicit_false_restores_reply_ping_compatibility(monkeypatch):
    monkeypatch.delenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", raising=False)
    adapter = _adapter(extra={"bots_require_inline_mention": False})

    admitted, _ = adapter._discord_message_admission(
        _reply_ping_from_bot(adapter, "legacy reply handoff"), claim=False)
    assert admitted is True


def _flagless_webhook_post(content: str):
    # Discord guarantees only ``webhook_id`` on webhook posts; the author is the webhook's
    # id/name/avatar and real payloads may omit ``bot`` (discord.py then reports bot=False).
    return SimpleNamespace(
        id=124, webhook_id=555, author=SimpleNamespace(id=555, bot=False),
        channel=SimpleNamespace(id=7), content=content, mentions=[],
        type=discord.MessageType.default,
    )


@pytest.mark.parametrize(("allow_bots", "content", "admitted"), [
    ("none", "<@99> deploy failed", False),  # allow-all human rung must not admit a webhook
    ("mentions", "deploy failed", False),
    ("mentions", "<@99> deploy failed", True),
])
def test_flagless_webhook_posts_follow_allow_bots(monkeypatch, allow_bots, content, admitted):
    monkeypatch.delenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", raising=False)
    monkeypatch.setenv("DISCORD_IGNORE_NO_MENTION", "false")
    adapter = _adapter(allow_bots=allow_bots)

    got, _ = adapter._discord_message_admission(_flagless_webhook_post(content), claim=False)
    assert got is admitted
