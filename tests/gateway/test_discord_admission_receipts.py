"""Every refused ordinary Discord message leaves an observable receipt (#91919).

With a configured allowlist an unknown sender's message used to be dropped as a bare
``(False, False)`` — no sender, guild, channel, or reason anywhere. Operators need a
structured refusal log with stable IDs and no message content, and deduplicated or
preview (``claim=False``) evaluations must not double-log.
"""

import logging
from types import SimpleNamespace

import discord
import pytest

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

SECRET_TEXT = "the launch code is 0000"


def _adapter(*, allowed_users=None, allow_bots: str = "mentions") -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set()
    adapter._get_allow_bots = lambda: allow_bots
    adapter._text_batch_delay_seconds = 0.6
    adapter._text_batch_split_delay_seconds = 2.0
    adapter._bot_tag_debounce_until = {}
    return adapter


def _dm_message(msg_id=123, author_id=42):
    return SimpleNamespace(
        id=msg_id,
        author=SimpleNamespace(id=author_id, bot=False),
        channel=SimpleNamespace(id=7),
        guild=None,
        content=SECRET_TEXT,
        mentions=[],
        type=discord.MessageType.default,
    )


def _guild_message(msg_id=124, author_id=42):
    guild = SimpleNamespace(id=1111)
    return SimpleNamespace(
        id=msg_id,
        author=SimpleNamespace(id=author_id, bot=False),
        channel=SimpleNamespace(id=777, guild=guild),
        guild=guild,
        content=SECRET_TEXT,
        mentions=[],
        type=discord.MessageType.default,
    )


def _receipts(caplog):
    return [r for r in caplog.records if "admission refused" in r.getMessage()]


def test_unknown_dm_sender_refusal_emits_receipt(caplog):
    adapter = _adapter(allowed_users={"555"})
    with caplog.at_level(logging.DEBUG):
        admitted, role_authorized = adapter._discord_message_admission(
            _dm_message(), claim=True)
    assert (admitted, role_authorized) == (False, False)
    receipts = _receipts(caplog)
    assert len(receipts) == 1
    text = receipts[0].getMessage()
    assert "reason=user_not_allowed" in text
    assert "user_id=42" in text
    assert "message_id=123" in text
    assert receipts[0].levelno == logging.WARNING
    assert SECRET_TEXT not in caplog.text


def test_unknown_guild_sender_refusal_emits_receipt(caplog):
    adapter = _adapter(allowed_users={"555"})
    with caplog.at_level(logging.DEBUG):
        admitted, _ = adapter._discord_message_admission(_guild_message(), claim=True)
    assert admitted is False
    receipts = _receipts(caplog)
    assert len(receipts) == 1
    text = receipts[0].getMessage()
    assert "reason=user_not_allowed" in text
    assert "guild_id=1111" in text
    assert "channel_id=777" in text
    assert SECRET_TEXT not in caplog.text


def test_deduplicated_event_does_not_emit_second_receipt(caplog):
    adapter = _adapter(allowed_users={"555"})
    with caplog.at_level(logging.DEBUG):
        adapter._discord_message_admission(_guild_message(), claim=True)
        adapter._discord_message_admission(_guild_message(), claim=True)
    assert len(_receipts(caplog)) == 1


def test_claim_false_preview_emits_no_receipt(caplog):
    adapter = _adapter(allowed_users={"555"})
    with caplog.at_level(logging.DEBUG):
        admitted, _ = adapter._discord_message_admission(_guild_message(), claim=False)
    assert admitted is False
    assert _receipts(caplog) == []


def test_authorized_sender_emits_no_receipt(caplog):
    adapter = _adapter(allowed_users={"42"})
    with caplog.at_level(logging.DEBUG):
        admitted, _ = adapter._discord_message_admission(_dm_message(), claim=True)
    assert admitted is True
    assert _receipts(caplog) == []


def test_bot_gate_refusal_receipt_is_debug(caplog):
    adapter = _adapter(allow_bots="mentions")
    bot_msg = SimpleNamespace(
        id=125,
        author=SimpleNamespace(id=55, bot=True),
        channel=SimpleNamespace(id=7),
        guild=None,
        content=SECRET_TEXT,
        mentions=[],
        type=discord.MessageType.default,
    )
    with caplog.at_level(logging.DEBUG):
        admitted, _ = adapter._discord_message_admission(bot_msg, claim=True)
    assert admitted is False
    receipts = _receipts(caplog)
    assert len(receipts) == 1
    assert "reason=bot_mention_required" in receipts[0].getMessage()
    assert receipts[0].levelno == logging.DEBUG
    assert SECRET_TEXT not in caplog.text
