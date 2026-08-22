"""Observable receipts for Discord admission refusals (#91919).

With a configured allowlist, an unknown sender's ordinary message was dropped
by ``_discord_message_admission`` with NO log line — the only helper on that
path (``_warn_if_fail_closed_default``) warns solely for the no-allowlist
fail-closed case. A user's message could vanish with no operator-visible
trace, breaking the "invite authorizes guild members; refusals must be
observable" security contract.

These tests pin the structured refusal receipt: sender/guild/channel/message
stable IDs + reason, never message content; one receipt per admission check
(deduplicated events never reach admission twice — the dedup gate runs
first).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("discord")

import discord as discord_lib

import plugins.platforms.discord.adapter as adapter_mod
from plugins.platforms.discord.adapter import DiscordAdapter


def _dm_message(sender_id="42", message_id="9001"):
    author = SimpleNamespace(id=int(sender_id), bot=False, name="stranger")
    msg = MagicMock()
    msg.id = int(message_id)
    msg.author = author
    msg.type = discord_lib.MessageType.default
    # guild=None drives the is_dm branch; no real DMChannel is needed
    # (is_dm = isinstance(channel, DMChannel) or guild is None).
    msg.channel = MagicMock()
    msg.guild = None
    return msg


def _guild_message(sender_id="42", message_id="9002"):
    msg = _dm_message(sender_id, message_id)
    guild = SimpleNamespace(id=1234)
    msg.guild = guild
    msg.channel = SimpleNamespace(id=5678)
    return msg


def _adapter_with_allowlist(monkeypatch, allowed_user_ids):
    from gateway.config import Platform

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._allowed_user_ids = set(allowed_user_ids)
    adapter._allowed_role_ids = set()
    adapter._warned_fail_closed_default = True  # isolate from that warning
    dedup = MagicMock()
    dedup.is_duplicate.return_value = False
    dedup.contains.return_value = False
    adapter._dedup = dedup
    client = MagicMock()
    client.user = SimpleNamespace(id=1, bot=True)
    adapter._client = client
    monkeypatch.setattr(
        DiscordAdapter, "_is_allowed_user", lambda self, *a, **k: False
    )
    return adapter


class TestRefusalReceipt:
    def test_unknown_dm_sender_produces_receipt(self, monkeypatch, caplog):
        adapter = _adapter_with_allowlist(monkeypatch, ["1"])
        msg = _dm_message(sender_id="42")

        with caplog.at_level(logging.WARNING, logger=adapter_mod.__name__):
            admitted, _ = adapter._discord_message_admission(msg, claim=True)

        assert admitted is False
        receipts = [r for r in caplog.records if "message refused" in r.getMessage()]
        assert receipts, "unknown DM sender with a configured allowlist must leave a receipt"
        text = receipts[0].getMessage()
        assert "sender_id=42" in text
        assert "dm" in text
        assert "reason=sender_not_in_allowlist" in text

    def test_unknown_guild_sender_includes_guild_and_channel(self, monkeypatch, caplog):
        adapter = _adapter_with_allowlist(monkeypatch, ["1"])
        msg = _guild_message(sender_id="42")

        with caplog.at_level(logging.WARNING, logger=adapter_mod.__name__):
            admitted, _ = adapter._discord_message_admission(msg, claim=True)

        assert admitted is False
        receipts = [r for r in caplog.records if "message refused" in r.getMessage()]
        assert receipts
        text = receipts[0].getMessage()
        assert "sender_id=42" in text
        assert "guild_id=1234" in text
        assert "channel_id=5678" in text

    def test_receipt_never_contains_message_content(self, monkeypatch, caplog):
        adapter = _adapter_with_allowlist(monkeypatch, ["1"])
        msg = _dm_message()
        msg.content = "SECRET PAYLOAD"

        with caplog.at_level(logging.WARNING, logger=adapter_mod.__name__):
            adapter._discord_message_admission(msg, claim=True)

        for r in caplog.records:
            assert "SECRET PAYLOAD" not in r.getMessage()

    def test_deduplicated_event_yields_no_second_receipt(self, monkeypatch, caplog):
        adapter = _adapter_with_allowlist(monkeypatch, ["1"])
        msg = _dm_message(message_id="777")
        adapter._dedup = MagicMock()
        adapter._dedup.contains.return_value = True  # already-seen event

        with caplog.at_level(logging.WARNING, logger=adapter_mod.__name__):
            admitted, _ = adapter._discord_message_admission(msg, claim=False)

        assert admitted is False
        assert not [r for r in caplog.records if "message refused" in r.getMessage()]

    def test_allowed_sender_produces_no_receipt(self, monkeypatch, caplog):
        adapter = _adapter_with_allowlist(monkeypatch, ["1"])
        monkeypatch.setattr(
            DiscordAdapter, "_is_allowed_user", lambda self, *a, **k: True
        )
        # Skip the downstream mention/channel gates this unit isn't about.
        monkeypatch.setattr(
            DiscordAdapter, "_self_is_explicitly_mentioned", lambda self, m: True
        )
        msg = _dm_message()

        with caplog.at_level(logging.WARNING, logger=adapter_mod.__name__):
            admitted, _ = adapter._discord_message_admission(msg, claim=True)

        assert admitted is True
        assert not [r for r in caplog.records if "message refused" in r.getMessage()]
