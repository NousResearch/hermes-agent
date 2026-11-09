"""A thread the bot joined is exempt from the mention requirement on every ingress path.

``_handle_message()`` and ``_dispatch_recovered_message()`` already consult
``_in_bot_thread()``; ``_discord_message_admission()`` runs on both and can veto what they
admit, so it has to agree with them. Without that, a message in a bot thread mentioning a
third party is dropped while the same message with no mention at all is admitted. See #116568.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import discord

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
THREAD_ID = 7
OTHER_USER_ID = 820


def _adapter(*, thread_require_mention: bool = False) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    # self.name (used by the drop-path debug log) reads platform.value.title().
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra={})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._is_allowed_user = Mock(return_value=True)
    adapter._get_parent_channel_id = Mock(return_value=None)
    # No free-response channels: the drop path under test is the one that survives that check.
    adapter._discord_free_response_channels = Mock(return_value=set())
    adapter._discord_channel_keys = Mock(return_value={str(THREAD_ID)})
    adapter._discord_thread_require_mention = Mock(return_value=thread_require_mention)
    adapter._threads = {str(THREAD_ID)}
    return adapter


def _thread_channel():
    # spec= keeps ``isinstance(channel, discord.Thread)`` true inside _in_bot_thread().
    channel = Mock(spec=discord.Thread)
    channel.id = THREAD_ID
    channel.parent_id = None
    return channel


def _mentions_third_party(channel):
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=42, bot=False),
        channel=channel,
        content=f"<@{OTHER_USER_ID}> test",
        mentions=[SimpleNamespace(id=OTHER_USER_ID, bot=False)],
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
    )


def test_third_party_mention_in_bot_thread_is_admitted(monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)

    admitted, _ = _adapter()._discord_message_admission(
        _mentions_third_party(_thread_channel()), claim=False)
    assert admitted is True
    # ``thread_require_mention`` gates threads like channels for multi-bot servers;
    # _in_bot_thread() encodes that, so the exemption must not override it.
    admitted, _ = _adapter(thread_require_mention=True)._discord_message_admission(
        _mentions_third_party(_thread_channel()), claim=False)
    assert admitted is False


def test_third_party_mention_in_a_plain_channel_is_still_dropped_and_logged(monkeypatch, caplog):
    # Outside a bot thread the gate keeps its meaning (don't barge into a conversation
    # addressed to someone else) and now says so, instead of dropping silently.
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    caplog.set_level("DEBUG", logger="plugins.platforms.discord.adapter")

    admitted, _ = _adapter()._discord_message_admission(
        _mentions_third_party(SimpleNamespace(id=THREAD_ID, parent_id=None)), claim=False)
    assert admitted is False
    assert any("admission: dropping message 123" in r.getMessage() for r in caplog.records)
