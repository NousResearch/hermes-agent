"""A thread the bot joined is exempt from the mention requirement on every ingress path.

``_handle_message()`` and ``_dispatch_recovered_message()`` already consult
``_in_bot_thread()``; ``_discord_message_admission()`` runs on both and can veto what they
admit, so it has to agree with them. Without that, a message in a bot thread mentioning a
third party is dropped while the same message with no mention at all is admitted. See #116568.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import discord
import pytest

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


def _message(channel, *, content: str, mentions: list):
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=42, bot=False),
        channel=channel,
        content=content,
        mentions=mentions,
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
    )


def _mentions_third_party(channel):
    return _message(
        channel,
        content=f"<@{OTHER_USER_ID}> test",
        mentions=[SimpleNamespace(id=OTHER_USER_ID, bot=False)],
    )


def test_third_party_mention_in_bot_thread_is_admitted(monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    adapter = _adapter()

    admitted, _ = adapter._discord_message_admission(
        _mentions_third_party(_thread_channel()), claim=False)
    assert admitted is True


def test_no_mention_in_bot_thread_stays_admitted(monkeypatch):
    # The asymmetry the fix removes: this case was already admitted, because an empty
    # ``mentions`` skips the enclosing block entirely.
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    adapter = _adapter()

    admitted, _ = adapter._discord_message_admission(
        _message(_thread_channel(), content="test", mentions=[]), claim=False)
    assert admitted is True


def test_third_party_mention_in_a_plain_channel_is_still_dropped(monkeypatch):
    # Outside a bot thread the gate keeps its original meaning: don't barge into a
    # conversation addressed to someone else.
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    adapter = _adapter()

    admitted, _ = adapter._discord_message_admission(
        _mentions_third_party(SimpleNamespace(id=THREAD_ID, parent_id=None)), claim=False)
    assert admitted is False


def test_thread_require_mention_keeps_multi_bot_threads_strict(monkeypatch):
    # ``thread_require_mention`` gates threads like channels for multi-bot servers;
    # _in_bot_thread() already encodes that, so the exemption must not override it.
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    adapter = _adapter(thread_require_mention=True)

    admitted, _ = adapter._discord_message_admission(
        _mentions_third_party(_thread_channel()), claim=False)
    assert admitted is False


@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_explicit_opt_out_still_admits_everything(monkeypatch, value):
    # The documented workaround keeps working: with the gate off, a third-party mention is
    # admitted in a plain channel too.
    monkeypatch.setenv("DISCORD_IGNORE_NO_MENTION", value)
    adapter = _adapter()

    admitted, _ = adapter._discord_message_admission(
        _mentions_third_party(SimpleNamespace(id=THREAD_ID, parent_id=None)), claim=False)
    assert admitted is True
