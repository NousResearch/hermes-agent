"""A human's reply to the bot's post that tags other people is addressed to them, not the bot.

Discord's reply-ping adds the replied-to author to ``message.mentions`` even though the
replier never typed the bot's handle. Admission treated that as "bot mentioned", so replying
to a bot's report with "@Alice, take a look" woke the bot in a channel it should stay quiet in.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import discord

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
CHANNEL_ID = 5
OTHER_USER_ID = 820


def _adapter() -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra={})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._is_allowed_user = Mock(return_value=True)
    adapter._get_parent_channel_id = Mock(return_value=None)
    adapter._discord_free_response_channels = Mock(return_value=set())
    adapter._discord_channel_keys = Mock(return_value={str(CHANNEL_ID)})
    adapter._discord_thread_require_mention = Mock(return_value=False)
    adapter._threads = set()
    return adapter


def _reply_to_bot(content: str, *, extra_mentions=()):
    bot = _adapter()._client.user
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=42, bot=False),
        channel=SimpleNamespace(id=CHANNEL_ID, parent_id=None),
        content=content,
        # Discord's reply-ping: the bot is in mentions although its handle isn't in the text.
        mentions=[bot, *extra_mentions],
        type=discord.MessageType.reply,
        guild=SimpleNamespace(id=1),
    )


def _admit(adapter, message):
    adapter._client = SimpleNamespace(user=message.mentions[0])
    return adapter._discord_message_admission(message, claim=False)[0]


def test_reply_to_bot_tagging_someone_else_is_not_for_the_bot(monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    other = SimpleNamespace(id=OTHER_USER_ID, bot=False)
    msg = _reply_to_bot(f"<@{OTHER_USER_ID}> see this", extra_mentions=[other])
    assert _admit(_adapter(), msg) is False


def test_plain_reply_to_bot_still_reaches_it(monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    assert _admit(_adapter(), _reply_to_bot("thanks, also check row 4")) is True


def test_reply_tagging_someone_and_the_bot_still_reaches_it(monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    other = SimpleNamespace(id=OTHER_USER_ID, bot=False)
    msg = _reply_to_bot(f"<@{OTHER_USER_ID}> <@{BOT_ID}> both of you", extra_mentions=[other])
    assert _admit(_adapter(), msg) is True


def test_reply_tagging_someone_in_a_bot_thread_keeps_the_thread_exemption(monkeypatch):
    # In a thread the bot joined it reads every message, tagged or not (#116568); unchanged.
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    adapter = _adapter()
    channel = Mock(spec=discord.Thread)
    channel.id = CHANNEL_ID
    channel.parent_id = None
    adapter._threads = {str(CHANNEL_ID)}
    other = SimpleNamespace(id=OTHER_USER_ID, bot=False)
    msg = _reply_to_bot(f"<@{OTHER_USER_ID}> see this", extra_mentions=[other])
    msg.channel = channel
    assert _admit(adapter, msg) is True
