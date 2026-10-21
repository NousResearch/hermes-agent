"""Pinging a bot's auto-created managed role (``<@&role>``) counts as mentioning the bot.

Discord lists the bot user and its same-named managed role side by side in the ``@``
autocomplete, so users regularly pick the role. Before this, such messages were silently
dropped by the require-mention gate."""

from types import SimpleNamespace

import discord

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
SELF_ROLE_ID = 555
OTHER_ROLE_ID = 777


def _role(role_id, bot_id=None):
    return SimpleNamespace(id=role_id, tags=SimpleNamespace(bot_id=bot_id) if bot_id else None)


def _adapter() -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._dedup = MessageDeduplicator()
    return adapter


def _msg(content, *, role_mentions=(), self_role=None, author_bot=False):
    return SimpleNamespace(
        id=1,
        author=SimpleNamespace(id=42, bot=author_bot),
        channel=SimpleNamespace(id=7),
        guild=SimpleNamespace(self_role=self_role),
        content=content,
        mentions=[],
        role_mentions=list(role_mentions),
        type=discord.MessageType.default,
    )


def test_own_managed_role_ping_counts_as_mention():
    adapter = _adapter()
    role = _role(SELF_ROLE_ID, BOT_ID)
    msg = _msg(f"<@&{SELF_ROLE_ID}> can you help?", role_mentions=[role], self_role=role)
    assert adapter._self_is_explicitly_mentioned(msg) is True
    assert adapter._self_is_raw_mentioned(msg) is True


def test_own_role_resolved_from_role_mentions_without_guild_self_role():
    adapter = _adapter()
    role = _role(SELF_ROLE_ID, BOT_ID)
    msg = _msg(f"<@&{SELF_ROLE_ID}> hi", role_mentions=[role], self_role=None)
    assert adapter._self_is_explicitly_mentioned(msg) is True


def test_other_role_ping_is_not_a_mention():
    adapter = _adapter()
    other = _role(OTHER_ROLE_ID)
    other_bot = _role(888, bot_id=1234)
    msg = _msg(f"<@&{OTHER_ROLE_ID}> <@&888> team update",
               role_mentions=[other, other_bot], self_role=_role(SELF_ROLE_ID, BOT_ID))
    assert adapter._self_is_explicitly_mentioned(msg) is False
    assert adapter._self_is_raw_mentioned(msg) is False


def test_self_role_present_but_not_pinged_is_not_a_mention():
    adapter = _adapter()
    msg = _msg("no ping here", self_role=_role(SELF_ROLE_ID, BOT_ID))
    assert adapter._self_is_explicitly_mentioned(msg) is False


def test_dm_without_guild_is_safe():
    adapter = _adapter()
    msg = _msg(f"<@&{SELF_ROLE_ID}>")
    msg.guild = None
    assert adapter._self_is_explicitly_mentioned(msg) is False
