"""A role ping naming a role the bot holds is a mention of the bot.

Discord resolves ``<@&ROLE>`` into ``message.role_mentions`` but never expands the role's
members into ``message.mentions``, so a bot could not tell a role ping aimed at it from
ordinary chatter — the ``require_mention`` gate dropped it before any reaction was added.
With ``discord.role_mentions`` enabled the bot answers a ping of a role it holds.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import discord

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
GUILD_ID = 1
EVERYONE_ROLE_ID = GUILD_ID  # Discord's @everyone role ID is the guild ID
HELD_ROLE_ID = 4242
OTHER_ROLE_ID = 4343


def _adapter(*, role_mentions: bool = True, extra=None) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra=extra or {})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._is_allowed_user = Mock(return_value=True)
    adapter._get_allow_bots = Mock(return_value="mentions")
    adapter._get_parent_channel_id = Mock(return_value=None)
    adapter._discord_free_response_channels = Mock(return_value=set())
    adapter._discord_channel_keys = Mock(return_value={"7"})
    adapter._discord_thread_require_mention = Mock(return_value=False)
    adapter._threads = set()
    adapter._discord_role_mentions = Mock(return_value=role_mentions)
    return adapter


def _guild(*, bot_role_ids=(HELD_ROLE_ID,)):
    """The bot's own member: @everyone plus whatever roles it holds."""
    roles = [SimpleNamespace(id=EVERYONE_ROLE_ID), *(SimpleNamespace(id=r) for r in bot_role_ids)]
    return SimpleNamespace(id=GUILD_ID, me=SimpleNamespace(id=BOT_ID, roles=roles))


def _role_ping(content: str, *, role_ids=(HELD_ROLE_ID,), bot_role_ids=(HELD_ROLE_ID,), resolved=True):
    """A message that pings ``role_ids``; ``resolved`` mirrors Discord populating role_mentions."""
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=42, bot=False),
        channel=SimpleNamespace(id=7),
        content=content,
        mentions=[],
        role_mentions=[SimpleNamespace(id=r) for r in role_ids] if resolved else [],
        type=discord.MessageType.default,
        guild=_guild(bot_role_ids=bot_role_ids),
    )


def test_role_ping_of_held_role_is_a_mention():
    adapter = _adapter()
    message = _role_ping(f"<@&{HELD_ROLE_ID}> 你在吗？")

    assert adapter._self_is_explicitly_mentioned(message) is True
    assert adapter._self_is_role_mentioned(message) is True


def test_role_ping_of_role_the_bot_lacks_is_not_a_mention():
    adapter = _adapter()
    message = _role_ping(f"<@&{OTHER_ROLE_ID}> 你在吗？", bot_role_ids=())

    assert adapter._self_is_role_mentioned(message) is False
    assert adapter._self_is_explicitly_mentioned(message) is False


def test_role_ping_is_ignored_until_the_option_is_enabled():
    adapter = _adapter(role_mentions=False)
    message = _role_ping(f"<@&{HELD_ROLE_ID}> 你在吗？")

    assert adapter._self_is_role_mentioned(message) is False
    assert adapter._self_is_explicitly_mentioned(message) is False


def test_everyone_role_is_never_a_summons():
    """Every member holds @everyone, so counting it would wake the bot on any server-wide ping."""
    adapter = _adapter()
    message = _role_ping(f"<@&{EVERYONE_ROLE_ID}> 大家好", bot_role_ids=())

    assert adapter._self_is_role_mentioned(message) is False


def test_raw_role_token_counts_when_discord_did_not_resolve_it():
    """Relayed/edited messages can arrive without ``role_mentions``; the raw token still counts."""
    adapter = _adapter()
    message = _role_ping(f"<@&{HELD_ROLE_ID}> 你在吗？", resolved=False)

    assert adapter._self_is_role_mentioned(message) is True


def test_role_ping_is_admitted_and_reaches_the_agent():
    """The whole ingress path: admission must not veto a role ping aimed at the bot."""
    adapter = _adapter()
    message = _role_ping(f"<@&{HELD_ROLE_ID}> 你在吗？")

    admitted, _ = adapter._discord_message_admission(message, claim=False)
    assert admitted is True


def test_role_token_of_a_held_role_is_stripped_from_the_prompt():
    """The token addressed the bot, not the reader — it must not survive into the message text."""
    adapter = _adapter()
    message = _role_ping(f"<@&{HELD_ROLE_ID}> 你在吗？")

    assert adapter._strip_self_role_mentions(message, message.content) == "你在吗？"


def test_token_of_a_role_the_bot_lacks_is_kept():
    adapter = _adapter()
    message = _role_ping(f"<@&{OTHER_ROLE_ID}> 大家好", bot_role_ids=())

    assert adapter._strip_self_role_mentions(message, message.content) == f"<@&{OTHER_ROLE_ID}> 大家好"


def test_role_ping_does_not_strip_a_user_mention():
    """The user-mention path owns ``<@id>`` stripping; role stripping must not touch it."""
    adapter = _adapter()
    message = _role_ping(f"<@&{HELD_ROLE_ID}> <@{OTHER_ROLE_ID}> 看一下")

    assert adapter._strip_self_role_mentions(message, message.content) == f"<@{OTHER_ROLE_ID}> 看一下"
