"""A mention of the bot's integration-managed role counts as a mention of the bot.

Discord lists a bot's managed role next to the bot in the @-autocomplete, so a human who picks
"@BotName" can send ``<@&role>`` instead of ``<@bot>``; the gateway used to drop those messages
as "no mention". Only the bot's *own* managed role counts: other roles and other bots' roles don't.
"""

from types import SimpleNamespace

import discord

from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
OWN_ROLE = 555
OTHER_BOT_ROLE = 556
PLAIN_ROLE = 557


def _adapter():
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    return adapter


def _role(role_id, *, managed, bot_id=None):
    tags = SimpleNamespace(bot_id=bot_id) if managed else None
    return SimpleNamespace(id=role_id, managed=managed, tags=tags)


def _message(content, *, role_mentions=()):
    me = SimpleNamespace(roles=[
        _role(1, managed=False),
        _role(OWN_ROLE, managed=True, bot_id=BOT_ID),
    ])
    return SimpleNamespace(
        content=content, mentions=[], role_mentions=list(role_mentions),
        guild=SimpleNamespace(id=1, me=me), channel=SimpleNamespace(id=7),
        type=discord.MessageType.default,
    )


def test_own_managed_role_mention_counts_as_self_mention():
    msg = _message(f"<@&{OWN_ROLE}> hi", role_mentions=[SimpleNamespace(id=OWN_ROLE)])
    assert _adapter()._self_is_explicitly_mentioned(msg) is True


def test_raw_role_token_counts_even_when_role_mentions_is_not_populated():
    assert _adapter()._self_is_explicitly_mentioned(_message(f"<@&{OWN_ROLE}> hi")) is True


def test_other_bots_managed_role_does_not_count():
    msg = _message(f"<@&{OTHER_BOT_ROLE}> hi", role_mentions=[SimpleNamespace(id=OTHER_BOT_ROLE)])
    assert _adapter()._self_is_explicitly_mentioned(msg) is False


def test_plain_role_does_not_count():
    msg = _message(f"<@&{PLAIN_ROLE}> hi", role_mentions=[SimpleNamespace(id=PLAIN_ROLE)])
    assert _adapter()._self_is_explicitly_mentioned(msg) is False


def test_no_mention_at_all_is_not_a_mention():
    assert _adapter()._self_is_explicitly_mentioned(_message("hi")) is False


def test_dm_without_guild_is_safe():
    msg = SimpleNamespace(content="hi", mentions=[], role_mentions=[], guild=None)
    assert _adapter()._self_is_explicitly_mentioned(msg) is False
