from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.discord import adapter as discord_adapter
from plugins.platforms.discord.adapter import DiscordAdapter


class _Channel:
    id = 20
    parent_id = None


class _Guild:
    id = 10


class _Author:
    id = 99
    bot = False


class _Dedup:
    def is_duplicate(self, _message_id):
        return False

    def contains(self, _message_id):
        return False


def _adapter(policy):
    obj = object.__new__(DiscordAdapter)
    obj.config = PlatformConfig(enabled=True, token="x", extra={"scope_policies": policy})
    obj._dedup = _Dedup()
    obj._client = SimpleNamespace(user=SimpleNamespace(id=123, bot=True))
    obj._allowed_role_ids = set()
    obj._get_allow_bots = lambda: "all"
    obj._discord_bots_require_inline_mention = lambda: False
    obj._self_is_explicitly_mentioned = lambda _message: False
    obj._self_is_raw_mentioned = lambda _message: False
    obj._is_allowed_user = lambda *args, **kwargs: True
    obj._warn_if_fail_closed_default = lambda: None
    return obj


def _message(content="hello"):
    return SimpleNamespace(
        id=1,
        author=_Author(),
        guild=_Guild(),
        channel=_Channel(),
        mentions=[],
        content=content,
        type=discord_adapter.discord.MessageType.default,
    )


def test_scoped_require_mention_is_enforced_in_early_admission():
    adapter = _adapter({"version": 1, "guilds": {"10": {"channels": {"20": {"require_mention": True}}}}})
    admitted, _ = adapter._discord_message_admission(_message(), claim=False)
    assert admitted is False


def test_scoped_explicit_false_admits_without_mention_before_legacy_gate():
    adapter = _adapter({"version": 1, "guilds": {"10": {"channels": {"20": {"require_mention": False}}}}})
    admitted, _ = adapter._discord_message_admission(_message(), claim=False)
    assert admitted is True


def test_dm_is_not_affected_by_scope_policy():
    adapter = _adapter({"version": 1, "guilds": {"10": {"channels": {"20": {"allow_humans": False}}}}})
    message = _message()
    message.guild = None
    message.channel = discord_adapter.discord.DMChannel.__new__(discord_adapter.discord.DMChannel)
    admitted, _ = adapter._discord_message_admission(message, claim=False)
    assert admitted is True
