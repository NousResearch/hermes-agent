"""Discord ``ignore_other_user_mentions``: free-response channels stay silent on messages
addressed to someone else.

When enabled, a channel message that @mentions another user/bot without also mentioning the
bot is dropped, so a free-response channel does not turn the bot into an answerer for other
people's conversations. Slack parity (``slack.ignore_other_user_mentions``), opt-in: default
False leaves free-response channels fully free-response.

Helper-level tests exercise the real gate helpers; the integration test drives the real
``DiscordAdapter._handle_message`` so the gate is verified end-to-end rather than against a
re-implementation.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
OTHER_USER_ID = 820
CHANNEL_ID = "1527306382653522010"


def _adapter(**extra) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra=dict(extra))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._discord_free_response_channels = Mock(return_value={CHANNEL_ID})
    adapter._discord_channel_keys = Mock(return_value={CHANNEL_ID})
    adapter._discord_require_mention = Mock(return_value=True)
    return adapter


def _channel():
    # ``discord`` is a stub in the test env, so build the channel as a plain namespace.
    return SimpleNamespace(
        id=int(CHANNEL_ID), parent_id=None, name="koenigsbrunn",
        __class__=discord.TextChannel if isinstance(discord.TextChannel, type) else object,
    )


def _message(content, *, mentions=(), channel=None):
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=42, bot=False, display_name="human", name="human"),
        channel=channel or _channel(),
        content=content,
        mentions=list(mentions),
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
        attachments=[],
        reference=None,
        stickers=[],
        embeds=[],
        created_at=None,
        edited_at=None,
        jump_url="https://discord.com/channels/1/1/123",
    )


# ---------------------------------------------------------------------------
# _discord_ignore_other_user_mentions() default
# ---------------------------------------------------------------------------

def test_ignore_other_user_mentions_defaults_off(monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_OTHER_USER_MENTIONS", raising=False)
    assert _adapter()._discord_ignore_other_user_mentions() is False


def test_ignore_other_user_mentions_reads_extra(monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_OTHER_USER_MENTIONS", raising=False)
    assert _adapter(ignore_other_user_mentions=True)._discord_ignore_other_user_mentions() is True


# ---------------------------------------------------------------------------
# _discord_message_addressed_to_other_user()
# ---------------------------------------------------------------------------

def _addressed(content, *, mentions=()):
    return _adapter()._discord_message_addressed_to_other_user(
        _message(content, mentions=mentions), str(BOT_ID))


def test_addressed_empty_and_untagged_are_not_addressed():
    assert _addressed("") is False
    assert _addressed("just a normal line") is False


def test_addressed_other_user_anywhere_in_line():
    # Discord tags people mid-sentence, unlike Slack's leading-token rule.
    assert _addressed(f"can you take this <@{OTHER_USER_ID}>") is True


def test_addressed_legacy_bang_form():
    assert _addressed(f"<@!{OTHER_USER_ID}> ping") is True


def test_addressed_self_mention_is_not_addressed_to_another():
    assert _addressed(f"<@{BOT_ID}> help me") is False


def test_addressed_both_self_and_other_is_not_addressed_to_another():
    assert _addressed(f"<@{OTHER_USER_ID}> and <@{BOT_ID}> compare") is False


def test_addressed_id_prefix_is_not_a_self_match():
    # A user whose ID merely starts with ours is a different person.
    assert _addressed(f"<@{BOT_ID}0> ping") is True


def test_addressed_everyone_is_a_room_broadcast_not_a_person():
    assert _addressed("@everyone standup in 5") is False


def test_addressed_falls_back_to_message_mentions():
    # Some ingress paths populate mentions without the raw token.
    assert _addressed(
        "ping", mentions=[SimpleNamespace(id=OTHER_USER_ID, bot=False)]) is True


# ---------------------------------------------------------------------------
# Integration: real _handle_message
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_OTHER_USER_MENTIONS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)


async def _run(adapter, message):
    adapter._self_is_explicitly_mentioned = lambda m: str(BOT_ID) in adapter._raw_mentioned_user_ids(m)
    adapter._is_bot_tag_debounce_continuation = lambda m: False
    adapter._in_bot_thread = lambda m: False
    adapter._get_allowed_channels = lambda: set()
    adapter._get_ignored_channels = lambda: set()
    adapter._get_no_thread_channels = lambda: set()
    adapter._voice_text_channels = {}
    adapter._is_thread = lambda m: False
    # auto_thread off (the gate under test runs before the auto-thread branch), but keep the
    # real flag getter for the option under test.
    _real_flag = DiscordAdapter._extra_or_env_flag
    adapter._extra_or_env_flag = lambda key, env, default, truthy=False: (
        _real_flag(adapter, key, env, default, truthy=truthy)
        if key == "ignore_other_user_mentions" else False
    )
    adapter.handle_message = AsyncMock()
    adapter._reactions = False
    await adapter._handle_message(message)
    return adapter.handle_message.await_count


@pytest.mark.asyncio
async def test_free_response_answers_untagged_message_when_gate_off():
    adapter = _adapter()
    assert await _run(adapter, _message("no tag here")) == 1


@pytest.mark.asyncio
async def test_free_response_answers_untagged_message_when_gate_on():
    adapter = _adapter(ignore_other_user_mentions=True)
    assert await _run(adapter, _message("no tag here")) == 1


@pytest.mark.asyncio
async def test_gate_on_drops_message_addressed_to_another_user():
    adapter = _adapter(ignore_other_user_mentions=True)
    assert await _run(adapter, _message(f"<@{OTHER_USER_ID}> take a look")) == 0


@pytest.mark.asyncio
async def test_gate_off_still_answers_message_addressed_to_another_user():
    adapter = _adapter()
    assert await _run(adapter, _message(f"<@{OTHER_USER_ID}> take a look")) == 1


@pytest.mark.asyncio
async def test_gate_on_answers_when_bot_is_also_tagged():
    adapter = _adapter(ignore_other_user_mentions=True)
    assert await _run(
        adapter, _message(f"<@{OTHER_USER_ID}> and <@{BOT_ID}> compare")) == 1
