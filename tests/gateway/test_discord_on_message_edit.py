"""Tests for the Discord ``on_message_edit`` handler / ``_handle_message_edit``.

When a user edits an existing message to add the bot's @-mention, Discord
fires ``MESSAGE_UPDATE`` and the original ``on_message`` handler never sees
it - so without ``_handle_message_edit`` the bot stays silent on a flow
that's natural for users (compose, hit edit, add the mention afterwards).

These tests pin down the routing rules:

  - in a guild channel: forward only when the bot is *newly* mentioned in
    the edit (so a typo-fix on an already-answered message does NOT trigger
    a duplicate reply)
  - in a DM: any genuine content change is treated as a new turn
  - embed-only updates (Discord auto-resolving link previews) are ignored
  - bot-author / system-message / disallowed-user filters mirror
    ``on_message`` exactly
  - dedup is keyed on ``edit:<id>:<edited_at>`` so a RESUME-replay of the
    same edit is suppressed but a second genuine edit re-triggers
"""

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_discord_mock():
    """Install (or augment) a mock ``discord`` module.

    Same pattern as ``test_discord_allowed_mentions.py`` - other test files
    in this dir stub ``discord`` via ``sys.modules.setdefault``, whoever
    imports first wins. We always force the symbols we need on top.
    """
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        # Real discord.py is installed - leave it alone.
        return

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.Client = MagicMock
        discord_mod.File = MagicMock
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.ui = SimpleNamespace(
            View=object,
            button=lambda *a, **k: (lambda fn: fn),
            Button=object,
        )
        discord_mod.ButtonStyle = SimpleNamespace(
            success=1, primary=2, danger=3,
            green=1, blurple=2, red=3, grey=4, secondary=5,
        )
        discord_mod.Color = SimpleNamespace(
            orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4,
        )
        discord_mod.Interaction = object
        discord_mod.Embed = MagicMock
        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        discord_mod.opus = SimpleNamespace(is_loaded=lambda: True)

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)

_ensure_discord_mock()

import discord  # noqa: E402

from gateway.platforms.discord import DiscordAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GuildChannel:
    """Sentinel for any non-DM channel - isinstance(_, DMChannel) is False."""


def _make_dm_channel():
    return discord.DMChannel()


def _make_user(uid=100, *, is_bot=False, name="Alice"):
    return SimpleNamespace(id=uid, bot=is_bot, name=name)


_DEFAULT_EDIT_TIME = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)


def _make_message(
    *,
    msg_id=555,
    content="hi",
    author=None,
    mentions=None,
    channel=None,
    msg_type=None,
    edited_at=_DEFAULT_EDIT_TIME,
):
    return SimpleNamespace(
        id=msg_id,
        content=content,
        author=author if author is not None else _make_user(),
        mentions=list(mentions or []),
        channel=channel if channel is not None else _GuildChannel(),
        type=msg_type if msg_type is not None else discord.MessageType.default,
        edited_at=edited_at,
    )


def _make_adapter(*, bot_user=None, allow_user=True, dedup_seen=None):
    """Build a DiscordAdapter shell with just enough state to drive
    ``_handle_message_edit``. We bypass ``__init__`` because the real one
    requires a token and starts the discord.py client - we only need the
    handler's collaborators here.
    """
    adapter = object.__new__(DiscordAdapter)

    adapter._ready_event = SimpleNamespace(
        is_set=lambda: True,
        wait=AsyncMock(),
    )
    adapter._client = SimpleNamespace(user=bot_user)
    seen = set(dedup_seen or [])

    def _is_dup(key):
        if key in seen:
            return True
        seen.add(key)
        return False

    adapter._dedup = SimpleNamespace(is_duplicate=_is_dup)
    adapter._is_allowed_user = lambda *_a, **_kw: allow_user
    adapter._handle_message = AsyncMock()
    return adapter


@pytest.fixture(autouse=True)
def _isolate_discord_env(monkeypatch):
    """Don't let DISCORD_ALLOW_BOTS leak between tests."""
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)


@pytest.fixture(autouse=True)
def _stable_message_type_and_dm_channel(monkeypatch):
    """Per-test override of ``discord.MessageType`` + ``discord.DMChannel``.

    Other test files in this directory share the same ``sys.modules["discord"]``
    object, and the handler relies on ``MessageType.default`` / ``.reply`` being
    *identity-comparable* (``in`` check) rather than truthy MagicMock auto-attrs.
    We override per test and restore on teardown so we never poison sibling
    test files that pytest-xdist co-schedules onto the same worker.
    """
    discord_mod = sys.modules["discord"]
    saved = {
        "MessageType": getattr(discord_mod, "MessageType", None),
        "DMChannel": getattr(discord_mod, "DMChannel", None),
    }
    discord_mod.MessageType = SimpleNamespace(
        default=object(),
        reply=object(),
        thread_starter_message=object(),
        pins_add=object(),
    )
    if not isinstance(discord_mod.DMChannel, type):
        discord_mod.DMChannel = type("DMChannel", (), {})
    yield
    if saved["MessageType"] is None:
        delattr(discord_mod, "MessageType")
    else:
        discord_mod.MessageType = saved["MessageType"]
    if saved["DMChannel"] is None:
        delattr(discord_mod, "DMChannel")
    else:
        discord_mod.DMChannel = saved["DMChannel"]


# ---------------------------------------------------------------------------
# Routing rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_only_edit_is_ignored():
    """Discord auto-resolves link previews via MESSAGE_UPDATE - same content
    before/after. That's not a user-initiated edit, ignore it."""
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot)
    same = "check https://example.com"
    before = _make_message(content=same, author=user, mentions=[bot])
    after = _make_message(content=same, author=user, mentions=[bot])

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_channel_edit_with_newly_added_mention_triggers_reply():
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot)
    before = _make_message(content="some question", author=user, mentions=[])
    after = _make_message(
        content="<@999> some question", author=user, mentions=[bot]
    )

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_awaited_once_with(after)


@pytest.mark.asyncio
async def test_channel_edit_when_mention_was_already_present_is_ignored():
    """Typo-fix on a message we already answered must NOT duplicate the
    reply. This is the regression we'd hit if we naively forwarded every
    edit."""
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot)
    before = _make_message(content="<@999> origenal", author=user, mentions=[bot])
    after = _make_message(content="<@999> original", author=user, mentions=[bot])

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_channel_edit_that_removes_mention_is_ignored():
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot)
    before = _make_message(content="<@999> nvm", author=user, mentions=[bot])
    after = _make_message(content="nvm", author=user, mentions=[])

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_dm_edit_with_content_change_triggers_reply():
    """In a DM every user message is implicitly addressed to the bot, so
    the mention check is bypassed - only the content-changed gate applies."""
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot)
    dm = _make_dm_channel()
    before = _make_message(content="ple", author=user, channel=dm)
    after = _make_message(content="please help", author=user, channel=dm)

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_awaited_once_with(after)


# ---------------------------------------------------------------------------
# Author / type / allowlist filters - must mirror on_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bots_own_edit_is_ignored():
    bot = _make_user(uid=999)
    adapter = _make_adapter(bot_user=bot)
    before = _make_message(content="a", author=bot)
    after = _make_message(content="b", author=bot)

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_non_default_message_type_is_ignored():
    """Pin notifications, member joins etc. are MessageType != default/reply
    and should never be routed to the LLM."""
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot)
    pin = discord.MessageType.pins_add
    before = _make_message(content="a", author=user, msg_type=pin)
    after = _make_message(content="b", author=user, mentions=[bot], msg_type=pin)

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_disallowed_human_user_is_ignored():
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot, allow_user=False)
    before = _make_message(content="hello", author=user, mentions=[])
    after = _make_message(content="<@999> hello", author=user, mentions=[bot])

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_bot_author_with_allow_bots_none_is_ignored(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    bot = _make_user(uid=999)
    other_bot = _make_user(uid=200, is_bot=True)
    adapter = _make_adapter(bot_user=bot)
    before = _make_message(content="hello", author=other_bot, mentions=[])
    after = _make_message(
        content="<@999> hello", author=other_bot, mentions=[bot]
    )

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_bot_author_with_allow_bots_mentions_passes_when_mentioned(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    bot = _make_user(uid=999)
    other_bot = _make_user(uid=200, is_bot=True)
    adapter = _make_adapter(bot_user=bot)
    before = _make_message(content="hello", author=other_bot, mentions=[])
    after = _make_message(
        content="<@999> hello", author=other_bot, mentions=[bot]
    )

    await adapter._handle_message_edit(before, after)

    adapter._handle_message.assert_awaited_once_with(after)


# ---------------------------------------------------------------------------
# Dedup behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_suppresses_replay_of_same_edit():
    """Discord RESUME after reconnect re-fires events; the
    ``edit:<id>:<edited_at>`` key blocks the duplicate."""
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot)
    edit_time = _DEFAULT_EDIT_TIME

    before = _make_message(content="hello", author=user, mentions=[], edited_at=edit_time)
    after = _make_message(
        content="<@999> hello", author=user, mentions=[bot], edited_at=edit_time
    )

    await adapter._handle_message_edit(before, after)
    await adapter._handle_message_edit(before, after)

    assert adapter._handle_message.await_count == 1


@pytest.mark.asyncio
async def test_dedup_allows_second_genuine_edit_with_later_timestamp():
    """The same DM message edited twice with different ``edited_at`` values
    must trigger two replies. Dedup is keyed on edit time, not message id,
    so genuine subsequent edits get through while RESUME-replays don't."""
    bot = _make_user(uid=999)
    user = _make_user(uid=100)
    adapter = _make_adapter(bot_user=bot)
    dm = _make_dm_channel()
    t1 = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 5, 8, 12, 1, 0, tzinfo=timezone.utc)

    msg_v1 = _make_message(content="ple", author=user, channel=dm, edited_at=t1)
    msg_v2 = _make_message(content="please", author=user, channel=dm, edited_at=t1)
    msg_v3 = _make_message(content="please help", author=user, channel=dm, edited_at=t2)

    await adapter._handle_message_edit(msg_v1, msg_v2)
    await adapter._handle_message_edit(msg_v2, msg_v3)

    assert adapter._handle_message.await_count == 2
