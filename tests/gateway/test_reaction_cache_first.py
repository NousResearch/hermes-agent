"""Tests for the PR-D cache-first reaction author lookup in
``gateway.platforms.discord_interactions.handle_inbound_reaction``.

The handler's message-author filter tries the in-memory message cache
(``client._connection._messages`` deque) before falling back to
``channel.fetch_message``. These tests pin that contract:

1. Cache hit, bot-authored        → no fetch_message; resolver reached
2. Cache hit, human-authored      → no fetch_message; counter increments
3. Cache miss, fallback bot       → fetch_message called; resolver reached
4. Cache miss, fallback raises    → early return, no counter increment
5. Cache is None (max_messages=None) → fallback to fetch_message
6. Cache is empty deque           → fallback to fetch_message
7. AttributeError safety          → fallback to fetch_message
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms import discord_interactions
from gateway.platforms.discord_interactions import (
    DiscordInteractionsHandler,
    get_event_counters,
)
from gateway.skill_resolver import SkillEntry


# ── Fixtures and builders ────────────────────────────────────────────────


def _entry(name: str, *triggers: dict) -> SkillEntry:
    return (name, {}, list(triggers))


def _payload(emoji: str = "✅", user_id: int = 42, message_id: int = 200) -> SimpleNamespace:
    p = SimpleNamespace(
        user_id=user_id,
        channel_id=100,
        message_id=message_id,
        guild_id=300,
    )
    p.emoji = emoji
    return p


def _message(message_id: int, author_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=message_id, author=SimpleNamespace(id=author_id))


def _make_handler(
    skills: List[SkillEntry],
    *,
    cache,  # deque | None | "missing" sentinel
    fetch_result=None,
    fetch_raises: BaseException | None = None,
) -> tuple[DiscordInteractionsHandler, MagicMock, AsyncMock]:
    """Build a handler with controlled cache + fetch_message behavior.

    ``cache``:
        - ``deque(...)`` — seeded recent-message cache
        - ``None`` — discord.py default when max_messages=None
        - sentinel string ``"missing"`` — _connection has no _messages attr

    ``fetch_result`` is the Message returned by channel.fetch_message on
    cache miss; ``fetch_raises`` overrides it with an exception.
    """
    adapter = MagicMock()
    adapter._client = MagicMock()
    adapter._client.user = SimpleNamespace(id=1)

    if cache == "missing":
        # _connection exists but has no _messages attribute at all.
        adapter._client._connection = SimpleNamespace()
    else:
        adapter._client._connection = SimpleNamespace(_messages=cache)

    fetch_message = AsyncMock()
    if fetch_raises is not None:
        fetch_message.side_effect = fetch_raises
    else:
        fetch_message.return_value = fetch_result

    channel = SimpleNamespace(fetch_message=fetch_message)
    adapter._client.get_channel = MagicMock(return_value=channel)
    adapter._client.fetch_channel = AsyncMock(return_value=channel)

    adapter.handle_message = AsyncMock()
    adapter.build_source = MagicMock(return_value="<source>")
    handler = DiscordInteractionsHandler(adapter=adapter, skill_provider=lambda: skills)
    return handler, adapter, fetch_message


@pytest.fixture(autouse=True)
def _reset_counters():
    for k in discord_interactions._event_counters:
        discord_interactions._event_counters[k] = 0
    yield
    for k in discord_interactions._event_counters:
        discord_interactions._event_counters[k] = 0


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_bot_authored_skips_fetch_and_reaches_resolver():
    skills = [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    handler, adapter, fetch_message = _make_handler(
        skills,
        cache=deque([_message(message_id=200, author_id=1)]),
    )

    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")

    fetch_message.assert_not_awaited()
    adapter._client.fetch_channel.assert_not_called()
    counters = get_event_counters()
    # resolver matched → invoked counter; author filter never tripped
    assert counters["discord.reactions.invoked"] == 1
    assert counters["discord.reactions.skipped_non_bot_author"] == 0
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_cache_hit_human_authored_skips_fetch_and_increments_counter():
    skills = [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    handler, adapter, fetch_message = _make_handler(
        skills,
        cache=deque([_message(message_id=200, author_id=999)]),  # human author
    )

    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")

    fetch_message.assert_not_awaited()
    counters = get_event_counters()
    assert counters["discord.reactions.skipped_non_bot_author"] == 1
    assert counters["discord.reactions.invoked"] == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_cache_miss_fallback_fetch_succeeds_bot_authored():
    skills = [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    handler, adapter, fetch_message = _make_handler(
        skills,
        cache=deque(),  # empty cache → miss
        fetch_result=_message(message_id=200, author_id=1),
    )

    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")

    fetch_message.assert_awaited_once_with(200)
    counters = get_event_counters()
    assert counters["discord.reactions.invoked"] == 1
    assert counters["discord.reactions.skipped_non_bot_author"] == 0
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_cache_miss_fallback_fetch_raises_returns_without_counter_bump():
    skills = [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    handler, adapter, fetch_message = _make_handler(
        skills,
        cache=deque(),
        fetch_raises=RuntimeError("network down"),
    )

    # Must not raise — fail-safe early return.
    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")

    fetch_message.assert_awaited_once_with(200)
    counters = get_event_counters()
    # Couldn't determine the author → do NOT increment the author-filter counter.
    assert counters["discord.reactions.skipped_non_bot_author"] == 0
    assert counters["discord.reactions.invoked"] == 0
    assert counters["discord.reactions.skipped_no_match"] == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_cache_is_none_falls_through_to_fetch_message():
    skills = [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    handler, adapter, fetch_message = _make_handler(
        skills,
        cache=None,  # discord.py default when max_messages=None
        fetch_result=_message(message_id=200, author_id=1),
    )

    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")

    fetch_message.assert_awaited_once_with(200)
    counters = get_event_counters()
    assert counters["discord.reactions.invoked"] == 1


@pytest.mark.asyncio
async def test_cache_is_empty_deque_falls_through_to_fetch_message():
    skills = [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    handler, adapter, fetch_message = _make_handler(
        skills,
        cache=deque(),
        fetch_result=_message(message_id=200, author_id=1),
    )

    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")

    fetch_message.assert_awaited_once_with(200)
    counters = get_event_counters()
    assert counters["discord.reactions.invoked"] == 1


@pytest.mark.asyncio
async def test_connection_missing_messages_attr_falls_through_safely():
    skills = [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    handler, adapter, fetch_message = _make_handler(
        skills,
        cache="missing",  # _connection has no _messages attribute at all
        fetch_result=_message(message_id=200, author_id=1),
    )

    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")

    fetch_message.assert_awaited_once_with(200)
    counters = get_event_counters()
    assert counters["discord.reactions.invoked"] == 1
