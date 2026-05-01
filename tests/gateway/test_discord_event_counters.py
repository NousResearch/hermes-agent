"""Tests for the Phase 0.5 Discord event counters in
``gateway.platforms.discord_interactions``.

Covers:
- counter dict has the expected keys at module load
- ``get_event_counters`` returns an immutable snapshot (dict, not the live ref)
- counters increment correctly on each decision branch of
  ``handle_inbound_reaction`` (skipped_non_bot_author / skipped_no_match /
  invoked)
- mention counters exist as keys but stay zero until PR-B wires them
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


def _entry(name: str, *triggers: dict) -> SkillEntry:
    return (name, {}, list(triggers))


def _payload(emoji: str = "✅", user_id: int = 42, guild_id: int = 300) -> SimpleNamespace:
    p = SimpleNamespace(
        user_id=user_id,
        channel_id=100,
        message_id=200,
        guild_id=guild_id,
    )
    p.emoji = emoji
    return p


def _bot_authored_message(message_id: int = 200, bot_id: int = 1) -> SimpleNamespace:
    """Build a fake Message whose author.id is the bot's user id, so that the
    PR-D message-author filter passes."""
    return SimpleNamespace(id=message_id, author=SimpleNamespace(id=bot_id))


def _make_handler(skills: List[SkillEntry]) -> tuple[DiscordInteractionsHandler, MagicMock]:
    adapter = MagicMock()
    adapter._client = MagicMock()
    adapter._client.user = SimpleNamespace(id=1)
    # PR-D cache-first author filter: seed message cache with a bot-authored
    # message at id=200 so the existing counter tests pass through to the
    # resolver branch (the cache is the zero-cost lookup path).
    adapter._client._connection = SimpleNamespace(
        _messages=deque([_bot_authored_message(message_id=200, bot_id=1)])
    )
    adapter.handle_message = AsyncMock()
    adapter.build_source = MagicMock(return_value="<source>")
    handler = DiscordInteractionsHandler(adapter=adapter, skill_provider=lambda: skills)
    return handler, adapter


@pytest.fixture(autouse=True)
def _reset_counters():
    """Each test starts from a zeroed counter dict."""
    for k in discord_interactions._event_counters:
        discord_interactions._event_counters[k] = 0
    yield
    for k in discord_interactions._event_counters:
        discord_interactions._event_counters[k] = 0


class TestCounterShape:
    def test_expected_keys_present(self):
        counters = get_event_counters()
        assert set(counters.keys()) == {
            "discord.reactions.skipped_non_bot_author",
            "discord.reactions.skipped_no_match",
            "discord.reactions.invoked",
            "discord.mentions.skipped_no_match",
            "discord.mentions.invoked",
        }
        assert all(v == 0 for v in counters.values())

    def test_get_event_counters_returns_snapshot(self):
        snapshot = get_event_counters()
        snapshot["discord.reactions.invoked"] = 999
        # Mutating the returned dict must not affect the live counters.
        assert discord_interactions._event_counters["discord.reactions.invoked"] == 0


# ── handle_inbound_reaction decision branches ────────────────────────────


@pytest.mark.asyncio
async def test_bot_self_reaction_increments_skipped_non_bot_author():
    handler, adapter = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    payload = _payload(emoji="✅", user_id=1)  # bot's own user_id
    await handler.handle_inbound_reaction(payload, action="add")

    counters = get_event_counters()
    assert counters["discord.reactions.skipped_non_bot_author"] == 1
    assert counters["discord.reactions.skipped_no_match"] == 0
    assert counters["discord.reactions.invoked"] == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_no_skill_match_increments_skipped_no_match():
    handler, adapter = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    # Different emoji → resolver returns no match.
    await handler.handle_inbound_reaction(_payload(emoji="👍"), action="add")

    counters = get_event_counters()
    assert counters["discord.reactions.skipped_no_match"] == 1
    assert counters["discord.reactions.skipped_non_bot_author"] == 0
    assert counters["discord.reactions.invoked"] == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_match_dispatches_and_increments_invoked():
    handler, adapter = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")

    counters = get_event_counters()
    assert counters["discord.reactions.invoked"] == 1
    assert counters["discord.reactions.skipped_no_match"] == 0
    assert counters["discord.reactions.skipped_non_bot_author"] == 0
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_counters_accumulate_across_calls():
    handler, _ = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    # 1× self-reaction + 2× no-match + 3× match
    await handler.handle_inbound_reaction(_payload(emoji="✅", user_id=1), "add")
    await handler.handle_inbound_reaction(_payload(emoji="👍"), "add")
    await handler.handle_inbound_reaction(_payload(emoji="🙈"), "add")
    await handler.handle_inbound_reaction(_payload(emoji="✅"), "add")
    await handler.handle_inbound_reaction(_payload(emoji="✅"), "remove")
    await handler.handle_inbound_reaction(_payload(emoji="✅"), "add")

    counters = get_event_counters()
    assert counters["discord.reactions.skipped_non_bot_author"] == 1
    assert counters["discord.reactions.skipped_no_match"] == 2
    assert counters["discord.reactions.invoked"] == 3


class TestMentionCountersUnwired:
    """Mention counters are pre-declared but PR-B will wire them. Until then,
    they MUST stay zero — this test pins the contract."""

    @pytest.mark.asyncio
    async def test_mention_counters_remain_zero_after_reactions(self):
        handler, _ = _make_handler(
            [_entry("completer", {"type": "reaction", "emoji": "✅"})]
        )
        await handler.handle_inbound_reaction(_payload(emoji="✅"), "add")
        await handler.handle_inbound_reaction(_payload(emoji="👍"), "add")
        await handler.handle_inbound_reaction(_payload(emoji="✅", user_id=1), "add")

        counters = get_event_counters()
        assert counters["discord.mentions.skipped_no_match"] == 0
        assert counters["discord.mentions.invoked"] == 0
