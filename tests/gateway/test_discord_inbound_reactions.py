"""Integration tests for Discord inbound reaction routing via the
unified trigger framework.

Covers the path: ``on_raw_reaction_add/remove`` →
:meth:`gateway.platforms.discord_interactions.DiscordInteractionsHandler.handle_inbound_reaction`
→ :func:`gateway.skill_resolver.resolve_event_skills` → adapter dispatch.

These tests deliberately stay above the discord.py event loop layer; they
exercise the handler's payload building and resolver interaction directly,
which is sufficient to assert the routing contract. End-to-end gateway
integration (live Discord client) is out of scope for unit tests and is
covered by manual verification at PR time.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.discord_interactions import DiscordInteractionsHandler
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


def _make_handler(skills: List[SkillEntry]) -> tuple[DiscordInteractionsHandler, MagicMock]:
    adapter = MagicMock()
    adapter._client = MagicMock()
    adapter._client.user = SimpleNamespace(id=1)
    adapter.handle_message = AsyncMock()
    adapter.build_source = MagicMock(return_value="<source>")
    handler = DiscordInteractionsHandler(adapter=adapter, skill_provider=lambda: skills)
    return handler, adapter


# ── Reaction add path ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reaction_add_dispatches_matched_skill() -> None:
    handler, adapter = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")
    adapter.handle_message.assert_called_once()
    event = adapter.handle_message.call_args[0][0]
    assert event.auto_skill == ["completer"]
    assert "[reaction:add]" in event.text


@pytest.mark.asyncio
async def test_reaction_remove_dispatches_with_remove_action() -> None:
    handler, adapter = _make_handler(
        [_entry("reverser", {"type": "reaction", "emoji": "✅"})]
    )
    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="remove")
    adapter.handle_message.assert_called_once()
    event = adapter.handle_message.call_args[0][0]
    assert event.auto_skill == ["reverser"]
    assert "[reaction:remove]" in event.text


@pytest.mark.asyncio
async def test_reaction_no_match_no_dispatch() -> None:
    handler, adapter = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    await handler.handle_inbound_reaction(_payload(emoji="👍"), action="add")
    adapter.handle_message.assert_not_called()


# ── Self-reaction filter (bot's own emoji adds are ignored) ──────────────


@pytest.mark.asyncio
async def test_bot_self_reaction_is_ignored() -> None:
    handler, adapter = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    payload = _payload(emoji="✅", user_id=1)  # bot's own user_id
    await handler.handle_inbound_reaction(payload, action="add")
    adapter.handle_message.assert_not_called()


# ── Multi-skill matching ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reaction_dispatches_to_all_matched_skills() -> None:
    handler, adapter = _make_handler(
        [
            _entry("a", {"type": "reaction", "emoji": "✅"}),
            _entry("b", {"type": "reaction", "emoji": "✅"}),
            _entry("c", {"type": "reaction", "emoji": "👍"}),
        ]
    )
    await handler.handle_inbound_reaction(_payload(emoji="✅"), action="add")
    adapter.handle_message.assert_called_once()
    event = adapter.handle_message.call_args[0][0]
    assert sorted(event.auto_skill) == ["a", "b"]


# ── Provider exception is non-fatal ───────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_exception_does_not_propagate() -> None:
    adapter = MagicMock()
    adapter._client = MagicMock()
    adapter._client.user = SimpleNamespace(id=1)
    adapter.handle_message = AsyncMock()

    def bad_provider() -> List[SkillEntry]:
        raise RuntimeError("boom")

    handler = DiscordInteractionsHandler(adapter=adapter, skill_provider=bad_provider)
    # Must not raise
    await handler.handle_inbound_reaction(_payload(), action="add")
    adapter.handle_message.assert_not_called()


# ── DM vs guild source typing ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reaction_in_dm_builds_dm_source() -> None:
    handler, adapter = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    payload = _payload(emoji="✅", guild_id=None)  # None → DM
    await handler.handle_inbound_reaction(payload, action="add")
    adapter.build_source.assert_called_once()
    kwargs = adapter.build_source.call_args.kwargs
    assert kwargs["chat_type"] == "dm"


@pytest.mark.asyncio
async def test_reaction_in_guild_builds_group_source() -> None:
    handler, adapter = _make_handler(
        [_entry("completer", {"type": "reaction", "emoji": "✅"})]
    )
    payload = _payload(emoji="✅", guild_id=300)  # set → group
    await handler.handle_inbound_reaction(payload, action="add")
    adapter.build_source.assert_called_once()
    kwargs = adapter.build_source.call_args.kwargs
    assert kwargs["chat_type"] == "group"
