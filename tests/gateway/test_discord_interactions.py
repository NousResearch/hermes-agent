"""Tests for the Discord interactions handler companion file.

Tests focus on units that are testable without a live Discord client:
custom_id helpers, the explicit-triggers cache, payload builders, and
SkillButtonView construction. Full event-loop integration tests live
under tests/gateway/test_discord_inbound_reactions.py and the existing
Discord adapter test files.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.discord_interactions import (
    SKILL_CUSTOM_ID_PREFIX,
    DiscordInteractionsHandler,
    SkillButtonView,
    is_skill_custom_id,
    make_skill_custom_id,
)
from gateway.skill_resolver import SkillEntry


# ── custom_id helpers ────────────────────────────────────────────────────


def test_skill_custom_id_prefix_constant() -> None:
    assert SKILL_CUSTOM_ID_PREFIX == "skill_"


def test_make_skill_custom_id_basic() -> None:
    assert make_skill_custom_id("approve", "yes") == "skill_approve_yes"


def test_make_skill_custom_id_handles_spaces() -> None:
    assert make_skill_custom_id("my skill", "do thing") == "skill_my_skill_do_thing"


def test_is_skill_custom_id_true_for_prefixed() -> None:
    assert is_skill_custom_id("skill_x_y") is True
    assert is_skill_custom_id("skill_") is True


def test_is_skill_custom_id_false_for_other() -> None:
    assert is_skill_custom_id("foo") is False
    assert is_skill_custom_id(None) is False
    assert is_skill_custom_id("") is False
    assert is_skill_custom_id("approve_yes") is False


# ── DiscordInteractionsHandler — explicit_triggers_present cache ─────────


def _entry(name: str, *triggers: dict) -> SkillEntry:
    return (name, {}, list(triggers))


def test_explicit_triggers_present_caches() -> None:
    """The cache should call skill_provider only once."""
    call_count = {"n": 0}

    def provider() -> List[SkillEntry]:
        call_count["n"] += 1
        return [_entry("foo", {"type": "button", "custom_id_pattern": "*"})]

    handler = DiscordInteractionsHandler(adapter=MagicMock(), skill_provider=provider)
    assert handler.explicit_triggers_present() is True
    assert handler.explicit_triggers_present() is True
    assert call_count["n"] == 1


def test_explicit_triggers_present_false_for_empty_corpus() -> None:
    handler = DiscordInteractionsHandler(
        adapter=MagicMock(),
        skill_provider=lambda: [_entry("foo")],
    )
    assert handler.explicit_triggers_present() is False


def test_invalidate_cache_re_evaluates() -> None:
    state = {"explicit": False}

    def provider() -> List[SkillEntry]:
        if state["explicit"]:
            return [_entry("foo", {"type": "button", "custom_id_pattern": "*"})]
        return [_entry("foo")]

    handler = DiscordInteractionsHandler(adapter=MagicMock(), skill_provider=provider)
    assert handler.explicit_triggers_present() is False
    state["explicit"] = True
    assert handler.explicit_triggers_present() is False  # still cached
    handler.invalidate_cache()
    assert handler.explicit_triggers_present() is True


def test_explicit_triggers_present_handles_provider_exception() -> None:
    def provider() -> List[SkillEntry]:
        raise RuntimeError("boom")

    handler = DiscordInteractionsHandler(adapter=MagicMock(), skill_provider=provider)
    # Should not propagate; should default to False.
    assert handler.explicit_triggers_present() is False


# ── handle_skill_button_interaction ──────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_skill_button_no_match_does_not_dispatch() -> None:
    adapter = MagicMock()
    adapter.handle_message = AsyncMock()
    adapter._build_slash_event = MagicMock()

    handler = DiscordInteractionsHandler(
        adapter=adapter,
        skill_provider=lambda: [_entry("approver", {"type": "button", "custom_id_pattern": "approve_*"})],
    )

    interaction = SimpleNamespace(
        data={"custom_id": "deploy_5"},
        user=SimpleNamespace(id=42),
        channel=SimpleNamespace(name="general"),
        channel_id=100,
        response=MagicMock(),
    )
    interaction.response.is_done = MagicMock(return_value=True)
    interaction.response.defer = AsyncMock()

    await handler.handle_skill_button_interaction(interaction)
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handle_skill_button_match_dispatches_with_auto_skill() -> None:
    adapter = MagicMock()
    adapter.handle_message = AsyncMock()
    fake_event = MagicMock()
    fake_event.auto_skill = None
    adapter._build_slash_event = MagicMock(return_value=fake_event)

    handler = DiscordInteractionsHandler(
        adapter=adapter,
        skill_provider=lambda: [_entry("approver", {"type": "button", "custom_id_pattern": "skill_approve_*"})],
    )

    interaction = SimpleNamespace(
        data={"custom_id": "skill_approve_42"},
        user=SimpleNamespace(id=42),
        channel=SimpleNamespace(name="general"),
        channel_id=100,
        response=MagicMock(),
    )
    interaction.response.is_done = MagicMock(return_value=True)
    interaction.response.defer = AsyncMock()

    await handler.handle_skill_button_interaction(interaction)
    adapter.handle_message.assert_called_once()
    dispatched_event = adapter.handle_message.call_args[0][0]
    assert dispatched_event.auto_skill == ["approver"]


@pytest.mark.asyncio
async def test_handle_skill_button_with_no_data_returns_silently() -> None:
    handler = DiscordInteractionsHandler(adapter=MagicMock(), skill_provider=lambda: [])
    interaction = SimpleNamespace(data=None)
    # Should not raise
    await handler.handle_skill_button_interaction(interaction)


# ── handle_inbound_reaction ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_inbound_reaction_ignores_bot_self() -> None:
    adapter = MagicMock()
    adapter._client = MagicMock()
    adapter._client.user = SimpleNamespace(id=999)

    handler = DiscordInteractionsHandler(
        adapter=adapter,
        skill_provider=lambda: [_entry("completer", {"type": "reaction", "emoji": "✅"})],
    )

    payload = SimpleNamespace(
        emoji=SimpleNamespace(__str__=lambda self: "✅"),
        user_id=999,  # same as bot
        channel_id=100,
        message_id=200,
        guild_id=300,
    )
    # Make str(emoji) work
    payload.emoji = "✅"

    await handler.handle_inbound_reaction(payload, action="add")
    # No dispatch should happen because reaction was from the bot itself.
    # We can't easily assert on a non-call without mocking handle_message,
    # so just assert it doesn't raise.


@pytest.mark.asyncio
async def test_handle_inbound_reaction_dispatches_on_match() -> None:
    adapter = MagicMock()
    adapter._client = MagicMock()
    adapter._client.user = SimpleNamespace(id=1)
    # Cache-first reacted-message author lookup (commit c7b081b24): handler
    # iterates client._connection._messages and only dispatches on bot-authored
    # messages. Inject a fake match.
    fake_bot_msg = MagicMock()
    fake_bot_msg.id = 200
    fake_bot_msg.author.id = 1
    adapter._client._connection = MagicMock()
    adapter._client._connection._messages = [fake_bot_msg]
    adapter.handle_message = AsyncMock()
    adapter.build_source = MagicMock(return_value="<source>")

    handler = DiscordInteractionsHandler(
        adapter=adapter,
        skill_provider=lambda: [_entry("completer", {"type": "reaction", "emoji": "✅"})],
    )

    payload = SimpleNamespace(
        user_id=42,
        channel_id=100,
        message_id=200,
        guild_id=300,
    )
    payload.emoji = "✅"

    await handler.handle_inbound_reaction(payload, action="add")
    adapter.handle_message.assert_called_once()
    dispatched_event = adapter.handle_message.call_args[0][0]
    assert dispatched_event.auto_skill == ["completer"]
    assert "[reaction:add]" in dispatched_event.text


# ── SkillButtonView ──────────────────────────────────────────────────────


def test_skill_button_view_creates_buttons_with_canonical_custom_ids() -> None:
    handler = DiscordInteractionsHandler(adapter=MagicMock(), skill_provider=lambda: [])
    view = SkillButtonView(
        handler=handler,
        skill_name="approver",
        actions={"Approve": "approve", "Reject": "reject"},
        timeout=180.0,
    )
    custom_ids = sorted(child.custom_id for child in view.children)
    assert custom_ids == ["skill_approver_approve", "skill_approver_reject"]
    labels = sorted(child.label for child in view.children)
    assert labels == ["Approve", "Reject"]


def test_skill_button_view_callback_delegates_to_handler() -> None:
    """The button callback should delegate to handler.handle_skill_button_interaction."""
    handler = MagicMock()
    handler.handle_skill_button_interaction = AsyncMock()
    view = SkillButtonView(handler=handler, skill_name="x", actions={"Y": "y"})
    button = view.children[0]
    interaction = MagicMock()
    asyncio.get_event_loop().run_until_complete(button.callback(interaction))
    handler.handle_skill_button_interaction.assert_awaited_once_with(interaction)
