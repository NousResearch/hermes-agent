"""Tests for the Telegram ``MessageReactionHandler`` wiring (issue #27438).

Focuses on the *handler* surface:

* Parsing :class:`telegram.MessageReactionUpdated` into one or more
  :class:`gateway.reactions.ReactionEvent` objects.
* Routing the events through :meth:`BasePlatformAdapter.handle_reaction`.
* Honouring the ``HERMES_REACTION_SIGNALS_ENABLED`` master flag.
* Safe handling of malformed / anonymous updates.

These tests construct a real ``TelegramAdapter`` via ``object.__new__``
(matching the established pattern in ``test_telegram_reactions.py``) so
they exercise the actual method bodies without spinning up PTB.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.reactions import (
    DEFAULT_REACTION_WEIGHTS,
    ReactionEvent,
    ReactionPolarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter():
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    # ``handle_reaction`` is on BasePlatformAdapter; replace with an
    # AsyncMock so we can assert what the Telegram callback dispatched
    # without touching the SQLite store.
    adapter.handle_reaction = AsyncMock()
    return adapter


def _reaction_type_emoji(emoji: str):
    """Stand-in for ``telegram.ReactionTypeEmoji`` -- only the .emoji attribute is read."""
    return SimpleNamespace(emoji=emoji, type="emoji")


def _make_reaction_update(
    *,
    chat_id: int = 42,
    user_id: int = 100,
    target_message_id: int = 7,
    chat_type: str = "private",
    old_reaction: List | None = None,
    new_reaction: List | None = None,
):
    mr = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type=chat_type, title=None),
        user=SimpleNamespace(id=user_id),
        message_id=target_message_id,
        old_reaction=old_reaction or [],
        new_reaction=new_reaction or [],
    )
    return SimpleNamespace(message_reaction=mr)


# ---------------------------------------------------------------------------
# Master-flag enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_noops_when_master_flag_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "false")
    adapter = _make_adapter()
    update = _make_reaction_update(
        new_reaction=[_reaction_type_emoji("\u2764\ufe0f")]
    )
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_runs_when_master_flag_enabled(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    update = _make_reaction_update(
        new_reaction=[_reaction_type_emoji("\u2764\ufe0f")]
    )
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_awaited_once()


# ---------------------------------------------------------------------------
# Set-diff semantics: old/new -> add/remove events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_added_reaction_emits_one_added_event(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    update = _make_reaction_update(
        old_reaction=[],
        new_reaction=[_reaction_type_emoji("\U0001F44D")],
    )
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_awaited_once()
    event = adapter.handle_reaction.call_args.args[0]
    assert isinstance(event, ReactionEvent)
    assert event.added is True
    assert event.emoji == "\U0001F44D"
    assert event.platform == "telegram"
    assert event.channel_id == "42"
    assert event.actor_user_id == "100"
    assert event.target_message_id == "7"
    assert event.signal.label == "thumbs_up"


@pytest.mark.asyncio
async def test_removed_reaction_emits_added_false(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    update = _make_reaction_update(
        old_reaction=[_reaction_type_emoji("\U0001F44D")],
        new_reaction=[],
    )
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_awaited_once()
    event = adapter.handle_reaction.call_args.args[0]
    assert event.added is False


@pytest.mark.asyncio
async def test_swap_emits_one_removed_and_one_added(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    # User changes their mind: 👍 -> ❤️.
    update = _make_reaction_update(
        old_reaction=[_reaction_type_emoji("\U0001F44D")],
        new_reaction=[_reaction_type_emoji("\u2764\ufe0f")],
    )
    await adapter._handle_message_reaction(update, context=None)
    assert adapter.handle_reaction.await_count == 2
    events = [call.args[0] for call in adapter.handle_reaction.call_args_list]
    added = [e for e in events if e.added]
    removed = [e for e in events if not e.added]
    assert len(added) == 1
    assert len(removed) == 1
    assert added[0].emoji == "\u2764\ufe0f"
    assert removed[0].emoji == "\U0001F44D"


@pytest.mark.asyncio
async def test_unchanged_reaction_emits_nothing(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    # Same reaction reported twice -- shouldn't double-record.
    update = _make_reaction_update(
        old_reaction=[_reaction_type_emoji("\U0001F44D")],
        new_reaction=[_reaction_type_emoji("\U0001F44D")],
    )
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_not_awaited()


# ---------------------------------------------------------------------------
# Anonymous / malformed updates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anonymous_group_reaction_dropped(monkeypatch):
    """No user object -> can't attribute the signal -> drop the event.

    Telegram delivers ``MessageReactionUpdated`` without a user for
    anonymous group reactions; we explicitly choose NOT to attribute
    these to "unknown" because that would pollute the per-user signal.
    """
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    update = SimpleNamespace(
        message_reaction=SimpleNamespace(
            chat=SimpleNamespace(id=42, type="supergroup", title="g"),
            user=None,
            message_id=7,
            old_reaction=[],
            new_reaction=[_reaction_type_emoji("\u2764\ufe0f")],
        )
    )
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_message_reaction_attribute_is_noop(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    update = SimpleNamespace(message_reaction=None)
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_swallows_exceptions(monkeypatch):
    """A faulty update must NOT propagate -- adapter logs and moves on."""
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    # Update with a missing chat object will raise AttributeError inside
    # the handler; ensure we don't propagate.
    broken_update = SimpleNamespace(
        message_reaction=SimpleNamespace(
            chat=None,
            user=None,
            message_id=None,
            old_reaction=[],
            new_reaction=[],
        )
    )
    await adapter._handle_message_reaction(broken_update, context=None)
    adapter.handle_reaction.assert_not_awaited()


# ---------------------------------------------------------------------------
# Unknown emoji handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_emoji_dropped_when_include_unknown_off(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    monkeypatch.setenv("HERMES_REACTION_INCLUDE_UNKNOWN", "false")
    adapter = _make_adapter()
    update = _make_reaction_update(
        new_reaction=[_reaction_type_emoji("\U0001F914")],  # 🤔
    )
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_emoji_recorded_when_include_unknown_on(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    monkeypatch.setenv("HERMES_REACTION_INCLUDE_UNKNOWN", "true")
    adapter = _make_adapter()
    update = _make_reaction_update(
        new_reaction=[_reaction_type_emoji("\U0001F914")],
    )
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_awaited_once()
    event = adapter.handle_reaction.call_args.args[0]
    assert event.signal.polarity is ReactionPolarity.NEUTRAL


@pytest.mark.asyncio
async def test_custom_emoji_without_emoji_attribute_is_skipped(monkeypatch):
    """``ReactionTypeCustomEmoji`` has no ``.emoji`` -> ignored.

    Telegram custom emojis aren't in our default weights table.  We
    deliberately skip them in v1 rather than guess.
    """
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    custom = SimpleNamespace(type="custom_emoji", custom_emoji_id="abc")
    # Note: SimpleNamespace would auto-return None for missing attrs
    # only when explicitly requested; we don't add an `emoji` attr.
    update = _make_reaction_update(new_reaction=[custom])
    await adapter._handle_message_reaction(update, context=None)
    adapter.handle_reaction.assert_not_awaited()


# ---------------------------------------------------------------------------
# Platform-data propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_type_propagated_to_platform_data(monkeypatch):
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    update = _make_reaction_update(
        chat_type="supergroup",
        new_reaction=[_reaction_type_emoji("\u2764\ufe0f")],
    )
    await adapter._handle_message_reaction(update, context=None)
    event = adapter.handle_reaction.call_args.args[0]
    assert event.platform_data["chat_type"] == "supergroup"


@pytest.mark.asyncio
async def test_multiple_added_emoji_emit_multiple_events(monkeypatch):
    """A single update can add several reactions in one call."""
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    adapter = _make_adapter()
    update = _make_reaction_update(
        new_reaction=[
            _reaction_type_emoji("\u2764\ufe0f"),
            _reaction_type_emoji("\U0001F44D"),
        ],
    )
    await adapter._handle_message_reaction(update, context=None)
    assert adapter.handle_reaction.await_count == 2
    emoji_set = {
        call.args[0].emoji for call in adapter.handle_reaction.call_args_list
    }
    assert emoji_set == {"\u2764\ufe0f", "\U0001F44D"}


# ---------------------------------------------------------------------------
# Integration: handle_reaction default path persists when flag enabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_reaction_persists_when_enabled(tmp_path, monkeypatch):
    """End-to-end: PTB update -> _handle_message_reaction -> SQLite row."""
    from gateway.reaction_store import ReactionStore, reset_reaction_store_for_tests
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "true")
    reset_reaction_store_for_tests()

    from gateway.platforms.telegram import TelegramAdapter
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    # Don't mock handle_reaction here -- we want the default impl to run.

    update = _make_reaction_update(
        new_reaction=[_reaction_type_emoji("\u2764\ufe0f")]
    )
    await adapter._handle_message_reaction(update, context=None)

    db_path = tmp_path / "reactions.db"
    assert db_path.exists()
    store = ReactionStore(db_path=db_path)
    assert store.count() == 1
    rows = store.recent_for_user(platform="telegram", actor_user_id="100")
    assert len(rows) == 1
    assert rows[0]["emoji"] == "\u2764\ufe0f"
    assert rows[0]["weight"] == 2.0  # heart positive
    reset_reaction_store_for_tests()


@pytest.mark.asyncio
async def test_handle_reaction_skips_persistence_when_disabled(tmp_path, monkeypatch):
    """Master-flag check happens in BOTH the parser AND handle_reaction."""
    from gateway.reaction_store import reset_reaction_store_for_tests
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REACTION_SIGNALS_ENABLED", "false")
    reset_reaction_store_for_tests()

    from gateway.platforms.telegram import TelegramAdapter
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")

    update = _make_reaction_update(
        new_reaction=[_reaction_type_emoji("\u2764\ufe0f")]
    )
    await adapter._handle_message_reaction(update, context=None)

    # No DB should have been created at all.
    assert not (tmp_path / "reactions.db").exists()
    reset_reaction_store_for_tests()
