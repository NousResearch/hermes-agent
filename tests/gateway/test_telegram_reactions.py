"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(**extra_env):
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra_env)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    return adapter


def _make_event(
    chat_id: str = "123",
    message_id: str = "456",
    chat_type: str = "dm",
    thread_id: str | None = None,
) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type=chat_type,
            user_id="42",
            user_name="TestUser",
            thread_id=thread_id,
        ),
        message_id=message_id,
    )


# ── _reactions_enabled ───────────────────────────────────────────────


def test_reactions_disabled_by_default(monkeypatch):
    """Telegram reactions should be disabled by default."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


def test_reactions_enabled_when_set_true(monkeypatch):
    """Setting TELEGRAM_REACTIONS=true enables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is True


def test_reactions_enabled_with_1(monkeypatch):
    """TELEGRAM_REACTIONS=1 enables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "1")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is True


def test_reactions_disabled_with_false(monkeypatch):
    """TELEGRAM_REACTIONS=false disables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


def test_reactions_disabled_with_0(monkeypatch):
    """TELEGRAM_REACTIONS=0 disables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "0")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


def test_reactions_disabled_with_no(monkeypatch):
    """TELEGRAM_REACTIONS=no disables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "no")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


# ── reaction chat types ──────────────────────────────────────────────


def test_reaction_chat_types_can_be_configured_via_extra(monkeypatch):
    monkeypatch.delenv("TELEGRAM_REACTION_CHAT_TYPES", raising=False)
    adapter = _make_adapter(reaction_chat_types=["dm", "group"])

    assert adapter._telegram_reaction_chat_types() == {"dm", "group"}


def test_reaction_chat_types_can_be_configured_via_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTION_CHAT_TYPES", "dm, channel")
    adapter = _make_adapter()

    assert adapter._telegram_reaction_chat_types() == {"dm", "channel"}


def test_reaction_chat_types_env_overrides_extra(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTION_CHAT_TYPES", "channel")
    adapter = _make_adapter(reaction_chat_types=["dm", "group"])

    assert adapter._telegram_reaction_chat_types() == {"channel"}


def test_reaction_chat_types_support_thread_alias_for_forum(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTION_CHAT_TYPES", "thread")
    adapter = _make_adapter()

    assert adapter._telegram_reaction_chat_types() == {"forum"}


def test_reaction_chat_types_do_not_fall_back_to_all_when_only_invalid_values_are_configured(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTION_CHAT_TYPES", "gruop")
    adapter = _make_adapter()

    assert adapter._telegram_reaction_chat_types() == set()


# ── _set_reaction ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_reaction_calls_bot_api(monkeypatch):
    """_set_reaction should call bot.set_message_reaction with correct args."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()

    result = await adapter._set_reaction("123", "456", "\U0001f440")

    assert result is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_set_reaction_returns_false_without_bot(monkeypatch):
    """_set_reaction should return False when bot is not available."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot = None

    result = await adapter._set_reaction("123", "456", "\U0001f440")
    assert result is False


@pytest.mark.asyncio
async def test_set_reaction_handles_api_error_gracefully(monkeypatch):
    """API errors during reaction should not propagate."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(side_effect=RuntimeError("no perms"))

    result = await adapter._set_reaction("123", "456", "\U0001f440")
    assert result is False


# ── on_processing_start ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_start_accepts_private_alias_in_event_chat_type(monkeypatch):
    """Processing start should normalize legacy/private chat_type aliases on events."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_CHAT_TYPES", raising=False)
    adapter = _make_adapter(reaction_chat_types=["dm"])

    await adapter.on_processing_start(_make_event(chat_type="private"))

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_on_processing_start_treats_group_threads_as_forum_for_reaction_filtering(monkeypatch):
    """Threaded group events should match the forum reaction chat type."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_CHAT_TYPES", raising=False)
    adapter = _make_adapter(reaction_chat_types=["forum"])

    await adapter.on_processing_start(_make_event(chat_type="group", thread_id="99"))

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_on_processing_start_respects_configured_chat_types(monkeypatch):
    """Processing start should follow configured reaction chat types."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_CHAT_TYPES", raising=False)
    adapter = _make_adapter(reaction_chat_types=["dm"])

    await adapter.on_processing_start(_make_event(chat_type="dm"))
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_on_processing_start_skips_group_when_not_in_configured_chat_types(monkeypatch):
    """Processing start should skip chats excluded by reaction_chat_types."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_CHAT_TYPES", raising=False)
    adapter = _make_adapter(reaction_chat_types=["dm"])

    await adapter.on_processing_start(_make_event(chat_type="group"))

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_processing_start_adds_eyes_reaction_in_group_chat(monkeypatch):
    """Processing start should add eyes reaction in group chats when enabled."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event(chat_type="group")

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_on_processing_start_adds_eyes_reaction_in_dm_by_default(monkeypatch):
    """Processing start should still react in DMs unless reaction_chat_types narrows it."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_CHAT_TYPES", raising=False)
    adapter = _make_adapter()
    event = _make_event(chat_type="dm")

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_on_processing_start_skipped_when_disabled(monkeypatch):
    """Processing start should not react when reactions are disabled."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_processing_start_handles_missing_ids(monkeypatch):
    """Should handle events without chat_id or message_id gracefully."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SimpleNamespace(chat_id=None),
        message_id=None,
    )

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_not_awaited()


# ── on_processing_complete ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_complete_success_in_group_chat(monkeypatch):
    """Successful processing should set thumbs-up reaction in group chats."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event(chat_type="group")

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44d",
    )


@pytest.mark.asyncio
async def test_on_processing_complete_adds_thumbs_up_in_dm_by_default(monkeypatch):
    """Processing complete should still react in DMs unless reaction_chat_types narrows it."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_CHAT_TYPES", raising=False)
    adapter = _make_adapter()
    event = _make_event(chat_type="dm")

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44d",
    )


@pytest.mark.asyncio
async def test_on_processing_complete_failure(monkeypatch):
    """Failed processing should set thumbs-down reaction in group chats."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event(chat_type="group")

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44e",
    )


@pytest.mark.asyncio
async def test_on_processing_complete_skipped_when_disabled(monkeypatch):
    """Processing complete should not react when reactions are disabled."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_clears_reaction(monkeypatch):
    """Cancelled processing should clear the in-progress reaction.

    Without this clear, the 👀 reaction lingers on the user's message
    indefinitely (until another agent run swaps it for 👍/👎). On a
    ``/stop`` that ends a session, that reaction never gets cleaned up.
    """
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event(chat_type="group")

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    # set_message_reaction with reaction=None clears all reactions on the
    # message (Bot API documented semantics; equivalent to Bot API 10.0's
    # deleteMessageReaction but works on PTB 22.6 already).
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction=None,
    )


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_skipped_when_disabled(monkeypatch):
    """Cancelled processing should not call the API when reactions are off."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_reactions_handles_api_error_gracefully(monkeypatch):
    """API errors during clear should not propagate."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(side_effect=RuntimeError("no perms"))

    result = await adapter._clear_reactions("123", "456")
    assert result is False


@pytest.mark.asyncio
async def test_clear_reactions_returns_false_without_bot(monkeypatch):
    """_clear_reactions should return False when bot is not available."""
    adapter = _make_adapter()
    adapter._bot = None

    result = await adapter._clear_reactions("123", "456")
    assert result is False


# ── config.py bridging ───────────────────────────────────────────────


def test_config_bridges_telegram_reactions(monkeypatch, tmp_path):
    """gateway/config.py bridges telegram reaction settings to env vars and extra config."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reactions": True,
            "reaction_chat_types": ["group", "forum", "channel"],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("TELEGRAM_REACTIONS", "")
    monkeypatch.setenv("TELEGRAM_REACTION_CHAT_TYPES", "")

    from gateway.config import Platform, load_gateway_config
    config = load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "true"
    assert os.getenv("TELEGRAM_REACTION_CHAT_TYPES") == "group,forum,channel"
    assert config.platforms[Platform.TELEGRAM].extra["reaction_chat_types"] == ["group", "forum", "channel"]


def test_config_reaction_chat_types_empty_list_stays_explicit(monkeypatch, tmp_path):
    """An explicit empty reaction_chat_types list should disable reactions rather than broadening them."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reaction_chat_types": [],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_REACTION_CHAT_TYPES", raising=False)

    from gateway.config import Platform, load_gateway_config
    config = load_gateway_config()

    assert config.platforms[Platform.TELEGRAM].extra["reaction_chat_types"] == []


def test_config_reactions_env_takes_precedence(monkeypatch, tmp_path):
    """Env vars should take precedence over config.yaml for reaction settings."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reactions": True,
            "reaction_chat_types": ["group", "forum", "channel"],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv("TELEGRAM_REACTION_CHAT_TYPES", "dm")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "false"
    assert os.getenv("TELEGRAM_REACTION_CHAT_TYPES") == "dm"
