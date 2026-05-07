"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(**extra_env):
    from gateway.platforms.telegram import TelegramAdapter

    extra_env.setdefault("progress_updates", False)
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra_env)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    adapter._bot.send_chat_action = AsyncMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="999"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="999"))
    return adapter


def _make_event(chat_id: str = "123", message_id: str = "456", thread_id: str | None = None) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="private",
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


def test_reactions_enabled_from_platform_config_when_env_unset(monkeypatch):
    """telegram.reactions in PlatformConfig.extra should enable reactions without env bridging."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter(reactions=True)
    assert adapter._reactions_enabled() is True


def test_reactions_env_overrides_platform_config(monkeypatch):
    """Explicit env should retain precedence over PlatformConfig.extra."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter(reactions=True)
    assert adapter._reactions_enabled() is False


# ── typing indicator ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_processing_start_sends_typing_action_in_thread(monkeypatch):
    """Processing start should show Telegram-native typing status in the source topic."""
    monkeypatch.delenv("TELEGRAM_TYPING", raising=False)
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter(typing_refresh=False)
    event = _make_event(thread_id="3220")

    await adapter.on_processing_start(event)

    adapter._bot.send_chat_action.assert_awaited_once_with(
        chat_id=123,
        action="typing",
        message_thread_id=3220,
    )


@pytest.mark.asyncio
async def test_processing_start_typing_is_best_effort(monkeypatch):
    """Typing action failures should not block processing hooks."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter(typing_refresh=False)
    adapter._bot.send_chat_action = AsyncMock(side_effect=RuntimeError("no perms"))

    await adapter.on_processing_start(_make_event())


@pytest.mark.asyncio
async def test_processing_complete_cancels_periodic_typing_refresh(monkeypatch):
    """Long-running processing should get a periodic typing task that is stopped on completion."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter(typing_refresh=True, typing_refresh_interval=30)
    event = _make_event(thread_id="3220")

    await adapter.on_processing_start(event)
    tasks = getattr(adapter, "_typing_indicator_tasks")
    assert len(tasks) == 1
    task = next(iter(tasks.values()))
    assert not task.done()

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert task.cancelled() or task.done()
    assert getattr(adapter, "_typing_indicator_tasks") == {}


@pytest.mark.asyncio
async def test_send_typing_respects_typing_indicator_disabled(monkeypatch):
    """telegram.typing_indicator=false should disable both hook and base keep-typing sends."""
    monkeypatch.delenv("TELEGRAM_TYPING", raising=False)
    adapter = _make_adapter(typing_indicator=False)

    await adapter.send_typing("123", metadata={"thread_id": "3220"})

    adapter._bot.send_chat_action.assert_not_called()


@pytest.mark.asyncio
async def test_progress_updates_send_then_edit_liveness_message(monkeypatch):
    """Long-running Telegram turns should send a durable progress message and edit it on completion."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter(
        typing_refresh=False,
        progress_updates=True,
        progress_initial_delay=0.01,
        progress_interval=30,
    )
    event = _make_event(thread_id="3220")

    await adapter.on_processing_start(event)
    await asyncio.sleep(0.05)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter.send.assert_awaited()
    send_kwargs = adapter.send.await_args.kwargs
    assert send_kwargs["chat_id"] == "123"
    assert send_kwargs["reply_to"] == "456"
    assert send_kwargs["metadata"] == {"thread_id": "3220"}
    assert "작업 중" in send_kwargs["content"]
    adapter.edit_message.assert_awaited()
    assert getattr(adapter, "_progress_indicator_tasks") == {}


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
async def test_on_processing_start_adds_eyes_reaction(monkeypatch):
    """Processing start should add eyes reaction when enabled."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

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
async def test_on_processing_complete_success(monkeypatch):
    """Successful processing should set thumbs-up reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44d",
    )


@pytest.mark.asyncio
async def test_on_processing_complete_failure(monkeypatch):
    """Failed processing should set thumbs-down reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

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
async def test_on_processing_complete_cancelled_keeps_existing_reaction(monkeypatch):
    """Expected cancellation should not replace the in-progress reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    adapter._bot.set_message_reaction.assert_not_awaited()


# ── config.py bridging ───────────────────────────────────────────────


def test_config_bridges_telegram_reactions(monkeypatch, tmp_path):
    """gateway/config.py bridges telegram.reactions to TELEGRAM_REACTIONS env var."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("TELEGRAM_REACTIONS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "true"


def test_config_reactions_env_takes_precedence(monkeypatch, tmp_path):
    """Env var should take precedence over config.yaml for reactions."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "false"
