"""Focused tests for Telegram dynamic processing-state reactions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import ProcessingOutcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "tool_name", "expected"),
    [
        ("reasoning.available", "_thinking", "🤔"),
        ("tool.completed", "terminal", "🤔"),
        ("tool.failed", "terminal", "🤔"),
        ("tool.started", "terminal", "👨‍💻"),
        ("tool.started", "browser_click", "👨‍💻"),
        ("tool.started", "write_file", "✍"),
        ("tool.started", "web_search", "🤓"),
        ("tool.started", "delegate_task", "🤝"),
        ("tool.started", "unknown_tool", "⚡"),
    ],
)
async def test_processing_activity_maps_state_to_standard_reaction(
    monkeypatch, event_type, tool_name, expected
):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()

    from gateway.session import SessionSource
    source = SessionSource(platform="telegram", chat_id="123", chat_type="private", user_id="42")
    await adapter.on_processing_activity(source, "456", event_type, tool_name)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction=expected,
    )


@pytest.mark.asyncio
async def test_processing_activity_deduplicates_same_reaction(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()

    from gateway.session import SessionSource
    source = SessionSource(platform="telegram", chat_id="123", chat_type="private", user_id="42")
    await adapter.on_processing_activity(source, "456", "tool.started", "terminal")
    await adapter.on_processing_activity(source, "456", "tool.started", "browser_click")

    adapter._bot.set_message_reaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_processing_activity_deduplicates_concurrent_same_reaction(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()

    async def delayed_reaction(**kwargs):
        await asyncio.sleep(0)

    adapter._bot.set_message_reaction = AsyncMock(side_effect=delayed_reaction)

    from gateway.session import SessionSource

    source = SessionSource(
        platform="telegram", chat_id="123", chat_type="private", user_id="42"
    )
    await asyncio.gather(
        adapter.on_processing_activity(source, "456", "tool.started", "terminal"),
        adapter.on_processing_activity(source, "456", "tool.started", "browser_click"),
    )

    adapter._bot.set_message_reaction.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (ProcessingOutcome.SUCCESS, "👍"),
        (ProcessingOutcome.FAILURE, "👎"),
    ],
)
async def test_processing_complete_forgets_terminal_reaction_state(
    monkeypatch, outcome, expected
):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()

    from gateway.session import SessionSource

    source = SessionSource(
        platform="telegram", chat_id="123", chat_type="private", user_id="42"
    )
    event = SimpleNamespace(source=source, message_id="456")

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)

    assert adapter._processing_reactions == {}
    assert adapter._bot.set_message_reaction.await_args_list[-1].kwargs == {
        "chat_id": 123,
        "message_id": 456,
        "reaction": expected,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "terminal_reaction"),
    [
        (ProcessingOutcome.SUCCESS, "👍"),
        (ProcessingOutcome.FAILURE, "👎"),
        (ProcessingOutcome.CANCELLED, None),
    ],
)
async def test_late_activity_cannot_overwrite_terminal_reaction(
    monkeypatch, outcome, terminal_reaction
):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()

    from gateway.session import SessionSource

    source = SessionSource(
        platform="telegram", chat_id="123", chat_type="private", user_id="42"
    )
    event = SimpleNamespace(source=source, message_id="456")

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)
    calls_after_completion = adapter._bot.set_message_reaction.await_count

    await adapter.on_processing_activity(source, "456", "tool.started", "terminal")

    assert adapter._bot.set_message_reaction.await_count == calls_after_completion
    assert (
        adapter._bot.set_message_reaction.await_args_list[-1].kwargs["reaction"]
        == terminal_reaction
    )


@pytest.mark.asyncio
async def test_terminal_reaction_guard_is_bounded(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()

    from gateway.session import SessionSource

    source = SessionSource(
        platform="telegram", chat_id="123", chat_type="private", user_id="42"
    )
    for message_id in range(1025):
        event = SimpleNamespace(source=source, message_id=str(message_id))
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert len(adapter._settled_processing_reactions) == 1024
    assert ("123", "0") not in adapter._settled_processing_reactions


@pytest.mark.asyncio
async def test_new_start_reopens_reaction_state_for_same_message(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()

    from gateway.session import SessionSource

    source = SessionSource(
        platform="telegram", chat_id="123", chat_type="private", user_id="42"
    )
    event = SimpleNamespace(source=source, message_id="456")

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    await adapter.on_processing_start(event)
    await adapter.on_processing_activity(source, "456", "tool.started", "terminal")

    assert adapter._bot.set_message_reaction.await_args_list[-1].kwargs["reaction"] == "👨‍💻"


@pytest.mark.asyncio
async def test_processing_activity_is_gated_by_existing_reactions_setting(monkeypatch):
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()

    from gateway.session import SessionSource
    source = SessionSource(platform="telegram", chat_id="123", chat_type="private", user_id="42")
    await adapter.on_processing_activity(source, "456", "tool.started", "terminal")

    adapter._bot.set_message_reaction.assert_not_awaited()
