"""Tests for Discord message reactions tied to processing lifecycle hooks."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome, SendResult
from gateway.session import SessionSource, build_session_key


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    return adapter


def _make_event(message_id: str, raw_message) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            chat_type="dm",
            user_id="42",
            user_name="Jezza",
        ),
        raw_message=raw_message,
        message_id=message_id,
    )


def _make_thread_event(message_id: str, raw_message, thread_id: str = "123") -> MessageEvent:
    event = _make_event(message_id, raw_message)
    event.source.chat_type = "thread"
    event.source.thread_id = thread_id
    event.source.chat_id = thread_id
    return event


def _activity_thread(adapter, title="Investigate flaky build"):
    thread = sys.modules["discord"].Thread()
    thread.id = 123
    thread.name = title

    async def edit(*, name, reason):
        thread.name = name

    thread.edit = AsyncMock(side_effect=edit)
    adapter._client.get_channel = lambda _id: thread
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    raw_message = SimpleNamespace(
        channel=thread,
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    return thread, raw_message


def _distinct_return_activity_thread(adapter, title="Investigate flaky build"):
    """Model discord.py Thread.edit returning a fresh, authoritative object."""
    thread_type = sys.modules["discord"].Thread
    server = SimpleNamespace(name=title, edits=[])

    def make_thread(name):
        thread = thread_type()
        thread.id = 123
        thread.name = name

        async def edit(*, name, reason):
            server.name = name
            server.edits.append(name)
            return make_thread(name)

        thread.edit = AsyncMock(side_effect=edit)
        return thread

    cached_thread = make_thread(title)
    adapter._client.get_channel = lambda _id: cached_thread
    adapter._client.fetch_channel = AsyncMock(side_effect=lambda _id: make_thread(server.name))
    raw_message = SimpleNamespace(
        channel=cached_thread,
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    return cached_thread, raw_message, server


@pytest.mark.asyncio
async def test_process_message_background_adds_and_swaps_reactions(adapter):
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("1", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    assert raw_message.add_reaction.await_args_list[0].args == ("👀",)
    assert raw_message.remove_reaction.await_args_list[0].args == ("👀", adapter._client.user)
    assert raw_message.add_reaction.await_args_list[1].args == ("✅",)


@pytest.mark.asyncio
async def test_reactions_disabled_via_env(adapter, monkeypatch):
    """When DISCORD_REACTIONS=false, no reactions should be added."""
    monkeypatch.setenv("DISCORD_REACTIONS", "false")

    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("4", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    raw_message.add_reaction.assert_not_awaited()
    raw_message.remove_reaction.assert_not_awaited()
    # Response should still be sent
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_thread_activity_indicator_marks_start_and_restores_on_success(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread = sys.modules["discord"].Thread()
    thread.id = 123
    thread.name = "Investigate flaky build"

    async def edit(*, name, reason):
        thread.name = name

    thread.edit = AsyncMock(side_effect=edit)
    adapter._client.get_channel = lambda _id: thread
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    raw_message = SimpleNamespace(
        channel=thread,
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    event = _make_thread_event("5", raw_message)

    await adapter.on_processing_start(event)
    assert thread.name == "⏳ Investigate flaky build"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert thread.name == "Investigate flaky build"


@pytest.mark.asyncio
async def test_thread_activity_indicator_marks_failure(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread = sys.modules["discord"].Thread()
    thread.id = 123
    thread.name = "Investigate flaky build"

    async def edit(*, name, reason):
        thread.name = name

    thread.edit = AsyncMock(side_effect=edit)
    adapter._client.get_channel = lambda _id: thread
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    event = _make_thread_event(
        "6",
        SimpleNamespace(
            channel=thread,
            add_reaction=AsyncMock(),
            remove_reaction=AsyncMock(),
        ),
    )

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert thread.name == "⛔ Investigate flaky build"


@pytest.mark.asyncio
async def test_thread_activity_indicator_waits_for_concurrent_turns_and_keeps_failure(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread = sys.modules["discord"].Thread()
    thread.id = 123
    thread.name = "Investigate flaky build"

    async def edit(*, name, reason):
        thread.name = name

    thread.edit = AsyncMock(side_effect=edit)
    adapter._client.get_channel = lambda _id: thread
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    raw_message = SimpleNamespace(
        channel=thread,
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    first = _make_thread_event("8", raw_message)
    second = _make_thread_event("9", raw_message)

    await adapter.on_processing_start(first)
    await adapter.on_processing_start(second)
    await adapter.on_processing_complete(first, ProcessingOutcome.SUCCESS)
    assert thread.name == "⏳ Investigate flaky build"

    await adapter.on_processing_complete(second, ProcessingOutcome.FAILURE)
    assert thread.name == "⛔ Investigate flaky build"
    assert adapter._thread_activity_states == {}


@pytest.mark.asyncio
async def test_semantic_thread_rename_updates_active_base_title(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread = sys.modules["discord"].Thread()
    thread.id = 123
    thread.name = "Investigate flaky build"

    async def edit(*, name, reason):
        thread.name = name

    thread.edit = AsyncMock(side_effect=edit)
    adapter._client.get_channel = lambda _id: thread
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    event = _make_thread_event(
        "10",
        SimpleNamespace(
            channel=thread,
            add_reaction=AsyncMock(),
            remove_reaction=AsyncMock(),
        ),
    )

    await adapter.on_processing_start(event)
    renamed = await adapter.rename_thread(
        "123",
        "Stable build",
        only_if_current_name="Investigate flaky build",
    )

    assert renamed is True
    assert thread.name == "⏳ Stable build"
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert thread.name == "Stable build"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, expected_title",
    [
        (ProcessingOutcome.SUCCESS, "Investigate flaky build"),
        (ProcessingOutcome.CANCELLED, "Investigate flaky build"),
        (ProcessingOutcome.FAILURE, "⛔ Investigate flaky build"),
    ],
)
async def test_thread_activity_indicator_finalizes_when_edit_returns_distinct_thread(
    adapter, monkeypatch, outcome, expected_title
):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    cached_thread, raw_message, server = _distinct_return_activity_thread(adapter)
    event = _make_thread_event("distinct-final", raw_message)

    await adapter.on_processing_start(event)
    assert cached_thread.name == "Investigate flaky build"
    assert server.name == "⏳ Investigate flaky build"

    await adapter.on_processing_complete(event, outcome)

    assert server.name == expected_title
    assert server.edits == ["⏳ Investigate flaky build", expected_title]


@pytest.mark.asyncio
async def test_semantic_thread_rename_uses_distinct_edit_result(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    cached_thread, raw_message, server = _distinct_return_activity_thread(adapter)
    event = _make_thread_event("distinct-semantic", raw_message)

    await adapter.on_processing_start(event)
    assert cached_thread.name == "Investigate flaky build"

    assert await adapter.rename_thread(
        "123",
        "Stable build",
        only_if_current_name="Investigate flaky build",
    ) is True
    assert server.name == "⏳ Stable build"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert server.name == "Stable build"


@pytest.mark.asyncio
async def test_thread_activity_indicator_fetches_before_preserving_human_rename(
    adapter, monkeypatch
):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    cached_thread, raw_message, server = _distinct_return_activity_thread(adapter)
    event = _make_thread_event("distinct-human", raw_message)

    await adapter.on_processing_start(event)
    cached_thread.name = "⏳ Investigate flaky build"
    server.name = "Human-chosen title"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert server.name == "Human-chosen title"
    assert server.edits == ["⏳ Investigate flaky build"]


@pytest.mark.asyncio
async def test_semantic_thread_rename_keeps_base_within_utf16_budget(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread, raw_message = _activity_thread(adapter)
    event = _make_thread_event("16", raw_message)

    await adapter.on_processing_start(event)
    assert await adapter.rename_thread("123", "😀" * 80) is True
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert len(thread.name.encode("utf-16-le")) // 2 <= 80
    assert all(
        len(call.kwargs["name"].encode("utf-16-le")) // 2 <= 80
        for call in thread.edit.await_args_list
    )


@pytest.mark.asyncio
async def test_thread_activity_indicator_preserves_human_rename_on_success(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread, raw_message = _activity_thread(adapter)
    event = _make_thread_event("11", raw_message)

    await adapter.on_processing_start(event)
    thread.name = "Human-chosen title"
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert thread.name == "Human-chosen title"
    assert adapter._thread_activity_states == {}


@pytest.mark.asyncio
async def test_thread_activity_indicator_restores_on_cancel(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread, raw_message = _activity_thread(adapter)
    event = _make_thread_event("12", raw_message)

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    assert thread.name == "Investigate flaky build"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,thread_id", [(False, "123"), (True, None)])
async def test_thread_activity_indicator_is_noop_when_disabled_or_not_a_thread(
    adapter, monkeypatch, enabled, thread_id
):
    monkeypatch.setenv(
        "DISCORD_THREAD_ACTIVITY_INDICATOR",
        "true" if enabled else "false",
    )
    thread, raw_message = _activity_thread(adapter)
    event = _make_thread_event("13", raw_message)
    event.source.thread_id = thread_id

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert thread.name == "Investigate flaky build"
    thread.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_thread_activity_indicator_rename_failure_is_fail_open(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread, raw_message = _activity_thread(adapter)
    thread.edit.side_effect = PermissionError("Manage Threads required")
    event = _make_thread_event("14", raw_message)

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert thread.name == "Investigate flaky build"
    assert adapter._thread_activity_states == {}


def test_thread_activity_indicator_title_respects_utf16_budget():
    title = DiscordAdapter._thread_activity_title("⏳ ", "😀" * 80)

    assert len(title.encode("utf-16-le")) // 2 <= 80
    assert title.endswith("😀")


@pytest.mark.asyncio
async def test_thread_activity_indicator_state_is_bounded_when_thread_resolution_fails(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread, raw_message = _activity_thread(adapter)
    event = _make_thread_event("15", raw_message)

    await adapter.on_processing_start(event)
    raw_message.channel = None
    adapter._client.get_channel = lambda _id: None
    adapter._client.fetch_channel = AsyncMock(side_effect=RuntimeError("gone"))
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert adapter._thread_activity_states == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_prefix", ["⏳ ", "⛔ "])
async def test_thread_activity_indicator_recovers_stale_prefix(adapter, monkeypatch, stale_prefix):
    monkeypatch.setenv("DISCORD_THREAD_ACTIVITY_INDICATOR", "true")
    thread = sys.modules["discord"].Thread()
    thread.id = 123
    thread.name = f"{stale_prefix}Investigate flaky build"

    async def edit(*, name, reason):
        thread.name = name

    thread.edit = AsyncMock(side_effect=edit)
    adapter._client.get_channel = lambda _id: thread
    adapter._client.fetch_channel = AsyncMock(return_value=thread)
    event = _make_thread_event(
        "7",
        SimpleNamespace(
            channel=thread,
            add_reaction=AsyncMock(),
            remove_reaction=AsyncMock(),
        ),
    )

    await adapter.on_processing_start(event)
    assert thread.name == "⏳ Investigate flaky build"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert thread.name == "Investigate flaky build"


