"""Hermes's own auto-thread title rename keeps the pinned session-context prompt.

``chat_name`` keys the pinned prompt, and the title lane renames a new auto-thread between turns 1
and 2. Any other rename still re-renders, including a later restore of Hermes's title.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import plugins.platforms.discord.adapter as discord_platform
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import build_session_context
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.adapter_thread_titles import SemanticThreadRenames

OPENING, TITLE = "what broke?", "Database outage"


class _Thread:
    def __init__(self, parent: SimpleNamespace) -> None:
        self.id, self.name, self.parent, self.parent_id = 800, OPENING, parent, parent.id
        self.guild, self.owner_id, self.archived = parent.guild, 42, False

    async def edit(self, *, name: str, reason: str | None = None) -> None:
        self.name = name


Turn = Callable[[object, int], Awaitable[str]]
Conversation = tuple[DiscordAdapter, SimpleNamespace, _Thread, Turn]


@pytest.fixture
def conversation(monkeypatch: pytest.MonkeyPatch) -> Conversation:
    monkeypatch.setattr(discord_platform.discord, "Thread", _Thread, raising=False)
    monkeypatch.setattr(discord_platform, "DISCORD_AVAILABLE", True)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    parent = SimpleNamespace(id=700, name="ops", topic=None, guild=SimpleNamespace(id=1, name="Hermes Server"))
    thread = _Thread(parent)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999), get_channel=lambda _id: thread)
    adapter._text_batch_delay_seconds = 0
    adapter._discord_history_backfill = lambda: False
    adapter._auto_create_thread = AsyncMock(return_value=thread)
    adapter.handle_message = AsyncMock()
    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})

    async def turn(channel: object, message_id: int) -> str:
        await adapter._handle_message(SimpleNamespace(
            id=message_id, content=OPENING, mentions=[], attachments=[], reference=None,
            created_at=datetime.now(timezone.utc), channel=channel,
            author=SimpleNamespace(id=42, display_name="Alice", name="alice")))
        source = adapter.handle_message.await_args.args[0].source
        return runner._pinned_session_context_prompt(build_session_context(source, config), False, "k")

    return adapter, parent, thread, turn


async def _raw_rename(adapter: DiscordAdapter, name: str) -> None:
    await adapter._on_platform_raw_thread_update(SimpleNamespace(thread_id=800, data={"name": name}))


@pytest.mark.asyncio
@pytest.mark.parametrize("moderator_name", ["Renamed by a moderator", OPENING])
async def test_hermes_title_keeps_the_pin_and_a_moderator_rename_does_not(
    conversation: Conversation, moderator_name: str,
) -> None:
    adapter, parent, thread, turn = conversation
    first = await turn(parent, 100)
    # The title lane's call, as gateway/run_topics.py makes it, and its own gateway event.
    assert await adapter.rename_thread("800", TITLE, only_if_current_name=OPENING)
    await _raw_rename(adapter, TITLE)
    thread.name = OPENING  # a cache that has not caught up with the edit
    assert await turn(thread, 101) == first
    thread.name = TITLE
    assert await turn(thread, 102) == first

    thread.name = moderator_name
    assert f"Hermes Server / #ops / {moderator_name}" in await turn(thread, 103)
    thread.name = TITLE  # the moderator restores Hermes's title: it shows as itself
    assert f"Hermes Server / #ops / {TITLE}" in await turn(thread, 104)


@pytest.mark.asyncio
async def test_a_moderator_round_trip_during_the_edit_retires_the_record(conversation: Conversation) -> None:
    adapter, parent, thread, turn = conversation
    await turn(parent, 100)

    async def edit(*, name: str, reason: str | None = None) -> None:
        # Gateway events overtake the REST response: Hermes's own, then a moderator's round trip.
        for event_name in (name, OPENING, name):
            await _raw_rename(adapter, event_name)
        thread.name = name

    thread.edit = edit
    assert await adapter.rename_thread("800", TITLE, only_if_current_name=OPENING)
    assert f"Hermes Server / #ops / {TITLE}" in await turn(thread, 101)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["failed", "cancelled"])
async def test_an_unfinished_edit_drops_its_record(conversation: Conversation, cancel: bool) -> None:
    adapter, parent, thread, turn = conversation
    await turn(parent, 100)
    started, finish = asyncio.Event(), asyncio.Event()

    async def edit(*, name: str, reason: str | None = None) -> None:
        started.set()
        await finish.wait()
        raise RuntimeError("REST edit failed")

    thread.edit = edit
    attempt = asyncio.create_task(adapter.rename_thread("800", TITLE, only_if_current_name=OPENING))
    await started.wait()
    if cancel:
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attempt
    else:
        finish.set()
        assert not await attempt
    thread.name = TITLE  # chosen by someone else
    assert f"Hermes Server / #ops / {TITLE}" in await turn(thread, 101)


def test_an_unfinished_attempt_keeps_a_newer_attempts_record() -> None:
    renames = SemanticThreadRenames()
    with pytest.raises(RuntimeError), renames.attempt("800", OPENING, TITLE):
        with renames.attempt("800", OPENING, TITLE):
            pass
        raise RuntimeError("REST edit failed")
    assert renames.display_name("800", TITLE) == OPENING
