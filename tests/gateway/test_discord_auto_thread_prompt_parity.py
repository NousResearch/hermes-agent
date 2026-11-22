"""An auto-threaded @mention reads the new thread's topic, as the thread's next message does.

The first turn's source pointed at the new thread but took its topic from the parent text channel,
while the thread's next message read the thread's own topic (none outside forums). ``Channel Topic``
is part of the pinned session-context prompt, so turn 2 re-rendered it. Hermes's title rename
between the turns is included, so both inputs that change on turn 2 are covered.
"""
from __future__ import annotations

from datetime import datetime, timezone, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import plugins.platforms.discord.adapter as discord_platform
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import build_session_context
from plugins.platforms.discord.adapter import DiscordAdapter

_BOT = SimpleNamespace(id=999)


class _Thread:
    def __init__(self, parent: SimpleNamespace) -> None:
        self.id, self.name, self.parent, self.parent_id = 800, "what-broke", parent, parent.id
        self.guild = parent.guild
        self.archived = False

    async def edit(self, *, name: str, reason: str | None = None) -> None:
        self.name = name


def _message(channel: object, message_id: int, *, mention: bool) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id, content=("<@999> " if mention else "") + "what broke?",
        mentions=[_BOT] if mention else [], attachments=[], reference=None,
        created_at=datetime.now(UTC), channel=channel,
        author=SimpleNamespace(id=42, display_name="Alice", name="alice"))


@pytest.mark.asyncio
async def test_auto_thread_first_turn_pins_the_threads_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discord_platform.discord, "Thread", _Thread, raising=False)
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    guild = SimpleNamespace(id=1, name="Hermes Server")
    parent = SimpleNamespace(id=700, name="ops", topic="Incident triage", guild=guild)
    thread = _Thread(parent)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._client = SimpleNamespace(user=_BOT, get_channel=lambda _: thread)
    adapter._text_batch_delay_seconds = 0
    adapter._discord_history_backfill = lambda: False
    adapter._auto_create_thread = AsyncMock(return_value=thread)
    adapter.handle_message = AsyncMock()
    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})

    assert not await adapter._handle_message(_message(parent, 99, mention=False))
    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
    prompts = []
    for channel, message_id, mention in ((parent, 100, True), (thread, 101, False)):
        assert await adapter._handle_message(_message(channel, message_id, mention=mention))
        source = adapter.handle_message.await_args.args[0].source
        assert source.chat_id == "800"
        prompts.append(runner._pinned_session_context_prompt(build_session_context(source, config), False, "k"))
        if message_id == 100:
            assert await adapter.rename_thread("800", "Database outage", only_if_current_name="what-broke")
            await adapter._on_platform_raw_thread_update(SimpleNamespace(thread_id=800, data={"name": "Database outage"}))
    adapter._auto_create_thread.assert_awaited_once()
    assert prompts[0] == prompts[1]
