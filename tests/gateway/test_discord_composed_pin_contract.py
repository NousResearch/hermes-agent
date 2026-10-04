"""Composed auto-thread, typed and slash prompt contract (#131243).

Exercise real message handling, auto-thread creation, native slash dispatch, and
runner pins with a collecting message handler and fake Discord transport objects.
Full gateway dispatch and the /plan inbound rewrite are outside this probe.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import plugins.platforms.discord.adapter as discord_platform
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import build_session_context
from plugins.platforms.discord.adapter import DiscordAdapter


class _Text:
    def __init__(self):
        self.id, self.name, self.topic = 700, "ops", "Incident triage"
        self.guild = SimpleNamespace(id=1, name="Hermes Server")


class _Thread:
    def __init__(self, parent, name):
        self.id, self.name, self.parent, self.parent_id = 800, name, parent, parent.id
        self.guild, self.owner_id = parent.guild, 999

    async def edit(self, *, name, reason=None):
        # discord.py edit returns a new object; the cached object changes on a
        # subsequent gateway event, not by mutating this HTTP request's target.
        return _Thread(self.parent, name)


_USER = SimpleNamespace(id=42, display_name="Alice", name="alice", bot=False)
_BOT = SimpleNamespace(id=999, name="Hermes", bot=True)


@pytest.mark.asyncio
async def test_composed_thread_pin_contract(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
    monkeypatch.setattr(discord_platform.discord, "Thread", _Thread)
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
    parent = _Text()
    opening, generated = "what broke?", "Database outage"
    channels = {}
    creates = []

    async def create_thread(*, name, auto_archive_duration):
        creates.append(name)
        channels[800] = _Thread(parent, name)
        return channels[800]

    def message(channel, message_id, mention=False):
        return SimpleNamespace(
            id=message_id, content=("<@999> " if mention else "") + opening,
            mentions=[_BOT] if mention else [], attachments=[], reference=None,
            created_at=datetime.now(timezone.utc), channel=channel, guild=parent.guild,
            author=_USER, create_thread=create_thread,
        )

    platform_config = PlatformConfig(enabled=True, token="fake", typing_indicator=False, extra={
        "auto_thread": True, "history_backfill": False, "reactions": False,
        "channel_prompts": {"700": "Answer in haiku."},
        "channel_skill_bindings": [{"id": "700", "skill": "triage"}],
    })
    adapter = DiscordAdapter(platform_config)
    adapter._client = SimpleNamespace(user=_BOT, get_channel=channels.get)
    adapter._allowed_user_ids = {str(_USER.id)}
    adapter._text_batch_delay_seconds = 0
    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: platform_config})
    received = []

    async def receive(event):
        context = build_session_context(event.source, config)
        channel_prompt, _ = runner._pinned_channel_inputs(
            "contract", event.channel_prompt, event.source, internal=False)
        received.append((event, runner._pinned_session_context_prompt(
            context, False, "contract"), channel_prompt))

    adapter.set_message_handler(receive)

    async def dispatched():
        await asyncio.gather(*tuple(adapter._background_tasks))
        event, prompt, channel_prompt = received[-1]
        assert event.source.chat_id == event.source.thread_id == "800"
        assert event.source.parent_chat_id == "700"
        assert event.channel_prompt == channel_prompt == "Answer in haiku."
        assert event.auto_skill == ["triage"]
        # Ordinary text threads do not inherit their text parent's topic.
        assert parent.topic and not event.source.chat_topic
        return event, prompt

    async def turn(channel, message_id, mention=False):
        count = len(received)
        assert await adapter._handle_message(message(channel, message_id, mention))
        result = await dispatched()
        assert len(received) == count + 1
        return result

    async def rename_event(name):
        before = channels[800]
        raw = SimpleNamespace(thread_id=800, data={"name": name}, thread=before)
        # discord.py dispatches this payload before the mutable cached-thread
        # event. The original implementation has no raw-event subscription.
        raw_handler = getattr(adapter, "_on_platform_raw_thread_update", None)
        if raw_handler is not None:
            await raw_handler(raw)
        after = _Thread(parent, name)
        channels[800] = after
        await adapter._on_platform_thread_update(before, after)

    # With the default mention policy, plain parent traffic never creates a
    # thread or enters the handler. A mention opens the real auto-thread lane.
    assert not await adapter._handle_message(message(parent, 99))
    assert not creates and not received
    _, first = await turn(parent, 100, mention=True)
    assert creates == [opening]
    assert await adapter.rename_thread("800", generated, only_if_current_name=opening)
    await rename_event(generated)

    ordinary, ordinary_prompt = await turn(channels[800], 102)
    assert ordinary.source.chat_name == f"Hermes Server / #ops / {opening}"
    assert ordinary_prompt is first

    interaction = SimpleNamespace(
        channel=channels[800], channel_id=800, guild=parent.guild, guild_id=1, user=_USER,
        response=SimpleNamespace(defer=AsyncMock()), delete_original_response=AsyncMock(),
    )
    count = len(received)
    await adapter._run_simple_slash(interaction, "/plan investigate")
    slash, slash_prompt = await dispatched()
    assert len(received) == count + 1 and slash.text == "/plan investigate"
    assert ordinary.source.message_id and slash.source.message_id is None
    assert slash_prompt is ordinary_prompt
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.delete_original_response.assert_awaited_once()

    await rename_event("Renamed by a moderator")
    changed, changed_prompt = await turn(channels[800], 103)
    assert changed.source.chat_name.endswith(" / Renamed by a moderator")
    assert changed_prompt != slash_prompt
    await rename_event(generated)
    restored, restored_prompt = await turn(channels[800], 104)
    assert restored.source.chat_name == f"Hermes Server / #ops / {generated}"
    assert restored_prompt != first
