"""An auto-threaded @mention opens the thread's session with the prompt inputs its later messages use.

The first turn's source points at the new thread, but its topic and its channel prompt / skill lookups
were read from the parent text channel the mention was posted in. The next message in the thread read
them from the thread, so the pinned session-context prompt (``Channel Topic``) was re-rendered on the
second turn of every auto-threaded conversation in a channel with a topic: a prompt-cache miss.
"""

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
    def __init__(self, channel_id, name="ops", topic="Incident triage"):
        self.id, self.name, self.topic = channel_id, name, topic
        self.guild = SimpleNamespace(id=1, name="Hermes Server")


class _Thread:
    def __init__(self, channel_id, parent, name="what-broke"):
        self.id, self.name, self.parent, self.parent_id = channel_id, name, parent, parent.id
        self.guild = parent.guild


_USER = SimpleNamespace(id=42, display_name="Alice", name="alice")


def _message(channel, message_id):
    return SimpleNamespace(
        id=message_id, content="what broke?", mentions=[], attachments=[], reference=None,
        created_at=datetime.now(timezone.utc), channel=channel, author=_USER)


@pytest.mark.asyncio
async def test_auto_thread_first_turn_matches_the_threads_next_message(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "Thread", _Thread, raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    parent = _Text(700)
    thread = _Thread(800, parent)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake", extra={
        "channel_prompts": {"700": "Answer in haiku."},
        "channel_skill_bindings": [{"id": "700", "skill": "triage"}]}))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter._discord_history_backfill = lambda: False
    adapter._auto_create_thread = AsyncMock(return_value=thread)
    adapter.handle_message = AsyncMock()

    events = []
    for channel, message_id in ((parent, 100), (thread, 101)):
        await adapter._handle_message(_message(channel, message_id))
        events.append(adapter.handle_message.await_args.args[0])
    adapter._auto_create_thread.assert_awaited_once()
    first, follow_up = events
    assert first.source.chat_id == follow_up.source.chat_id == "800"

    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    pinned = []
    for event in (first, follow_up):
        context = build_session_context(event.source, config)
        channel_prompt, _ = runner._pinned_channel_inputs("k", event.channel_prompt, event.source, internal=False)
        pinned.append((runner._pinned_session_context_prompt(context, False, "k"), channel_prompt))
    assert pinned[0][1] == "Answer in haiku."
    assert pinned[0] == pinned[1], (pinned[0][0], pinned[1][0])
    assert first.auto_skill == follow_up.auto_skill == ["triage"]
