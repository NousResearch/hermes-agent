"""Discord turns built outside on_message carry the same prompt inputs as a message in that chat.

Native slash commands (``/skill``, ``/queue``, ``/plan``), the ``/thread`` starter and voice-channel
input are human turns: each re-pins the session-context prompt and the channel prompt. Built with
another chat label (DM: none; thread: without its parent channel), another topic or a bare user id,
they re-rendered the pinned prompt, and the next typed message rendered it back: a prompt-cache miss
each way. Built without ``auto_skill``, a session they opened never loaded the channel's bound skill.
Voice rebuilds its source from a ``/voice join`` copy, so it must also see the parent's bindings from a
thread and the channel's current name and topic. A programmatic join binds no copy: the voice turn then
builds the source a typed message in that channel carries, or it keys another session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
import plugins.platforms.discord.adapter as discord_platform
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import build_session_context, build_session_key
from plugins.platforms.discord.adapter import DiscordAdapter


class _DM:
    def __init__(self, channel_id: int) -> None:
        self.id, self.name = channel_id, "dm"


class _Text:
    def __init__(self, channel_id: int, name: str = "ops", topic: str | None = "Incident triage") -> None:
        self.id, self.name, self.topic = channel_id, name, topic
        self.guild = SimpleNamespace(id=1, name="Hermes Server")


class _Thread:
    def __init__(self, channel_id: int, parent: _Text, name: str = "incident-42") -> None:
        self.id, self.name, self.parent, self.parent_id = channel_id, name, parent, parent.id
        self.guild, self.topic = parent.guild, None


_USER = SimpleNamespace(id=42, display_name="Alice", name="alice")


def _adapter(monkeypatch: pytest.MonkeyPatch, bound_id: int) -> DiscordAdapter:
    monkeypatch.setattr(discord_platform.discord, "DMChannel", _DM, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", _Thread, raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake", extra={
        "channel_prompts": {str(bound_id): "Answer in haiku."},
        "channel_skill_bindings": [{"id": str(bound_id), "skill": "triage"}]}))
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=999), get_channel=lambda _id: None,
        get_guild=lambda _id: SimpleNamespace(get_member=lambda _uid: _USER))
    adapter._text_batch_delay_seconds = 0
    adapter._discord_history_backfill = lambda: False
    adapter.handle_message = AsyncMock()
    return adapter


async def _typed(adapter: DiscordAdapter, channel: Any) -> Any:
    await adapter._handle_message(SimpleNamespace(
        id=123, content="what broke?", mentions=[], attachments=[], reference=None,
        created_at=datetime.now(timezone.utc), channel=channel, author=_USER,
        guild=getattr(channel, "guild", None)))
    return adapter.handle_message.await_args.args[0]


def _assert_same_prompt_inputs(typed: Any, other: Any) -> None:
    """typed -> other -> typed through the real pins leaves one context prompt and one channel prompt."""
    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    pinned = []
    for event in (typed, other, typed):
        context = build_session_context(event.source, config)
        channel_prompt, _ = runner._pinned_channel_inputs(
            "k", event.channel_prompt, event.source, internal=False)
        pinned.append((runner._pinned_session_context_prompt(context, False, "k"), channel_prompt))
    assert typed.channel_prompt == "Answer in haiku."
    assert len(set(pinned)) == 1, (pinned[0][0], pinned[1][0])
    assert other.auto_skill == typed.auto_skill == ["triage"]


def _parent() -> _Text:
    return _Text(700)


def _interaction(channel: Any) -> SimpleNamespace:
    return SimpleNamespace(
        channel=channel, channel_id=channel.id, guild=getattr(channel, "guild", None), guild_id=None,
        user=_USER, response=SimpleNamespace(defer=AsyncMock()), followup=SimpleNamespace(send=AsyncMock()),
        delete_original_response=AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [False, True], ids=["no-discord-tools", "discord-tools"])
@pytest.mark.parametrize("case", ["dm", "channel", "thread", "thread-starter"])
async def test_slash_and_thread_starter_turns_match_a_message_turn(
    monkeypatch: pytest.MonkeyPatch, case: str, tools: bool,
) -> None:
    # With Discord tools on, the pinned notes list IDs; a slash turn has no triggering message.
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: tools)
    parent = _parent()
    channel = {"dm": _DM(500), "channel": parent}.get(case) or _Thread(800, parent)
    adapter = _adapter(monkeypatch, channel.id)
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    typed = await _typed(adapter, channel)
    if case == "thread-starter":
        channel.send = AsyncMock()
        parent.create_thread = AsyncMock(return_value=channel)
        await adapter._handle_thread_create_slash(_interaction(parent), channel.name, "what broke?")
    else:
        await adapter._run_simple_slash(_interaction(channel), "/skill triage what broke?")
    assert adapter.handle_message.await_count == 2
    _assert_same_prompt_inputs(typed, adapter.handle_message.await_args.args[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["channel", "thread-under-bound-parent", "renamed-after-join", "speaker-uncached",
                                  "programmatic-join", "programmatic-join-thread"])
async def test_voice_channel_turn_matches_a_typed_turn(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    parent = _parent()
    channel = _Thread(800, parent) if case.endswith("thread") or case.startswith("thread") else parent
    adapter = _adapter(monkeypatch, parent.id)
    join = adapter._build_slash_event(
        SimpleNamespace(channel=channel, channel_id=channel.id, guild=parent.guild, guild_id=1, user=_USER),
        "/voice join")
    adapter._voice_text_channels = {1: channel.id}
    # A typed `/voice join` binds that message's source, id included; a voice turn must not inherit it
    # as its own trigger.
    adapter._voice_sources = {1: {**join.source.to_dict(), "message_id": "1554000000000000000"}}
    if case.startswith("programmatic-join"):
        adapter._voice_sources = {}  # join_voice_channel(..., text_channel_id=...) without a source
    adapter._client.get_channel = {channel.id: channel}.get
    if case == "renamed-after-join":
        channel.name, channel.topic = "renamed", "New topic"
    if case == "speaker-uncached":
        adapter._client.get_guild = lambda _id: SimpleNamespace(get_member=lambda _uid: None)
        # Cached as users only: the joiner's bound nickname still beats their global name.
        adapter._client.get_user = {42: SimpleNamespace(display_name="alice_global"),
                                    43: SimpleNamespace(display_name="Bob")}.get
    typed = await _typed(adapter, channel)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_mode = {}
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source: True

    await runner._handle_voice_channel_input(1, _USER.id, "what broke?", adapter=adapter)

    spoken = adapter.handle_message.await_args.args[0]
    assert spoken is not typed
    _assert_same_prompt_inputs(typed, spoken)
    assert build_session_key(spoken.source) == build_session_key(typed.source)
    assert spoken.source.message_id is None
    # The join-time name belongs to the joiner only; another uncached speaker never borrows it, and
    # gets their cached user name (global name, no server nickname) rather than a bare id.
    other = runner._voice_input_source(adapter, 1, 43, channel.id).user_name
    assert other == ("Bob" if case == "speaker-uncached" else "Alice")
    if case == "speaker-uncached":
        assert runner._voice_input_source(adapter, 1, 44, channel.id).user_name == "44"  # cached nowhere
        adapter._voice_sources = {}
        assert runner._voice_input_source(adapter, 1, 43, channel.id).user_name == "Bob"  # no binding
