"""A kickoff slash command must open the thread a plain channel message would get.

Before #140720, ``/goal`` (and the rest of the loop family) entered through the slash path,
which never ran the auto-thread gate in ``on_message``: no thread was created, the session was
born channel-scoped, and every loop continuation reused that source and posted into the shared
channel for the whole life of the goal.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_discord_slash_commands import _ensure_discord_mock


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture(autouse=True)
def _clean_discord_gate_env(monkeypatch):
    """Keep the gate values in this module's own config: another test's leaked DISCORD_* env
    would otherwise decide whether a kickoff threads."""
    for key in (
        "DISCORD_AUTO_THREAD",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_FREE_RESPONSE_AUTO_THREAD",
        "DISCORD_NO_THREAD_CHANNELS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_adapter(extra=None) -> DiscordAdapter:
    config = PlatformConfig(enabled=True, token="***")
    config.extra = dict(extra or {})
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=99999, name="HermesBot"),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
    )
    adapter._text_batch_delay_seconds = 0
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter.handle_message = AsyncMock()
    adapter._threads.mark_async = AsyncMock()
    return adapter


def _make_channel(channel_id=12345, thread=None):
    parent = SimpleNamespace(
        id=channel_id,
        name="general",
        guild=SimpleNamespace(name="TestGuild", id=42),
        create_thread=AsyncMock(return_value=thread),
        send=AsyncMock(),
    )
    channel = SimpleNamespace(
        id=channel_id,
        name="general",
        guild=parent.guild,
        parent=None,
        create_thread=parent.create_thread,
        send=parent.send,
    )
    return channel


def _make_interaction(channel, channel_id=12345):
    return SimpleNamespace(
        channel=channel,
        channel_id=channel_id,
        guild_id=42,
        guild=SimpleNamespace(id=42, name="TestGuild", get_member=lambda _uid: None),
        user=SimpleNamespace(id=42, name="Jezza", display_name="Jezza"),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        edit_original_response=AsyncMock(),
        delete_original_response=AsyncMock(),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def _event(adapter):
    adapter.handle_message.assert_awaited_once()
    return adapter.handle_message.await_args.args[0]


@pytest.mark.asyncio
async def test_goal_kickoff_opens_a_thread_and_routes_the_session_into_it():
    thread = SimpleNamespace(id=777, name="/goal ship the release", topic=None, parent=None)
    adapter = _make_adapter()
    channel = _make_channel(thread=thread)
    interaction = _make_interaction(channel)

    await adapter._run_simple_slash(interaction, "/goal ship the release")

    channel.create_thread.assert_awaited_once()
    event = _event(adapter)
    assert event.source.chat_type == "thread"
    assert event.source.chat_id == "777"
    assert event.source.thread_id == "777"
    assert event.text == "/goal ship the release"
    adapter._threads.mark_async.assert_awaited_once_with("777")


@pytest.mark.asyncio
async def test_loop_kickoff_uses_the_command_text_as_the_thread_name():
    thread = SimpleNamespace(id=777, name="x", topic=None, parent=None)
    adapter = _make_adapter()
    channel = _make_channel(thread=thread)

    await adapter._run_simple_slash(_make_interaction(channel), "/loop 30m check the queue")

    name = channel.create_thread.await_args.kwargs["name"]
    assert name == "30m check the queue"
    assert _event(adapter).source.chat_id == "777"


@pytest.mark.asyncio
async def test_local_slash_command_keeps_the_channel_scope():
    """Only kickoffs that start a session are threaded; /status still answers in place."""
    adapter = _make_adapter()
    channel = _make_channel()

    await adapter._run_simple_slash(_make_interaction(channel), "/status")

    channel.create_thread.assert_not_awaited()
    event = _event(adapter)
    assert event.source.chat_type == "group"
    assert event.source.chat_id == "12345"


@pytest.mark.asyncio
async def test_no_thread_channel_opts_the_kickoff_out():
    adapter = _make_adapter({"no_thread_channels": ["12345"]})
    channel = _make_channel()

    await adapter._run_simple_slash(_make_interaction(channel), "/goal ship it")

    channel.create_thread.assert_not_awaited()
    assert _event(adapter).source.chat_type == "group"


@pytest.mark.asyncio
async def test_free_response_channel_without_the_opt_in_keeps_the_channel_scope():
    adapter = _make_adapter({"free_response_channels": ["12345"]})
    channel = _make_channel()

    await adapter._run_simple_slash(_make_interaction(channel), "/goal ship it")

    channel.create_thread.assert_not_awaited()
    assert _event(adapter).source.chat_type == "group"


@pytest.mark.asyncio
async def test_free_response_channel_with_the_opt_in_threads_the_kickoff():
    thread = SimpleNamespace(id=777, name="x", topic=None, parent=None)
    adapter = _make_adapter(
        {"free_response_channels": ["12345"], "free_response_auto_thread": True},
    )
    channel = _make_channel(thread=thread)

    await adapter._run_simple_slash(_make_interaction(channel), "/goal ship it")

    channel.create_thread.assert_awaited_once()
    assert _event(adapter).source.chat_type == "thread"


@pytest.mark.asyncio
async def test_auto_thread_disabled_keeps_the_kickoff_channel_scoped():
    adapter = _make_adapter({"auto_thread": False})
    channel = _make_channel()

    await adapter._run_simple_slash(_make_interaction(channel), "/goal ship it")

    channel.create_thread.assert_not_awaited()
    assert _event(adapter).source.chat_type == "group"


@pytest.mark.asyncio
async def test_voice_linked_channel_keeps_the_kickoff_channel_scoped():
    adapter = _make_adapter()
    adapter._voice_text_channels = {1: 12345}
    channel = _make_channel()

    await adapter._run_simple_slash(_make_interaction(channel), "/goal ship it")

    channel.create_thread.assert_not_awaited()
    assert _event(adapter).source.chat_type == "group"


@pytest.mark.asyncio
async def test_thread_creation_failure_still_runs_the_kickoff():
    """No thread is not fatal here: the command runs channel-scoped instead of being dropped."""
    adapter = _make_adapter()
    channel = _make_channel()
    channel.create_thread = AsyncMock(side_effect=RuntimeError("missing permissions"))
    channel.send = AsyncMock(side_effect=RuntimeError("missing permissions"))

    await adapter._run_simple_slash(_make_interaction(channel), "/goal ship it")

    event = _event(adapter)
    assert event.text == "/goal ship it"
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_kickoff_inside_a_thread_does_not_open_a_nested_thread():
    import discord

    thread_channel = discord.Thread()
    thread_channel.id = 555
    thread_channel.parent_id = 12345
    thread_channel.name = "Planning"
    adapter = _make_adapter()

    await adapter._run_simple_slash(_make_interaction(thread_channel, 555), "/goal ship it")

    event = _event(adapter)
    assert event.source.chat_type == "thread"
    assert event.source.thread_id == "555"


@pytest.mark.asyncio
async def test_kickoff_in_a_dm_never_opens_a_thread():
    import discord

    adapter = _make_adapter()
    dm_channel = discord.DMChannel()

    await adapter._run_simple_slash(_make_interaction(dm_channel, 999), "/goal ship it")

    event = _event(adapter)
    assert event.source.chat_type == "dm"
