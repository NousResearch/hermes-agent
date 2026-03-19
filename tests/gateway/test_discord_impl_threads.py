from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import importlib
import sys

import pytest


def _load_threads_module():
    class FakeIntents:
        @staticmethod
        def default():
            return SimpleNamespace()

    discord_mod = ModuleType("discord")
    discord_mod.__file__ = "mock-discord.py"
    discord_mod.Message = type("Message", (), {})
    discord_mod.Intents = FakeIntents
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1,
        primary=2,
        danger=3,
        green=1,
        blurple=2,
        red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(
        is_loaded=lambda: True,
        load_opus=lambda *_args, **_kwargs: None,
        Decoder=MagicMock,
    )
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    sys.modules["discord"] = discord_mod
    ext_mod = ModuleType("discord.ext")
    commands_mod = ModuleType("discord.ext.commands")
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules["discord.ext"] = ext_mod
    sys.modules["discord.ext.commands"] = commands_mod
    return importlib.reload(importlib.import_module("gateway.platforms.discord_impl.threads"))


threads = _load_threads_module()


@pytest.mark.asyncio
async def test_auto_create_thread_uses_message_content():
    thread = SimpleNamespace(id=1, name="hello")
    message = SimpleNamespace(
        content="hello world",
        create_thread=AsyncMock(return_value=thread),
    )

    result = await threads.auto_create_thread(message)

    assert result is thread
    message.create_thread.assert_awaited_once_with(
        name="hello world",
        auto_archive_duration=1440,
    )


@pytest.mark.asyncio
async def test_auto_create_thread_truncates_long_message_names():
    thread = SimpleNamespace(id=2, name="truncated")
    message = SimpleNamespace(
        content="a" * 200,
        create_thread=AsyncMock(return_value=thread),
    )

    result = await threads.auto_create_thread(message)

    assert result is thread
    thread_name = message.create_thread.await_args.kwargs["name"]
    assert len(thread_name) <= 80
    assert thread_name.endswith("...")


@pytest.mark.asyncio
async def test_create_thread_succeeds_with_direct_creation():
    created_thread = SimpleNamespace(id=555, name="Planning", send=AsyncMock())
    parent_channel = SimpleNamespace(
        create_thread=AsyncMock(return_value=created_thread),
        send=AsyncMock(),
    )
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="Jezza"),
    )

    async def resolve_channel_fn(_client, _interaction):
        return SimpleNamespace(parent=parent_channel)

    result = await threads.create_thread(
        client=MagicMock(),
        interaction=interaction,
        name="Planning",
        message="Kickoff",
        auto_archive_duration=1440,
        resolve_channel_fn=resolve_channel_fn,
    )

    assert result == {
        "success": True,
        "thread_id": "555",
        "thread_name": "Planning",
    }
    parent_channel.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )
    created_thread.send.assert_awaited_once_with("Kickoff")


@pytest.mark.asyncio
async def test_create_thread_falls_back_to_seed_message():
    created_thread = SimpleNamespace(id=777, name="Planning")
    seed_message = SimpleNamespace(create_thread=AsyncMock(return_value=created_thread))
    parent_channel = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("direct failed")),
        send=AsyncMock(return_value=seed_message),
    )
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="Jezza"),
    )

    async def resolve_channel_fn(_client, _interaction):
        return SimpleNamespace(parent=parent_channel)

    result = await threads.create_thread(
        client=MagicMock(),
        interaction=interaction,
        name="Planning",
        message="Kickoff",
        auto_archive_duration=1440,
        resolve_channel_fn=resolve_channel_fn,
    )

    assert result == {
        "success": True,
        "thread_id": "777",
        "thread_name": "Planning",
    }
    parent_channel.send.assert_awaited_once_with("Kickoff")
    seed_message.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "message", "auto_archive_duration", "resolve_mode", "expected_error"),
    [
        ("", "", 1440, "unused", "Thread name is required."),
        (
            "Planning",
            "",
            999,
            "unused",
            "auto_archive_duration must be one of: 60, 1440, 4320, 10080.",
        ),
        (
            "Planning",
            "",
            1440,
            "missing",
            "Could not resolve the current Discord channel.",
        ),
    ],
)
async def test_create_thread_returns_expected_errors(
    name,
    message,
    auto_archive_duration,
    resolve_mode,
    expected_error,
):
    if resolve_mode == "missing":
        async def resolve_channel_fn(_client, _interaction):
            return None
    else:
        async def resolve_channel_fn(_client, _interaction):
            return SimpleNamespace(parent=SimpleNamespace())

    result = await threads.create_thread(
        client=MagicMock(),
        interaction=SimpleNamespace(user=SimpleNamespace(display_name="Jezza")),
        name=name,
        message=message,
        auto_archive_duration=auto_archive_duration,
        resolve_channel_fn=resolve_channel_fn,
    )

    assert result == {"error": expected_error}


def test_thread_parent_channel_returns_parent_or_self():
    parent = SimpleNamespace(id=1)
    assert threads.thread_parent_channel(SimpleNamespace(parent=parent)) is parent
    channel = SimpleNamespace(parent=None)
    assert threads.thread_parent_channel(channel) is channel


@pytest.mark.asyncio
async def test_resolve_interaction_channel_returns_available_channel():
    channel = SimpleNamespace(id=123)
    interaction = SimpleNamespace(channel=channel, channel_id=123)

    result = await threads.resolve_interaction_channel(MagicMock(), interaction)

    assert result is channel


@pytest.mark.asyncio
async def test_resolve_interaction_channel_returns_none_when_missing():
    client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(side_effect=RuntimeError("missing")),
    )
    interaction = SimpleNamespace(channel=None, channel_id=123)

    result = await threads.resolve_interaction_channel(client, interaction)

    assert result is None
