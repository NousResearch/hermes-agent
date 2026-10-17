"""Tests for Discord message reactions tied to processing lifecycle hooks."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
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

from plugins.platforms.discord.adapter import DiscordAdapter


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
@pytest.mark.parametrize("configured", [False, None, "", "   ", 123])
async def test_invalid_processing_emoji_falls_back_to_default(configured):
    config = PlatformConfig(enabled=True, token="***", extra={"processing_emoji": configured})
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event = _make_event("invalid-processing-emoji", raw_message)

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert raw_message.add_reaction.await_args_list[0].args == ("👀",)
    assert raw_message.remove_reaction.await_args_list[0].args == ("👀", adapter._client.user)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,final_emoji", [
    (ProcessingOutcome.SUCCESS, "✅"),
    (ProcessingOutcome.FAILURE, "❌"),
])
async def test_processing_emoji_from_profile_yaml(tmp_path, monkeypatch, outcome, final_emoji):
    from agent.secret_scope import set_multiplex_active
    from gateway.config import load_gateway_config
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "a", tmp_path / "b"]
    for home, setting in zip(homes, ['  processing_emoji: "🛠️"\n', '']):
        home.mkdir()
        (home / "config.yaml").write_text(
            'discord:\n  enabled: true\n  token: "test-token"\n' + setting,
            encoding="utf-8",
        )
    set_multiplex_active(True)
    try:
        for home, expected in [(homes[0], "🛠️"), (homes[1], "👀"), (homes[0], "🛠️")]:
            token = set_hermes_home_override(str(home))
            try:
                config = load_gateway_config().platforms[Platform.DISCORD]
                adapter = DiscordAdapter(config)
            finally:
                reset_hermes_home_override(token)
            adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
            calls = []

            async def add(emoji):
                calls.append(("add", emoji))

            async def remove(emoji, user):
                assert user is adapter._client.user
                calls.append(("remove", emoji))

            event = _make_event("1", SimpleNamespace(add_reaction=add, remove_reaction=remove))
            await adapter.on_processing_start(event)
            await adapter.on_processing_complete(event, outcome)
            assert calls == [("add", expected), ("remove", expected), ("add", final_emoji)]
    finally:
        set_multiplex_active(False)
