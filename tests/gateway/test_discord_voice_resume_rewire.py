"""Regression tests for voice receiver rewiring after a gateway RESUME.

A RESUMED gateway session (as opposed to a full reconnect) can re-establish
voice UDP state — secret key, ssrc, socket reader — underneath a running
``VoiceReceiver``. The receiver captures those once at ``start()``, so after
a resume it goes silently deaf: SPEAKING events still arrive through the
connection-state hook, but no audio reaches STT. The adapter must rewire
receivers on ``on_resumed``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

# Re-use the shared discord-stub bootstrap and FakeBot from the connect
# test module so this file doesn't duplicate the (large) mock surface.
from tests.gateway.test_discord_connect import (  # noqa: E402
    FakeBot,
    _ensure_discord_mock,
)

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _make_adapter() -> DiscordAdapter:
    return DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))


def _fake_vc(connected: bool = True) -> SimpleNamespace:
    return SimpleNamespace(is_connected=lambda: connected)


def _fake_receiver() -> Mock:
    receiver = Mock()
    receiver.calls = []
    receiver.stop.side_effect = lambda: receiver.calls.append("stop")
    receiver.start.side_effect = lambda: receiver.calls.append("start")
    return receiver


async def _connect(adapter: DiscordAdapter, monkeypatch, bot_factory=FakeBot):
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock", lambda scope, identity: None
    )
    intents = SimpleNamespace(
        message_content=False,
        dm_messages=False,
        guild_messages=False,
        members=False,
        voice_states=False,
    )
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)
    monkeypatch.setattr(discord_platform.commands, "Bot", bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())
    assert await adapter.connect() is True


@pytest.mark.asyncio
async def test_on_resumed_is_registered_and_schedules_rewire(monkeypatch):
    """``on_resumed`` must exist and must invoke the rewire coroutine."""
    adapter = _make_adapter()
    await _connect(adapter, monkeypatch)

    handler = adapter._client._events.get("on_resumed")
    assert handler is not None, "adapter must register an on_resumed handler"

    rewire = AsyncMock()
    monkeypatch.setattr(adapter, "_rewire_voice_receivers_after_resume", rewire)
    await handler()
    # The handler schedules a task; give the loop one tick to run it.
    import asyncio

    await asyncio.sleep(0)
    rewire.assert_awaited_once()


@pytest.mark.asyncio
async def test_rewire_restarts_receiver_for_connected_guild():
    """A connected guild's receiver is stopped then started, in that order."""
    adapter = _make_adapter()
    receiver = _fake_receiver()
    adapter._voice_clients[123] = _fake_vc(connected=True)
    adapter._voice_receivers[123] = receiver

    await adapter._rewire_voice_receivers_after_resume(settle_seconds=0)

    assert receiver.calls == ["stop", "start"]


@pytest.mark.asyncio
async def test_rewire_skips_disconnected_voice_client():
    """A guild whose voice client dropped is left alone (no stale restart)."""
    adapter = _make_adapter()
    receiver = _fake_receiver()
    adapter._voice_clients[123] = _fake_vc(connected=False)
    adapter._voice_receivers[123] = receiver

    await adapter._rewire_voice_receivers_after_resume(settle_seconds=0)

    receiver.stop.assert_not_called()
    receiver.start.assert_not_called()


@pytest.mark.asyncio
async def test_rewire_tolerates_missing_receiver():
    """A voice client without a receiver must not raise."""
    adapter = _make_adapter()
    adapter._voice_clients[123] = _fake_vc(connected=True)

    await adapter._rewire_voice_receivers_after_resume(settle_seconds=0)


@pytest.mark.asyncio
async def test_rewire_failure_in_one_guild_does_not_block_others():
    """An exception rewiring one guild must not stop the sweep."""
    adapter = _make_adapter()
    broken = _fake_receiver()
    broken.stop.side_effect = RuntimeError("boom")
    healthy = _fake_receiver()
    adapter._voice_clients[1] = _fake_vc(connected=True)
    adapter._voice_receivers[1] = broken
    adapter._voice_clients[2] = _fake_vc(connected=True)
    adapter._voice_receivers[2] = healthy

    await adapter._rewire_voice_receivers_after_resume(settle_seconds=0)

    assert healthy.calls == ["stop", "start"]
