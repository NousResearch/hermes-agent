"""Discord adapter race polish: concurrent join_voice_channel must not
double-invoke channel.connect() on the same guild."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="t")
    adapter._ready_event = asyncio.Event()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._client = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_concurrent_joins_do_not_double_connect():
    """Two concurrent join_voice_channel calls on the same guild must
    serialize through the per-guild lock — only ONE channel.connect()
    actually fires; the second sees the _voice_clients entry the first
    just installed."""
    adapter = _make_adapter()

    connect_count = [0]
    release = asyncio.Event()

    class FakeVC:
        def __init__(self, channel):
            self.channel = channel

        def is_connected(self):
            return True

        async def move_to(self, _channel):
            return None

    async def slow_connect(self):
        connect_count[0] += 1
        await release.wait()
        return FakeVC(self)

    channel = MagicMock()
    channel.id = 111
    channel.guild.id = 42
    channel.connect = lambda: slow_connect(channel)

    from plugins.platforms.discord import adapter as discord_mod
    # Ensure the PyNaCl guard inside join_voice_channel is a no-op for this
    # test (the test venv may not have nacl). Use patch.dict so the mock
    # doesn't leak into other test files in the same pytest session.
    with patch.dict("sys.modules", {"nacl": MagicMock()}):
        with patch.object(discord_mod, "VoiceReceiver",
                          MagicMock(return_value=MagicMock(start=lambda: None))):
            with patch.object(discord_mod.asyncio, "ensure_future",
                              lambda _c: asyncio.create_task(asyncio.sleep(0))):
                t1 = asyncio.create_task(adapter.join_voice_channel(channel))
                t2 = asyncio.create_task(adapter.join_voice_channel(channel))
                await asyncio.sleep(0.05)
                release.set()
                r1, r2 = await asyncio.gather(t1, t2)

    assert connect_count[0] == 1, (
        f"expected 1 channel.connect() call, got {connect_count[0]} — "
        "per-guild lock is not serializing join_voice_channel"
    )
    assert r1 is True and r2 is True
    assert 42 in adapter._voice_clients


# ---------------------------------------------------------------------------
# PyNaCl availability guard (Codex PR #10)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_join_voice_returns_false_when_pynacl_missing():
    """When PyNaCl is not installed, join_voice_channel must fail fast with
    a logged warning instead of crashing inside channel.connect() with an
    opaque missing-module error. The `voice` extra no longer ships PyNaCl
    (vulnerable pin), so this guard is the user-facing safety net."""
    import builtins

    adapter = _make_adapter()

    real_import = builtins.__import__

    def _block_nacl(name, *args, **kwargs):
        if name == "nacl":
            raise ImportError("simulated missing PyNaCl (test)")
        return real_import(name, *args, **kwargs)

    channel = MagicMock()
    channel.guild.id = 123

    with patch("builtins.__import__", side_effect=_block_nacl):
        result = await adapter.join_voice_channel(channel)

    assert result is False
    # channel.connect() must NOT have been called — the guard fires before it.
    channel.connect.assert_not_called()


@pytest.mark.asyncio
async def test_join_voice_proceeds_when_pynacl_available():
    """When PyNaCl IS importable, join_voice_channel proceeds normally
    (the guard is a no-op). Pins the happy path so the guard above is
    confirmed to be the regression boundary."""
    adapter = _make_adapter()

    channel = MagicMock()
    channel.guild.id = 456
    channel.connect = AsyncMock(return_value=MagicMock())

    # Ensure the PyNaCl guard is a no-op; patch.dict prevents sys.modules
    # leakage into other test files.
    with patch.dict("sys.modules", {"nacl": MagicMock()}):
        result = await adapter.join_voice_channel(channel)

    assert result is True
    channel.connect.assert_awaited_once()
