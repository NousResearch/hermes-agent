"""Tests for DiscordAdapter.send_or_update_status (issue #134288, cf. #30045).

The status-update path must:
  1. Send a fresh message on the first call for a (chat, key).
  2. Edit that same message on subsequent calls with the same key (lease-wait ticks
     every ~15s must not append new bubbles).
  3. Fall back to sending fresh when the cached message edit fails (deleted, …).
  4. Keep distinct keys independent.
  5. Edit in the thread channel when the status was sent into a thread.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Embed = MagicMock
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class FakeChannel:
    """Text-channel fake: records sends, hands out partial messages for edits."""

    def __init__(self, channel_id: int):
        self.id = channel_id
        self.sends: list[str] = []
        self.edits: list[tuple[int, str]] = []
        self.deleted: set[int] = set()
        self._next_id = channel_id * 10

        async def _send(*, content, reference=None):
            self._next_id += 1
            self.sends.append(content)
            return SimpleNamespace(id=self._next_id)

        async def _edit(message_id, *, content):
            message_id = int(message_id)
            if message_id in self.deleted:
                raise RuntimeError(f"error code: 10008: Unknown Message {message_id}")
            self.edits.append((message_id, content))
            return SimpleNamespace(id=message_id)

        self.send = AsyncMock(side_effect=_send)

        def get_partial_message(message_id):

            async def _do_edit(*, content):
                return await _edit(message_id, content=content)

            return SimpleNamespace(id=message_id, edit=AsyncMock(side_effect=_do_edit))

        self.get_partial_message = MagicMock(side_effect=get_partial_message)


def _make_adapter_with(channels: dict[int, FakeChannel]):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: channels.get(cid),
        fetch_channel=AsyncMock(side_effect=lambda cid: channels[int(cid)]),
    )
    return adapter


@pytest.mark.asyncio
async def test_first_call_sends_then_repeats_edit_in_place():
    channel = FakeChannel(555)
    adapter = _make_adapter_with({555: channel})

    result = await adapter.send_or_update_status(
        "555", "lease_wait", "⏳ Another Hermes process is using this session..."
    )
    assert result.success is True
    assert len(channel.sends) == 1
    assert channel.edits == []
    first_id = result.message_id
    assert first_id

    for text in (
        "⏳ Still waiting (15s)...",
        "⏳ Still waiting (31s)...",
        "⏳ Still waiting (46s)...",
    ):
        result = await adapter.send_or_update_status("555", "lease_wait", text)
        assert result.success is True
        assert result.message_id == first_id

    # One bubble total: the 15s lease-wait ticks edited the same message, never appended.
    assert len(channel.sends) == 1
    assert [c for (_mid, c) in channel.edits] == [
        "⏳ Still waiting (15s)...",
        "⏳ Still waiting (31s)...",
        "⏳ Still waiting (46s)...",
    ]
    assert all(mid == int(first_id) for (mid, _c) in channel.edits)


@pytest.mark.asyncio
async def test_failed_edit_falls_back_to_fresh_send():
    channel = FakeChannel(555)
    adapter = _make_adapter_with({555: channel})

    first = await adapter.send_or_update_status("555", "lease_wait", "waiting")
    assert first.success is True
    assert len(channel.sends) == 1

    # Simulate the cached bubble being deleted: the next edit fails and a fresh message is sent.
    channel.deleted.add(int(first.message_id))
    second = await adapter.send_or_update_status("555", "lease_wait", "still waiting")
    assert second.success is True
    assert len(channel.sends) == 2
    assert second.message_id != first.message_id

    # The new id is cached: the following tick edits it in place again.
    await adapter.send_or_update_status("555", "lease_wait", "still waiting (15s)")
    assert len(channel.sends) == 2
    assert channel.edits[-1][0] == int(second.message_id)


@pytest.mark.asyncio
async def test_distinct_keys_do_not_crosstalk():
    channel = FakeChannel(555)
    adapter = _make_adapter_with({555: channel})

    await adapter.send_or_update_status("555", "lease_wait", "waiting")
    await adapter.send_or_update_status("555", "context_pressure", "compressing")
    assert len(channel.sends) == 2
    assert channel.edits == []

    await adapter.send_or_update_status("555", "lease_wait", "waiting (15s)")
    await adapter.send_or_update_status("555", "context_pressure", "compressing more")
    assert len(channel.sends) == 2
    assert len(channel.edits) == 2


@pytest.mark.asyncio
async def test_thread_status_edits_in_thread_channel():
    parent, thread = FakeChannel(555), FakeChannel(900)
    adapter = _make_adapter_with({555: parent, 900: thread})
    metadata = {"thread_id": "900"}

    first = await adapter.send_or_update_status(
        "555", "lease_wait", "waiting", metadata=metadata
    )
    assert first.success is True
    assert len(thread.sends) == 1
    assert parent.sends == []

    # The edit targets the thread channel (metadata thread_id wins over chat_id).
    result = await adapter.send_or_update_status(
        "555", "lease_wait", "waiting (15s)", metadata=metadata
    )
    assert result.success is True
    assert len(thread.sends) == 1
    assert len(thread.edits) == 1
    assert thread.edits[0][0] == int(first.message_id)
    assert parent.sends == [] and parent.edits == []
