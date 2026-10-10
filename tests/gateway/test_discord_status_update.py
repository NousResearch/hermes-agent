"""Tests for DiscordAdapter.send_or_update_status (issue #134288, cf. #30045).

The status-update path must:
  1. Send a fresh message on the first call for a (chat, key).
  2. Edit that same message on subsequent calls with the same key (lease-wait ticks
     every ~15s must not append new bubbles).
  3. Fall back to sending fresh when the cached message edit fails (deleted, …).
  4. Keep distinct keys independent.
  5. Edit in the thread channel when the status was sent into a thread.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


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


def _make_forum_adapter():
    channels = {}
    threads = {}

    async def create_thread(*, name, content):
        thread_id = 900 + len(threads)
        thread = FakeChannel(thread_id)
        threads[thread_id] = thread
        channels[thread_id] = thread
        return SimpleNamespace(thread=thread, message=SimpleNamespace(id=thread_id * 10 + 1))

    forum = SimpleNamespace(id=555, type=15, create_thread=AsyncMock(side_effect=create_thread))
    channels[555] = forum
    return _make_adapter_with(channels), forum, threads


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
async def test_concurrent_first_status_updates_share_one_bubble():
    channel = FakeChannel(555)
    adapter = _make_adapter_with({555: channel})
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    original_send = channel.send.side_effect

    async def held_send(*, content, reference=None):
        send_started.set()
        await release_send.wait()
        return await original_send(content=content, reference=reference)

    channel.send.side_effect = held_send
    first = asyncio.create_task(adapter.send_or_update_status("555", "lease_wait", "waiting"))
    await asyncio.wait_for(send_started.wait(), timeout=2)
    second = asyncio.create_task(adapter.send_or_update_status("555", "lease_wait", "still waiting"))
    try:
        for _ in range(3):
            await asyncio.sleep(0)
        assert channel.send.await_count == 1
    finally:
        release_send.set()
        results = await asyncio.gather(first, second)

    assert all(result.success for result in results)
    assert len(channel.sends) == 1
    assert channel.edits == [(int(results[0].message_id), "still waiting")]


@pytest.mark.asyncio
async def test_distinct_status_destinations_send_in_parallel():
    channels = {555: FakeChannel(555), 900: FakeChannel(900)}
    adapter = _make_adapter_with(channels)
    started = {target: asyncio.Event() for target in channels}
    release = asyncio.Event()

    def hold_send(channel: FakeChannel, target: int):
        original_send = channel.send.side_effect

        async def send(*, content, reference=None):
            started[target].set()
            await release.wait()
            return await original_send(content=content, reference=reference)

        return send

    for target, channel in channels.items():
        channel.send.side_effect = hold_send(channel, target)

    first = asyncio.create_task(adapter.send_or_update_status("555", "lease_wait", "parent"))
    second = asyncio.create_task(adapter.send_or_update_status(
        "555", "lease_wait", "thread", metadata={"thread_id": "900"},
    ))
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), timeout=2)
    finally:
        release.set()
        results = await asyncio.gather(first, second)

    assert all(result.success for result in results)
    assert [len(channel.sends) for channel in channels.values()] == [1, 1]


@pytest.mark.asyncio
async def test_failed_edit_does_not_remove_a_replaced_status_id():
    channel = FakeChannel(555)
    adapter = _make_adapter_with({555: channel})
    key = ("555", "lease_wait")
    adapter._status_message_ids[key] = "101"
    edit_started = asyncio.Event()
    release_edit = asyncio.Event()

    async def failed_edit(*args, **kwargs):
        edit_started.set()
        await release_edit.wait()
        return SendResult(success=False, error="deleted")

    adapter.edit_message = AsyncMock(side_effect=failed_edit)
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="offline"))
    update = asyncio.create_task(adapter.send_or_update_status("555", "lease_wait", "waiting"))
    await asyncio.wait_for(edit_started.wait(), timeout=2)
    adapter._status_message_ids[key] = "202"
    release_edit.set()
    result = await update

    assert not result.success
    assert adapter._status_message_ids[key] == "202"


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


@pytest.mark.asyncio
async def test_forum_parent_status_edits_the_created_thread_starter():
    adapter, forum, threads = _make_forum_adapter()

    first = await adapter.send_or_update_status("555", "lease_wait", "waiting")
    assert first.success
    assert first.raw_response["thread_id"] == "900"
    assert forum.create_thread.await_count == 1

    second = await adapter.send_or_update_status("555", "lease_wait", "still waiting")
    assert second.success
    assert forum.create_thread.await_count == 1
    assert list(threads) == [900]
    assert threads[900].edits == [(int(first.message_id), "still waiting")]


@pytest.mark.asyncio
async def test_forum_parent_status_rebinds_after_deleted_starter():
    adapter, forum, threads = _make_forum_adapter()

    first = await adapter.send_or_update_status("555", "lease_wait", "waiting")
    threads[900].deleted.add(int(first.message_id))
    second = await adapter.send_or_update_status("555", "lease_wait", "still waiting")
    assert second.success
    assert second.raw_response["thread_id"] == "901"
    assert forum.create_thread.await_count == 2

    third = await adapter.send_or_update_status("555", "lease_wait", "still waiting (15s)")
    assert third.success
    assert forum.create_thread.await_count == 2
    assert threads[901].edits == [(int(second.message_id), "still waiting (15s)")]
