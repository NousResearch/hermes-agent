"""Discord lifecycle status delivery through the gateway callback (#134288)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.status_output import StatusOutputMixin
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.discord.adapter import DiscordAdapter


class _Channel:
    def __init__(self, channel_id: int):
        self.id = channel_id
        self.sent: list[tuple[int, str]] = []
        self.edited: list[tuple[int, str]] = []
        self._next_id = channel_id * 100

    async def send(self, *, content: str, reference=None):
        self._next_id += 1
        self.sent.append((self._next_id, content))
        return SimpleNamespace(id=self._next_id)

    def get_partial_message(self, message_id: int):
        async def edit(*, content: str):
            if message_id not in {mid for mid, _ in self.sent}:
                raise RuntimeError("Unknown Message")
            self.edited.append((message_id, content))

        return SimpleNamespace(edit=edit)


def _turn(adapter: DiscordAdapter, thread_id: int) -> TurnRunner:
    source = SessionSource(platform=Platform.DISCORD, chat_id="555", thread_id=str(thread_id))
    ctx = TurnContext(
        source=source,
        user_config={"display": {}},
        _run_still_current=lambda: True,
        _status_adapter=adapter,
        _status_chat_id="555",
        _status_thread_metadata={"thread_id": str(thread_id)},
    )
    return TurnRunner(object.__new__(GatewayRunner), ctx)


@pytest.mark.asyncio
async def test_gateway_lifecycle_status_edits_each_thread_without_affecting_direct_sends(monkeypatch):
    """Lease-wait ticks replace their own status bubble in each Discord thread."""
    from gateway import run

    threads = {900: _Channel(900), 901: _Channel(901)}
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test"))
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: threads.get(cid),
        fetch_channel=lambda cid: threads[cid],
    )
    scheduled = []

    def schedule(coro, *args, **kwargs):
        task = asyncio.create_task(coro)
        scheduled.append(task)
        return task

    monkeypatch.setattr(run, "safe_schedule_threadsafe", schedule)

    first = StatusOutputMixin()
    first.status_callback = _turn(adapter, 900)._status_callback_sync
    second = StatusOutputMixin()
    second.status_callback = _turn(adapter, 901)._status_callback_sync

    for emitter in (first, second):
        emitter._emit_status("⏳ Another Hermes process is using this session...")
        await asyncio.gather(*scheduled)
    for emitter in (first, second):
        emitter._emit_status("⏳ Still waiting for the other Hermes process (15s)...")
        await asyncio.gather(*scheduled)

    for channel in threads.values():
        assert len(channel.sent) == 1
        assert len(channel.edited) == 1
        assert channel.edited[0][0] == channel.sent[0][0]

    direct = await adapter.send("555", "A regular reply", metadata={"thread_id": "900"})
    assert direct.success
    assert len(threads[900].sent) == 2
