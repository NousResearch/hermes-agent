"""Approval pause must reach the pinned SDK's internal HTTP retry and rate-limit gates."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp.tracing import Trace
import discord.http as sdk_http
import pytest
import pytest_asyncio
from yarl import URL

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
import plugins.platforms.discord.adapter as adapter_module
from tests.gateway.test_discord_typing_pause import TypingClock, advance_cycle


class Gate(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()

    async def wait(self):
        self.entered.set()
        return await super().wait()


class Wire:
    def __init__(self, trace_config):
        self.posts = []
        self.responses = []
        self.active = 0
        self.headers_gate = None
        self.trace = None
        if trace_config is not None:
            trace_config.freeze()
            self.trace = Trace(self, trace_config, trace_config.trace_config_ctx())

    @asynccontextmanager
    async def request(self, method, url, **kwargs):
        # aiohttp emits start before connector acquisition, headers immediately before writing.
        url = URL(url)
        if self.trace:
            await self.trace.send_request_start(method, url, kwargs['headers'])
        if self.headers_gate:
            await self.headers_gate.wait()
        if self.trace:
            await self.trace.send_request_headers(method, url, kwargs['headers'])
        self.posts.append((method, url.path))
        status, body = self.responses.pop(0) if self.responses else (204, '')
        response = SimpleNamespace(status=status, reason='test',
            headers={'content-type': 'application/json', 'Via': 'test'},
            text=AsyncMock(return_value=json.dumps(body)))
        self.active += 1
        try:
            yield response
        finally:
            self.active -= 1


@pytest_asyncio.fixture
async def transport(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='test-token'))
    monkeypatch.setattr(adapter, '_acquire_platform_lock', lambda *a: True)
    monkeypatch.setattr(adapter, '_snapshot_gate_env', lambda: None)
    monkeypatch.setattr(adapter, '_register_slash_commands', lambda: None)
    monkeypatch.setattr(adapter, '_start_liveness_probe', lambda: None)
    monkeypatch.setattr(adapter_module.discord.opus, 'is_loaded', lambda: True)
    bot_type = adapter_module.commands.Bot
    wires = []

    def bot_factory(**kwargs):
        bot = bot_type(**kwargs)
        wire = Wire(bot.http.http_trace)
        wires.append(wire)
        bot.http._HTTPClient__session = wire
        bot.http._global_over = asyncio.Event()
        bot.http._global_over.set()

        async def start(token):
            adapter._ready_event.set()
            await asyncio.Event().wait()
        bot.start = start
        return bot

    monkeypatch.setattr(adapter_module.commands, 'Bot', bot_factory)
    assert await adapter.connect()
    inner_clock, sdk_clock = TypingClock(12), TypingClock(1)
    monkeypatch.setattr(adapter_module, 'asyncio', inner_clock)
    monkeypatch.setattr(sdk_http, 'asyncio', sdk_clock)
    loop = asyncio.get_running_loop()
    old_debug = loop.get_debug()
    loop.set_debug(True)
    try:
        yield adapter, wires[0], inner_clock, sdk_clock
    finally:
        await adapter.stop_typing('77')
        adapter._bot_task.cancel()
        await asyncio.gather(adapter._bot_task, return_exceptions=True)
        assert wires[0].active == 0
        assert not adapter._typing_tasks
        loop.set_debug(old_debug)


async def finish_attempt(clock, task):
    """Observe either request abandonment or its next ordinary refresh boundary."""
    boundary = asyncio.create_task(clock.parked())
    try:
        done, _ = await asyncio.wait({task, boundary}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        assert done, 'transport neither stopped nor reached a refresh boundary'
    finally:
        boundary.cancel()
        await asyncio.gather(boundary, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('global_limit', [False, True], ids=['bucket-429', 'global-429'])
@pytest.mark.parametrize('rapid_resume', [False, True])
async def test_pause_during_sdk_retry(transport, global_limit, rapid_resume):
    adapter, wire, clock, sdk_clock = transport
    wire.responses.append((429, {'retry_after': 1, 'global': global_limit}))
    await adapter.send_typing('77')
    task = adapter._typing_tasks['77']
    retry = await sdk_clock.parked()
    assert len(wire.posts) == 1
    await asyncio.to_thread(adapter.pause_typing_for_chat, '77')
    if rapid_resume:
        await asyncio.to_thread(adapter.resume_typing_for_chat, '77')
    retry.set_result(None)
    await finish_attempt(clock, task)
    assert len(wire.posts) == (2 if rapid_resume else 1), 'SDK retried after approval pause'
    assert adapter._client.http._global_over.is_set(), 'pause stranded the SDK global gate'
    if not rapid_resume:
        await asyncio.to_thread(adapter.resume_typing_for_chat, '77')
        await adapter.send_typing('77')
        await clock.parked()
        assert len(wire.posts) == 2
    assert wire.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('gate_kind', ['global', 'bucket', 'headers'])
async def test_pause_while_first_post_is_gate_delayed(transport, gate_kind):
    adapter, wire, clock, _sdk_clock = transport
    http = adapter._client.http
    gate = Gate()
    bucket = None
    if gate_kind == 'global':
        http._global_over = gate
    elif gate_kind == 'headers':
        wire.headers_gate = gate
    else:
        route = sdk_http.Route('POST', '/channels/{channel_id}/typing', channel_id='77')
        bucket = http.get_ratelimit(f'{route.key}:{route.major_parameters}')
        bucket.remaining = 0
    await adapter.send_typing('77')
    task = adapter._typing_tasks['77']
    if bucket is None:
        await asyncio.wait_for(gate.entered.wait(), 5)
    else:
        from tests.waiting import async_wait_until
        await async_wait_until(lambda: bool(bucket._pending_requests), timeout=5)
    await asyncio.to_thread(adapter.pause_typing_for_chat, '77')
    if bucket is None:
        gate.set()
    else:
        bucket.remaining = 1
        bucket._wake_next()
    await finish_attempt(clock, task)
    assert not wire.posts, 'gate-delayed first POST escaped approval pause'
    await asyncio.to_thread(adapter.resume_typing_for_chat, '77')
    await adapter.send_typing('77')
    await clock.parked()
    assert len(wire.posts) == 1


@pytest.mark.asyncio
async def test_pause_resume_does_not_cancel_replacement_or_other_routes(transport):
    adapter, wire, clock, _sdk_clock = transport
    await adapter.send_typing('77')
    sleeper = await clock.parked()
    old = adapter._typing_tasks['77']
    await asyncio.to_thread(adapter.pause_typing_for_chat, '77')
    await advance_cycle(clock, sleeper, old)
    assert old.done()
    await asyncio.to_thread(adapter.resume_typing_for_chat, '77')
    await adapter.send_typing('77')
    await clock.parked()
    current = adapter._typing_tasks['77']
    assert current is not old and not current.done()
    await asyncio.to_thread(adapter.pause_typing_for_chat, '77')
    await adapter._client.http.request(sdk_http.Route('POST', '/channels/{channel_id}/messages', channel_id='77'))
    assert wire.posts[-1][1].endswith('/messages')
    await adapter.stop_typing('77')
    assert current.done() and not adapter._typing_tasks


@pytest.mark.asyncio
async def test_stop_during_sdk_gate_does_not_orphan_replacement(transport):
    adapter, wire, clock, _sdk_clock = transport
    gate = Gate()
    adapter._client.http._global_over = gate
    await adapter.send_typing('77')
    await asyncio.wait_for(gate.entered.wait(), 5)
    old = adapter._typing_tasks['77']
    await asyncio.to_thread(adapter.pause_typing_for_chat, '77')
    await adapter.stop_typing('77')
    assert old.done() and not adapter._typing_tasks
    await asyncio.to_thread(adapter.resume_typing_for_chat, '77')
    await adapter.send_typing('77')
    current = adapter._typing_tasks['77']
    gate.set()
    await clock.parked()
    assert wire.posts == [('POST', '/api/v10/channels/77/typing')]
    assert adapter._typing_tasks['77'] is current and not current.done()
