"""Authority-owned replies must retire transport typing without releasing another pause."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_ingress import deliver_response, deliver_settled
from plugins.platforms.discord.adapter import DiscordAdapter
from tests.waiting import async_wait_until


@pytest.fixture
def typing_adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='test-token'))
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock()))
    return adapter


async def start_typing(adapter):
    await adapter.send_typing('77')
    await async_wait_until(lambda: adapter._client.http.request.await_count > 0,
                           what='first Discord typing request')
    task = adapter._typing_tasks['77']
    assert not task.done()
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize('paused', [False, True], ids=['unpaused', 'another-session-paused'])
@pytest.mark.parametrize('outcome', ['normal', 'empty', 'error', 'cancel'])
@pytest.mark.parametrize('recovered', [False, True], ids=['busy-delivery', 'recovered-delivery'])
async def test_authority_delivery_stops_typing_on_exit(typing_adapter, monkeypatch, outcome, paused, recovered):
    adapter = typing_adapter
    source = adapter.build_source(chat_id='77', chat_type='dm', user_id='fixture-user')
    event = MessageEvent(text='work', source=source, message_id='native-message')
    started = asyncio.Event()
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        started.set()
        if outcome == 'error':
            raise RuntimeError('delivery failed')
        if outcome == 'cancel':
            await asyncio.Event().wait()
        return SendResult(success=True, message_id='reply')

    monkeypatch.setattr(adapter, 'send', send)
    task = await start_typing(adapter)
    if paused:
        adapter.pause_typing_for_chat('77')
    response = None if outcome == 'empty' else 'DONE'
    key = 'agent:main:discord:dm:77'
    authority = SimpleNamespace(
        profile_id='default', runner=SimpleNamespace(), native_waiters=set(),
        pending_deliveries={'admission': (adapter, event, key, response, None)})
    delivery = asyncio.create_task(deliver_settled(authority, 'admission') if recovered
                                   else deliver_response(adapter, event, key, response))
    try:
        if outcome == 'cancel':
            await asyncio.wait_for(started.wait(), 5)
            delivery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await delivery
        elif outcome == 'error' and not recovered:
            with pytest.raises(RuntimeError, match='delivery failed'):
                await delivery
        else:
            delivered = await delivery
            if not recovered:
                assert delivered is True, 'typing cleanup must preserve the delivery verdict'
        if recovered:
            assert not authority.pending_deliveries
        assert task.done(), 'authority delivery left persistent transport typing alive'
        assert '77' not in adapter._typing_tasks
        assert ('77' in adapter._typing_paused) is paused
        assert sent == ([] if outcome == 'empty' else ['DONE'])
    finally:
        delivery.cancel()
        await asyncio.gather(delivery, return_exceptions=True)
        await adapter.stop_typing('77')


@pytest.mark.asyncio
async def test_busy_authority_reply_stops_rearmed_typing(typing_adapter, monkeypatch):
    adapter = typing_adapter
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    runner.adapters = {Platform.DISCORD: adapter}
    adapter.gateway_runner = runner
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='typing-owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='77', chat_type='dm', user_id='fixture-user')
    key = runner.session_store._generate_session_key(source)
    event = MessageEvent(text='work', source=source, message_id='native-message')
    adapter._active_sessions[key] = asyncio.Event()
    typing_tasks = []
    sent = []

    async def turn(event, source, quick_key, generation):
        await start_typing(adapter)
        await runner._hmwa_stop_typing_for_turn(event, source)
        # A refresh during post-turn persistence can re-arm the transport loop.
        await adapter.send_typing(source.chat_id)
        typing_tasks.append(adapter._typing_tasks[source.chat_id])
        return 'WORK_DONE'

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id='reply')

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    monkeypatch.setattr(adapter, 'send', send)
    try:
        await asyncio.wait_for(adapter.handle_message(event), 10)
        await async_wait_until(lambda: sent and not adapter._background_tasks,
                               what='authority reply and busy observer completion')
        from hermes_state_runtime import list_session_admissions
        rows = [row for session_id in authority.sessions
                for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False)]
        assert [row['status'] for row in rows] == ['terminal']
        assert sent == ['WORK_DONE']
        assert typing_tasks[0].done(), 'busy reply left transport typing alive'
        assert '77' not in adapter._typing_tasks
    finally:
        await adapter.stop_typing('77')
