"""The real gateway and agent send startup input in the same first model request."""

import asyncio
from copy import deepcopy
import threading
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('mode, anchor', [('interrupt', 'm2'), ('steer', 'm1')])
@pytest.mark.parametrize('veto_first', [False, True])
@pytest.mark.parametrize('platform_name', ['feishu', 'photon'])
async def test_followup_during_real_gateway_setup_reaches_first_request(
        tmp_path, monkeypatch, mode, anchor, veto_first, platform_name):
    from agent import turn_context
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from plugins.platforms.photon.adapter import PhotonAdapter
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _mock_response
    from tests.waiting import async_wait_until

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('agent.title_generator.maybe_auto_title', lambda *args, **kwargs: None)
    monkeypatch.setattr('agent.turn_api_call._should_stream', lambda _agent: False)
    collect = turn_context._collect_pre_llm_call_context

    def plugin_context(agent, **kwargs):
        if veto_first and kwargs['original_user_message'] == 'first question':
            kwargs['blocks'].append('Synthetic plugin veto')
        return collect(agent, **kwargs)

    monkeypatch.setattr(turn_context, '_collect_pre_llm_call_context', plugin_context)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = (FeishuAdapter if platform_name == 'feishu' else PhotonAdapter)(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters = {adapter.platform: adapter}
    runner._is_user_authorized_for_source = lambda source, **kwargs: True
    monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: mode)
    monkeypatch.setattr(runner, '_effective_busy_text_mode', lambda source: 'interrupt')
    monkeypatch.setattr(runner, '_hm_in_telegram_grace', lambda *args: False)
    monkeypatch.setattr(runner, '_deliver_platform_notice', AsyncMock())
    monkeypatch.setattr('gateway.run_inbound.pending_setup_busy_notice', lambda *args: None)
    monkeypatch.setattr(runner, '_resolve_session_agent_runtime', lambda **kwargs: (
        'synthetic', {'provider': 'openai-compat', 'api_mode': 'chat_completions',
                      'api_key': 'synthetic', 'base_url': 'http://127.0.0.1:1/v1'}))
    requests, sends = [], []

    def model(_agent, kwargs):
        requests.append(deepcopy(kwargs['messages']))
        return _mock_response(content='one answer', finish_reason='stop')

    monkeypatch.setattr(AIAgent, '_interruptible_api_call', model)

    async def send(chat_id, content, reply_to=None, **_kwargs):
        sends.append((content, reply_to))
        return SendResult(success=True, message_id='sent')

    adapter.send = send
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='startup-e2e')
    runner._wire_adapter_handlers(adapter, authorization_check=lambda *args, **kwargs: True)
    source = adapter.build_source(chat_id='chat', chat_type='dm', user_id='user')
    first = MessageEvent(text='first question', source=source, message_id='m1')
    second = MessageEvent(text='also answer this', source=source, message_id='m2')
    entered, release = asyncio.Event(), asyncio.Event()
    resolve = runner._hmwa_resolve_session

    async def delayed_setup(*args, **kwargs):
        result = await resolve(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            await asyncio.wait_for(release.wait(), 15)
        return result

    monkeypatch.setattr(runner, '_hmwa_resolve_session', delayed_setup)

    async def deliver(event):
        if platform_name == 'photon':
            await adapter._dispatch_inbound({
                'messageId': event.message_id, 'platform': 'iMessage',
                'space': {'id': 'chat', 'type': 'dm'}, 'sender': {'id': 'user'},
                'content': {'type': 'text', 'text': event.text},
            })
        else:
            await adapter.handle_message(event)

    try:
        await deliver(first)
        await asyncio.wait_for(entered.wait(), 15)
        await deliver(second)
        sid = next(iter(authority.sessions))
        await async_wait_until(
            lambda: len(authority.persistence.admissions.list(session_id=sid, pending_only=False)) == 2,
            timeout=10, what='both inputs are durably admitted')
        release.set()
        await async_wait_until(lambda: all(not live.task or live.task.done() for live in authority.sessions.values()),
                               timeout=30, what='the gateway session settles')
        await async_wait_until(lambda: len(sends) >= (2 if veto_first else 1),
                               timeout=10, what='the answer is sent')
        assert len(requests) == 1
        assert 'also answer this' in requests[0][-1]['content']
        if veto_first:
            assert sends == [('Synthetic plugin veto', 'm1'), ('one answer', 'm2')]
        else:
            assert 'first question' in requests[0][-1]['content']
            assert sends == [('one answer', anchor)]
    finally:
        release.set()
        runner.session_store.close_all_db_handles()
        if platform_name == 'feishu':
            adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
async def test_stop_during_followup_transcript_write_never_sends_the_stale_request(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from persistence.sqlite.conversation.messages import SQLiteMessageRepository
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from run_agent import AIAgent
    from tests.waiting import async_wait_until

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('agent.title_generator.maybe_auto_title', lambda *args, **kwargs: None)
    monkeypatch.setattr('agent.turn_api_call._should_stream', lambda _agent: False)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.FEISHU: adapter}
    runner._is_user_authorized_for_source = lambda source, **kwargs: True
    monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: 'interrupt')
    monkeypatch.setattr(runner, '_effective_busy_text_mode', lambda source: 'interrupt')
    monkeypatch.setattr(runner, '_hm_in_telegram_grace', lambda *args: False)
    monkeypatch.setattr(runner, '_deliver_platform_notice', AsyncMock())
    monkeypatch.setattr('gateway.run_inbound.pending_setup_busy_notice', lambda *args: None)
    monkeypatch.setattr(runner, '_resolve_session_agent_runtime', lambda **kwargs: (
        'synthetic', {'provider': 'openai-compat', 'api_mode': 'chat_completions',
                      'api_key': 'synthetic', 'base_url': 'http://127.0.0.1:1/v1'}))
    requests = []
    monkeypatch.setattr(AIAgent, '_interruptible_api_call', lambda _agent, kwargs: requests.append(kwargs))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id='sent'))
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='stop-write')
    runner._wire_adapter_handlers(adapter, authorization_check=lambda *args, **kwargs: True)
    source = adapter.build_source(chat_id='chat', chat_type='dm', user_id='user')
    first = MessageEvent(text='first', source=source, message_id='m1')
    second = MessageEvent(text='second', source=source, message_id='m2')
    setup_entered, release_setup = asyncio.Event(), asyncio.Event()
    write_entered, release_write = threading.Event(), threading.Event()
    resolve, write = runner._hmwa_resolve_session, SQLiteMessageRepository.set_message_api_content

    async def delayed_setup(*args, **kwargs):
        result = await resolve(*args, **kwargs)
        setup_entered.set()
        await asyncio.wait_for(release_setup.wait(), 30)
        return result

    def delayed_write(repository, *args, **kwargs):
        if kwargs.get('new_content') is not None and not write_entered.is_set():
            write_entered.set()
            assert release_write.wait(30)
        return write(repository, *args, **kwargs)

    monkeypatch.setattr(runner, '_hmwa_resolve_session', delayed_setup)
    monkeypatch.setattr(SQLiteMessageRepository, 'set_message_api_content', delayed_write)
    try:
        await adapter.handle_message(first)
        await asyncio.wait_for(setup_entered.wait(), 20)
        await adapter.handle_message(second)
        sid = next(iter(authority.sessions))
        await async_wait_until(
            lambda: any(row.outcome == 'steering' for row in authority.persistence.admissions.list(
                session_id=sid, pending_only=False)),
            timeout=20, what='the follow-up transfers to the running turn')
        release_setup.set()
        assert await asyncio.to_thread(write_entered.wait, 20)
        await runner._hm_stop_owner_turn(adapter._event_session_key(first))
        release_write.set()
        await async_wait_until(lambda: all(not live.task or live.task.done() for live in authority.sessions.values()),
                               timeout=30, what='the stopped turn settles')
        assert requests == []
        assert authority.persistence.admissions.list(session_id=sid, pending_only=False)[-1].outcome in (
            'cancelled', 'interrupted')
    finally:
        release_setup.set()
        release_write.set()
        runner.session_store.close_all_db_handles()
        adapter._shutdown_sdk_executor()
