"""Canonical messaging children need consent after their spawning claim settles."""
import asyncio
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.run_turn_runner_approval_transport import approval_transport
from gateway.session_authority import initialize_session_authority
from gateway.session_contract import Principal, SessionRef
from gateway.session_results import execution_result
from gateway.turn_context import TurnContext
from tools import approval, async_delegation as ad
from tools.approval_context import set_current_session_key, reset_current_session_key


@pytest.mark.asyncio
@pytest.mark.parametrize('scenario', ['approve', 'rollover', 'interrupt', 'clear', 'send_revoke',
                                     'actor', 'token', 'native_stale', 'generation_bypass', 'rpc', 'replay',
                                     'profile', 'incarnation', 'deny', 'native_deny', 'foreground_coexist'])
async def test_child_approval_after_parent_settled(tmp_path, monkeypatch, scenario):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    monkeypatch.setattr(approval.approval_context, '_get_approval_mode', lambda: 'manual')
    monkeypatch.setattr(approval.approval_context, '_get_approval_timeout', lambda: 5)
    monkeypatch.setattr(approval, '_presence', lambda cb=None: (None, False, True, False))
    ad._reset_for_tests()
    monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', tmp_path / 'voice.json')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='dispatch')
    ask, done = threading.Event(), threading.Event()
    delivered = asyncio.Event()
    decisions, cards, owners = [], [], []
    loop = asyncio.get_running_loop()

    async def send(chat_id, content, **kwargs):
        cards.append((chat_id, content, kwargs))
        delivered.set()
        if scenario == 'send_revoke':
            approval.clear_session(route)
        return SendResult(success=True, message_id='approval-card')

    async def turn(event, *_args, **_kwargs):
        route = runner._session_key_for_source(event.source)
        entry = runner.session_store.lookup_by_session_key(route)
        ctx = TurnContext(session_key=route, session_id=entry.session_id, source=event.source,
            _loop_for_step=loop, _status_adapter=adapter, _status_chat_id=event.source.chat_id,
            _status_thread_metadata={}, _run_still_current=lambda: False)
        presenter = TurnRunner(runner, ctx)
        assert presenter._approval_owner is not None
        owners.append(presenter._approval_owner)
        approval.register_gateway_notify(route, approval_transport(presenter))

        def dispatch():
            token = set_current_session_key(route)
            try:
                def child():
                    try:
                        assert ask.wait(5)
                        decisions.append(approval.check_execute_code_guard("print('safe fixture')", 'local'))
                        return {'status': 'completed'}
                    finally:
                        done.set()
                result = ad.dispatch_async_delegation(goal='approval fixture', context=None,
                    toolsets=None, role='leaf', model=None, session_key=route,
                    parent_session_id=entry.session_id, runner=child)
                assert result['status'] == 'dispatched'
            finally:
                reset_current_session_key(token)
        await asyncio.to_thread(dispatch)
        approval.unregister_gateway_notify(route)
        execution_result.get()['result'] = {'final_response': 'parent done', 'messages': []}
        return 'parent done'

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    event = MessageEvent(text='start child', source=adapter.build_source(
        chat_id='15551234567@s.whatsapp.net', user_id='15551234567@s.whatsapp.net', chat_type='dm'),
        message_id='start-child')
    try:
        route = runner._session_key_for_source(event.source)
        owned, reply = await runner._hm_authority_admission(event, event.source, route, False)
        assert owned and reply == 'parent done'
        _, sid, generation = owners[0]
        ref = SessionRef('default', sid)
        assert authority._handle(ref).execution_state == 'idle'
        ask.set()
        try:
            await asyncio.wait_for(delivered.wait(), 3)
        except TimeoutError:
            pytest.fail(f'No background approval delivered after canonical parent settled: {decisions}')
        from tests.waiting import async_wait_until
        await async_wait_until(lambda: authority.sessions[sid].controls.pending or done.is_set(),
                               timeout=3, what='canonical approval publication')
        if scenario == 'send_revoke':
            assert await asyncio.to_thread(done.wait, 5)
            assert decisions[0]['approved'] is False
            assert not authority.sessions[sid].controls.snapshot(sid, None)
            return
        assert not done.is_set(), decisions
        prompts = authority.sessions[sid].controls.snapshot(sid, None)
        assert len(prompts) == 1
        prompt = prompts[0]
        assert prompt['dispatch_owner']
        assert 'execution_generation' not in prompt
        actor = Principal('owner', 'default', frozenset({'session:read', 'session:approve'}), 'viewer')
        authority.sessions[sid].subscribers['viewer'] = actor
        from hermes_state_runtime import RuntimeStoreError
        if scenario == 'rollover':
            async def successor(*args, **kwargs):
                execution_result.get()['result'] = {'final_response': 'successor done', 'messages': []}
                return 'successor done'
            monkeypatch.setattr(runner, '_handle_message_with_agent', successor)
            next_event = MessageEvent(text='successor', source=event.source, message_id='next-turn')
            owned, reply = await runner._hm_authority_admission(next_event, event.source, route, False)
            assert owned and reply == 'successor done'
            assert authority._handle(ref).execution_generation > generation
            assert authority.sessions[sid].controls.snapshot(sid, None)[0]['dispatch_owner'] == prompt['dispatch_owner']
        if scenario == 'actor':
            stranger = Principal('stranger', 'default', frozenset({'session:read'}), 'stranger')
            with pytest.raises(RuntimeStoreError):
                await authority.respond(stranger, ref, None, prompt['prompt_id'], {'choice': 'once'},
                                        dispatch_owner=prompt['dispatch_owner'])
            assert not done.is_set()
        if scenario == 'profile':
            with pytest.raises(RuntimeStoreError):
                await authority.respond(actor, SessionRef('other', sid), None, prompt['prompt_id'],
                                        {'choice': 'once'}, dispatch_owner=prompt['dispatch_owner'])
            assert not done.is_set()
        if scenario == 'incarnation':
            old_live = authority.sessions[sid]
            from copy import copy
            authority.sessions[sid] = copy(old_live)
            with pytest.raises(RuntimeStoreError):
                await authority.respond(actor, ref, None, prompt['prompt_id'], {'choice': 'once'},
                                        dispatch_owner=prompt['dispatch_owner'])
            authority.sessions[sid] = old_live
            assert not done.is_set()
        if scenario == 'replay':
            assert approval.resolve_gateway_approval(route, 'once', request_id='stale-request') == 0
            assert not done.is_set()
        if scenario == 'token':
            with pytest.raises(RuntimeStoreError):
                await authority.respond(actor, ref, None, prompt['prompt_id'], {'choice': 'once'},
                                        dispatch_owner='wrong-owner')
            assert not done.is_set()
        if scenario == 'native_stale':
            assert approval.resolve_gateway_approval(route, 'once') == 0
            assert not done.is_set()
        if scenario in ('interrupt', 'clear'):
            if scenario == 'interrupt':
                ad.interrupt_all()
            else:
                approval.clear_session(route)
            with pytest.raises(RuntimeStoreError):
                await authority.respond(actor, ref, None, prompt['prompt_id'], {'choice': 'once'},
                                        dispatch_owner=prompt['dispatch_owner'])
            assert await asyncio.to_thread(done.wait, 5)
            assert decisions[0]['approved'] is False
            return
        if scenario == 'generation_bypass':
            with pytest.raises(RuntimeStoreError):
                authority.sessions[sid].controls.respond(sid, generation, prompt['prompt_id'], {'choice': 'once'})
            assert not done.is_set()
        if scenario == 'rpc':
            from gateway.session_controls import AuthorityConnection
            connection = AuthorityConnection.__new__(AuthorityConnection)
            connection.authority, connection.actor = authority, actor
            connection.subscriptions = {}
            result = await AuthorityConnection.respond(connection, ref, {
                'session_id': sid, 'prompt_id': prompt['prompt_id'], 'choice': 'once',
                'dispatch_owner': prompt['dispatch_owner']})
            assert result['status'] == 'resolved'
            assert await asyncio.to_thread(done.wait, 5)
            assert decisions[0]['approved'] is True
            return
        if scenario in ('native_stale', 'native_deny'):
            command = 'deny' if scenario == 'native_deny' else 'approve'
            response_event = MessageEvent(text=f'/{command} request:' + prompt['prompt_id'], source=event.source)
            await getattr(runner, f'_handle_{command}_command')(response_event)
            assert await asyncio.to_thread(done.wait, 5)
            assert decisions[0]['approved'] is (scenario != 'native_deny')
            return
        if scenario == 'foreground_coexist':
            controls = authority.sessions[sid].controls
            controls.pending['foreground'] = (route, {'kind': 'approval', 'prompt_id': 'foreground',
                'execution_generation': generation, 'choices': ['once', 'deny']})
            controls.remote_responders['foreground'] = lambda *args: None
        choice = 'deny' if scenario == 'deny' else 'once'
        result = await authority.respond(actor, ref, None, prompt['prompt_id'], {'choice': choice},
                                          dispatch_owner=prompt['dispatch_owner'])
        assert result['status'] == 'resolved'
        assert await asyncio.to_thread(done.wait, 5)
        assert decisions[0]['approved'] is (scenario != 'deny')
        assert len(cards) == 1
        if scenario == 'approve':
            await async_wait_until(
                lambda: any(record['status'] == 'completed' for record in ad.list_async_delegations()),
                timeout=3, what='successful async child finalization')
            assert not authority.sessions[sid].controls.snapshot(sid, None)
        if scenario == 'foreground_coexist':
            assert 'foreground' in authority.sessions[sid].controls.pending
    finally:
        ask.set()
        ad.interrupt_all()
        await asyncio.to_thread(done.wait, 5)
        ad._reset_for_tests()
        authority.db.close()
