"""Canonical Desktop Retry preserves the failed provider request and completed effects (#137277)."""
import asyncio
import os
from functools import wraps
from pathlib import Path

import pytest

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall, write_hermes_home


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['retry', 'regenerate', 'operator', 'changed', 'restart', 'storage', 'queued', 'follower', 'image-replay'])
async def test_canonical_retry_keeps_completed_tool_and_consumes_token(monkeypatch, mode):
    import model_tools
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_controls import AuthorityConnection
    from run_agent import AIAgent

    effects = []
    ingress_effects = []
    failed = []

    async def lifecycle_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            ingress_effects.append(1)
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", lifecycle_hook)

    def tool(*args, **kwargs):
        effects.append(1)
        return '{"output":"count=1","exit_code":0}'

    def respond(request):
        if not any(row.get('role') == 'tool' for row in request['body']['messages']):
            return ToolCall('terminal', {'command': 'count'}, text='Counting')
        if not failed:
            failed.append(True)
            return Error(429, 'rate limit exceeded')
        return Text('finished using count=1')

    original_run = AIAgent.run_conversation

    @wraps(original_run)
    def run(agent, *args, **kwargs):
        if getattr(agent, "_resume_user_row_id", None) is not None:
            from tools.approval_context import is_viewer_turn_with_approver
            assert is_viewer_turn_with_approver(), "Retry lost its attached viewer approval scope"
        return original_run(agent, *args, **kwargs)

    monkeypatch.setattr(AIAgent, "run_conversation", run)
    original_init = AIAgent.__init__

    def init(agent, *args, **kwargs):
        original_init(agent, *args, **kwargs)
        agent._api_max_retries = 1
        agent._auto_recovery_cycles = 0

    monkeypatch.setattr(AIAgent, '__init__', init)
    monkeypatch.setattr(model_tools, 'handle_function_call', tool)
    home = Path(os.environ['HERMES_HOME'])
    with FakeLLMServer(respond) as provider:
        write_hermes_home(home, provider.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW +
                          'streaming:\n  enabled: false\n')
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-fake')
        monkeypatch.setenv('OPENAI_BASE_URL', provider.base_url)
        if mode == 'storage':
            def unavailable(*args):
                raise OSError('injected unavailable transcript')
            monkeypatch.setattr('gateway.session_retry._history', unavailable)
        runner = GatewayRunner()
        authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='retry-test')

        class Transport:
            def __init__(self):
                self.frames = []

            def write(self, frame):
                self.frames.append(frame)
                return True

        transport = Transport()
        connection = AuthorityConnection(authority, transport, {'user_id': 'tester'})
        owner_connection = connection

        async def rpc(method, **params):
            return await connection.dispatch({'id': method, 'method': method, 'params': params})

        try:
            created = await rpc('session.create', request_id='retry-chat', source='gui',
                                model='fake-model', provider='custom', base_url=provider.base_url,
                                api_key='sk-fake', toolsets=['terminal'])
            assert 'result' in created, created
            sid = created['result']['session_id']
            ref = SessionRef(authority.profile_id, sid)
            if mode == 'image-replay':
                authority.db.append_messages_batch(authority.physical_target(ref), [
                    {'role': 'user', 'content': '[Image attached at: missing.png]\n[screenshot]'},
                    {'role': 'assistant', 'content': 'Earlier image context'},
                ])
            submitted = await rpc('prompt.submit', session_id=sid, input_id='first', text='count once then finish')
            assert 'result' in submitted, submitted
            await asyncio.wait_for(authority.sessions[sid].task, 30)
            assert effects == [1]
            terminal = [f['params']['payload'] for f in transport.frames
                        if f.get('params', {}).get('type') == 'message.complete']
            assert terminal[-1]['status'] == 'error', terminal
            if mode in {'storage', 'image-replay'}:
                assert not terminal[-1].get('retry_token'), terminal
                requests_before = len(provider.main_requests())
                refused = await rpc('prompt.retry', session_id=sid, retry_token='unavailable')
                assert 'error' in refused, refused
                assert effects == [1]
                assert len(provider.main_requests()) == requests_before
                return
            assert terminal[-1].get('retry_token'), terminal
            token = terminal[-1]['retry_token']
            snapshot = await rpc('session.resume', session_id=sid)
            assert snapshot['result']['inflight']['retry_token'] == token
            assert snapshot['result']['inflight']['user'] == 'count once then finish'
            before = authority.db.get_messages(authority.physical_target(ref))
            if mode == 'queued':
                monkeypatch.setattr(authority, '_schedule', lambda *args: None)
                queued = await rpc('prompt.submit', session_id=sid, input_id='new', text='new intent')
                assert 'result' in queued, queued
                snapshot = await rpc('session.resume', session_id=sid)
                assert snapshot['result']['inflight'] is None
                refused = await rpc('prompt.retry', session_id=sid, retry_token=token)
                assert refused['error']['message'] == 'session_busy'
                assert effects == [1]
                return
            if mode in {'changed', 'restart'}:
                if mode == 'changed':
                    authority.db.append_message(authority.physical_target(ref), 'user', 'new intent')
                else:
                    authority.sessions[sid].retry_checkpoint = None
                unchanged = authority.db.get_messages(authority.physical_target(ref))
                refused = await rpc('prompt.retry', session_id=sid, retry_token=token)
                assert refused['error']['message'] == 'retry_checkpoint_unavailable', refused
                assert authority.db.get_messages(authority.physical_target(ref)) == unchanged
                assert effects == [1]
                return
            if mode == 'operator':
                connection = AuthorityConnection(authority, transport, {'user_id': 'operator'}, operator=True)
                attached = await rpc('session.resume', session_id=sid)
                assert 'result' in attached, attached
            if mode == 'regenerate':
                rewind = await rpc('session.mutate', session_id=sid, request_id='regenerate',
                                   expected_revision=authority._handle(ref).revision,
                                   expected_generation=authority._handle(ref).execution_generation, operation='rewind',
                                   payload={'target_message_id': next(row['id'] for row in before if row['role'] == 'user')})
                assert 'result' in rewind, rewind
                response = await rpc('prompt.submit', session_id=sid, input_id='regenerate',
                                     text='count once then finish')
            elif mode == 'follower':
                live = authority.sessions[sid]
                await live.claim_gate.acquire()
                try:
                    response = await rpc('prompt.retry', session_id=sid, retry_token=token)
                    queued = await rpc('prompt.submit', session_id=sid, input_id='follower', text='new intent')
                    assert 'result' in queued, queued
                    assert live.retry_checkpoint['spent']
                finally:
                    live.claim_gate.release()
            else:
                response = await rpc('prompt.retry', session_id=sid, retry_token=token)
            assert 'result' in response, response
            duplicate = await rpc('prompt.retry', session_id=sid, retry_token=token)
            assert 'error' in duplicate, duplicate
            await asyncio.wait_for(authority.sessions[sid].task, 30)
            assert effects == [1] * (2 if mode == 'regenerate' else 1)
            assert ingress_effects == []  # admitted viewer input never repeats the ingress hook
            terminal = [f['params']['payload'] for f in transport.frames
                        if f.get('params', {}).get('type') == 'message.complete']
            assert terminal[-1]['status'] == 'complete', terminal
            requests = provider.main_requests()
            if mode != 'regenerate':
                offset = -1 if mode == 'follower' else 0
                assert requests[-1 + offset]['messages'] == requests[-2 + offset]['messages']
            after = authority.db.get_messages(authority.physical_target(ref))
            assert sum(row['role'] == 'user' for row in after) == (2 if mode == 'follower' else 1)
            if mode != 'regenerate':
                assert [row['id'] for row in after[:len(before)]] == [row['id'] for row in before]
        finally:
            await connection.close()
            if owner_connection is not connection:
                await owner_connection.close()
