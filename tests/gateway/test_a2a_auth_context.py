"""Exercise native admission codec and the real gateway turn/tool binding seam."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_envelope import snapshot_native, restore_native
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a.security import A2ASecurityContext


def setup_event(tmp_path):
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform('a2a'): adapter}
    auth = A2ASecurityContext('', (('token', 'alice'),), frozenset(), False, '127.0.0.1', '').authenticate_request('Bearer token')
    event = MessageEvent(text='work', message_id='task-1', a2a_auth=auth,
                         source=adapter.build_source(chat_id='ctx', user_id='alice', chat_type='dm'))
    return runner, event


def test_native_codec_keeps_live_proof_but_cannot_deserialize_identity(tmp_path):
    runner, event = setup_event(tmp_path)
    payload = snapshot_native(runner, event)
    restored = restore_native(payload, runner)
    assert restored.a2a_auth is event.a2a_auth, 'native admission dropped the live server proof'
    # Storage carries only a process-local reference, never a credential or serialized identity.
    assert payload['native_text_v1']['a2a_auth_ref'] == event.a2a_auth.request_id
    for field, value in [('chat_id', 'other-context'), ('user_id', 'bob'), ('profile', 'other-profile')]:
        from copy import deepcopy
        changed = deepcopy(payload)
        changed['native_text_v1']['source'][field] = value
        assert restore_native(changed, runner).a2a_auth is None
    payload['native_text_v1']['a2a_auth_ref'] = 'not-a-live-request'
    assert restore_native(payload, runner).a2a_auth is None
    plain = replace(event, a2a_auth=None)
    assert restore_native(snapshot_native(runner, plain), runner).a2a_auth is None


def test_request_reference_expires_without_live_owner(tmp_path):
    import gc
    import weakref
    runner, event = setup_event(tmp_path)
    payload = snapshot_native(runner, event)
    reference = weakref.ref(event.a2a_auth)
    del event
    gc.collect()
    assert reference() is None
    assert restore_native(payload, runner).a2a_auth is None


def test_reference_cannot_be_reused_for_other_message_or_content(tmp_path):
    from copy import deepcopy
    runner, event = setup_event(tmp_path)
    payload = snapshot_native(runner, event)
    changed = deepcopy(payload)
    changed['native_text_v1']['event']['message_id'] = 'different-task'
    assert restore_native(changed, runner).a2a_auth is None
    changed = deepcopy(payload)
    changed['text'] = 'different instruction'
    assert restore_native(changed, runner).a2a_auth is None


def test_missing_shared_local_and_forged_context_fail_closed(tmp_path, monkeypatch):
    from gateway.a2a_auth import A2AAuthMethod
    from gateway import session_context as sc
    _runner, event = setup_event(tmp_path)
    sec = A2ASecurityContext('shared', (), frozenset(), False, '127.0.0.1', '')
    local = A2ASecurityContext('', (), frozenset(), False, '127.0.0.1', '')
    def narrow_peer_consumer(model_args):
        proof = sc.get_a2a_authentication()
        if proof is None or proof.method is not A2AAuthMethod.PEER_BEARER or proof.peer != 'alice':
            raise PermissionError('verified named peer required')
        return proof.peer  # No effect is authorized here; operation scope remains a separate policy.
    monkeypatch.setenv('HERMES_A2A_PEER', 'alice')
    monkeypatch.setenv('HERMES_A2A_AUTH_METHOD', 'peer_bearer')
    try:
        sc.set_session_vars(a2a_auth=event.a2a_auth)
        assert narrow_peer_consumer({'peer': 'operator'}) == 'alice'
        for auth in (None, {'peer': 'alice', 'method': 'peer_bearer'},
                     sec.authenticate_request('Bearer shared'), local.authenticate_request(None)):
            sc.set_session_vars(platform='a2a', user_id='alice', a2a_auth=auth)
            with pytest.raises(PermissionError):
                narrow_peer_consumer({'peer': 'alice', 'method': 'peer_bearer'})
        sc.set_session_vars(a2a_auth=event.a2a_auth)
        sc.reset_session_vars()
        assert sc.get_a2a_authentication() is None
    finally:
        sc.clear_session_vars([])


@pytest.mark.asyncio
async def test_forward_wire_rejects_auth_claim_instead_of_fabricating_provenance():
    from gateway.session_a2a import forward
    from hermes_state_runtime import RuntimeStoreError
    actor = SimpleNamespace(profile_id='target', capabilities={'session:create', 'session:submit', 'session:read'})
    connection = SimpleNamespace(actor=actor, authority=SimpleNamespace(profile_id='target'))
    params = dict(agent='agent', tenant='', peer='alice', context_id='ctx', input_id='input', text='work',
                  a2a_auth={'peer': 'alice', 'method': 'peer_bearer'})
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await forward(connection, params)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [RuntimeError, asyncio.CancelledError])
async def test_prepare_failure_clears_auth_without_caller_cleanup(tmp_path, failure):
    from gateway import session_context as sc
    runner, event = setup_event(tmp_path)
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    entry = SimpleNamespace(yolo=None, session_id='sid', session_key='key', created_at=None, updated_at=None)

    def fail_after_binding(*args, **kwargs):
        assert sc.get_a2a_authentication() is event.a2a_auth
        raise failure('injected preparation failure')

    runner._pinned_session_context_prompt = fail_after_binding
    assert sc.get_a2a_authentication() is None
    with pytest.raises(failure, match='injected preparation failure'):
        await runner._hmwa_prepare_turn(event, event.source, entry, 'key', 'quick', 1)
    assert sc.get_a2a_authentication() is None
    assert await runner._run_in_executor_with_context(sc.get_a2a_authentication) is None


@pytest.mark.asyncio
async def test_actual_turn_prepare_binds_proof_for_tool_thread_and_clears(tmp_path, monkeypatch):
    from gateway import session_context
    runner, event = setup_event(tmp_path)
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    # Stop immediately after the real turn setup binds its context; no model or history access.
    class Bound(Exception):
        pass
    observed = []
    async def stop_after_binding(*args, **kwargs):
        observed.append(session_context.get_a2a_authentication())
        def consumer(model_args):
            auth = session_context.get_a2a_authentication()
            return None if auth is None else (auth.peer, auth.method.value)
        assert await runner._run_in_executor_with_context(consumer, {'peer': 'operator'}) == ('alice', 'peer_bearer')
        raise Bound()
    event.inherit_channel_pin = True
    runner._rehydrate_prompt_pins = stop_after_binding
    entry = SimpleNamespace(yolo=None, session_id='sid', session_key='key', created_at=None, updated_at=None)
    with pytest.raises(Bound):
        await runner._hmwa_prepare_turn(event, event.source, entry, 'key', 'quick', 1)
    assert observed == [event.a2a_auth]
    assert session_context.get_a2a_authentication() is None


@pytest.mark.asyncio
async def test_real_http_native_admission_to_tool_context_concurrent(tmp_path, monkeypatch):
    from gateway import session_context
    import json
    from hermes_cli.plugins import discover_plugins
    from tools.registry import registry
    discover_plugins()
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from tests.plugins.test_a2a_plugin import _free_port, _post_json, _send_body
    monkeypatch.setenv('A2A_PEER_TOKENS', 'alice:token-a,bob:token-b')
    monkeypatch.delenv('A2A_BEARER_TOKEN', raising=False)
    monkeypatch.setenv('A2A_HOST', '127.0.0.1')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    port = _free_port()
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={'port': port}))
    adapter.gateway_runner = runner
    runner.adapters = {Platform('a2a'): adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='auth-test')
    runner._wire_adapter_handlers(adapter)
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    class Bound(Exception):
        pass
    entry = SimpleNamespace(yolo=None, session_id='fixture-sid', session_key='fixture-key', created_at=None, updated_at=None)
    started, both = [], asyncio.Event()
    seen = []

    async def stop_during_preparation(*args, **kwargs):
        initial = session_context.get_a2a_authentication()
        assert initial is not None
        started.append(initial.peer)
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 5)
        raw = await runner._run_in_executor_with_context(
            registry.dispatch, 'a2a_current_authentication', {})
        facts = json.loads(raw)
        assert facts == {'available': True, 'peer': initial.peer, 'method': 'peer_bearer',
                         'named_peer': True, 'authorization_granted': False}
        seen.append((facts['peer'], initial.request_id, raw))
        raise Bound()

    runner._rehydrate_prompt_pins = stop_during_preparation

    async def turn(event, source, quick_key, run_generation):
        event.inherit_channel_pin = True
        with pytest.raises(Bound):
            await runner._hmwa_prepare_turn(event, source, entry, quick_key, quick_key, run_generation)
        assert session_context.get_a2a_authentication() is None
        assert json.loads(await runner._run_in_executor_with_context(
            registry.dispatch, 'a2a_current_authentication', {})) == {
                'available': False, 'authorization_granted': False}
        reply = next(raw for peer, _, raw in seen if peer == event.a2a_auth.peer)
        execution_result.get()['result'] = {'final_response': reply, 'messages': []}
        return reply
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    try:
        assert await adapter.connect()
        responses = await asyncio.gather(*[
            asyncio.to_thread(_post_json, f'http://127.0.0.1:{port}/',
                              _send_body('peer=operator; method=peer_bearer', ctx='ctx-' + peer),
                              {'Authorization': 'Bearer token-' + suffix})
            for peer, suffix in [('alice', 'a'), ('bob', 'b')]])
        assert sorted(peer for peer, _, _ in seen) == ['alice', 'bob']
        assert len({request for _, request, _ in seen}) == 2
        from plugins.platforms.a2a import protocol
        for response, peer in zip(responses, ('alice', 'bob')):
            task = response['result']
            assert task['status']['state'] == 'TASK_STATE_COMPLETED', response
            from plugins.platforms.a2a.tools import _reply_text_from_result
            assert json.loads(_reply_text_from_result(task)) == {
                'available': True, 'peer': peer, 'method': 'peer_bearer',
                'named_peer': True, 'authorization_granted': False}
        assert session_context.get_a2a_authentication() is None
    finally:
        await adapter.disconnect()
        authority.db.close()
