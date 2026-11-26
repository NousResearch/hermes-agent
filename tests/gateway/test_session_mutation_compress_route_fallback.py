"""Canonical ``/compress`` follows the turn's fallback chain and preserves typed refusals."""
import json
from types import SimpleNamespace

import pytest

from gateway.session_local import create_local_session
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages
from tests.gateway.test_session_follow_profile_config import _owner

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')


async def _compress(owner, db, sid):
    row = session_row(db, sid)
    return await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
        'session_id': sid, 'request_id': 'compress',
        'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation'],
        'operation': 'compress', 'payload': {}}})


@pytest.mark.asyncio
@pytest.mark.parametrize('failure, expected_reason, expected_field, expected_detail', [
    ('auth', 'compression_provider_unavailable', 'runtime', None),
    ('stopped_endpoint', 'compression_provider_unavailable', 'runtime', None),
    ('missing_policy', 'storage_unavailable', 'policy', 'no restored policy'),
    ('missing_launch_key', 'launch_credentials_unavailable', 'model.api_key', None),
])
async def test_compress_returns_the_appropriate_resolution_refusal(
    tmp_path, failure, expected_reason, expected_field, expected_detail,
):
    """Provider failures get a named refusal; authority failures survive with their wire details."""
    from hermes_cli.auth import AuthError
    from persistence.core.errors import AuthorityRefusal

    def resolver(**kwargs):
        raise {
            'auth': AuthError('primary provider quota exhausted', provider='openai-codex', retryable=True),
            'stopped_endpoint': ValueError('local model server stopped'),
            'missing_policy': AuthorityRefusal('storage_unavailable', field='policy', detail='no restored policy'),
            'missing_launch_key': AuthorityRefusal('launch_credentials_unavailable', field='model.api_key'),
        }[failure]

    store, runner, authority, owner = _owner(tmp_path)
    runner._resolve_session_agent_runtime = resolver
    try:
        ref = create_local_session(authority, owner.actor, dict(request_id='route', source='cli', cwd=str(tmp_path),
                                                                model='frozen', toolsets=[]))
        reply = await _compress(owner, authority.persistence, ref.session_id)
        assert 'result' not in reply, reply
        assert reply['error']['message'] == expected_reason, reply
        assert reply['error']['data']['reason'] == expected_reason, reply
        assert reply['error']['data']['field'] == expected_field, reply
        assert reply['error']['data'].get('detail') == expected_detail, reply
    finally:
        await owner.close()
        store.close_all_db_handles()


class _AgentRouteResolved(Exception):
    """Stop the turn just before configuring its agent, after normal route resolution."""


def _run_turn_until_agent_config(runner, live):
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_policy import policy_for_source, policy_scope

    authority = runner.session_authority
    policy = policy_for_source(runner, live.source)
    turn = TurnRunner.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = SimpleNamespace(source=live.source, session_key=live.route, user_config=policy.config(authority),
                                message='next prompt', inbound_message_id=None, enabled_toolsets=[])
    turn._combined_ephemeral_prompt = lambda: None
    turn._setup_stream_consumer = lambda _platform: (None, None, None, False)

    def resolved(_message, model, runtime, **kwargs):
        raise _AgentRouteResolved(model, runtime['provider'])

    runner._resolve_turn_agent_config = resolved
    with policy_scope(policy, authority=authority):
        return turn._run_sync_scoped()


@pytest.mark.asyncio
@pytest.mark.parametrize('backup', ['anthropic', None], ids=['replaced-chain', 'removed-chain'])
async def test_profile_fallback_edit_reaches_compress_and_turn_before_agent_init(tmp_path, monkeypatch, backup):
    """A pinned Desktop session keeps its primary but not the receipt's obsolete backup."""
    from agent.context_compressor import ContextCompressor
    from gateway import run
    from gateway.session_policy import policy_for_source
    from hermes_cli.auth import AuthError
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    primary = {'model': {'provider': 'openai-codex', 'default': 'gpt-5.6-sol'}}
    legacy = {'provider': 'nous', 'model': 'old-backup'}
    config_path = home / 'config.yaml'
    config_path.write_text(json.dumps({**primary, 'fallback_model': legacy}), encoding='utf-8')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda _key: None
    runner._fallback_model = None
    runner._refresh_fallback_model = run.GatewayRunner._refresh_fallback_model.__get__(runner)
    runner._resolve_session_agent_runtime = run.GatewayRunner._resolve_session_agent_runtime.__get__(runner)
    attempted, compressor_routes = [], []

    def resolve_credentials(*, requested, target_model, **kwargs):
        attempted.append((requested, target_model))
        if requested == 'openai-codex' or (requested == 'nous' and backup is not None):
            raise AuthError('test quota exhausted', provider=requested, retryable=True)
        assert requested in {'nous', 'anthropic'}
        return {'provider': requested, 'base_url': 'https://relay.example.com/v1',
                'api_mode': 'chat_completions', 'api_key': 'fake-key'}

    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', resolve_credentials)
    monkeypatch.setattr(ContextCompressor, '__init__',
                        lambda self, model, **kw: compressor_routes.append((model, kw['provider'])))
    monkeypatch.setattr(ContextCompressor, 'compress', lambda self, messages, **kw: messages)
    try:
        created = await owner.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'desktop', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'gpt-5.6-sol',
            'follow_profile_config': False, 'toolsets': []}})
        assert 'result' in created, created
        sid = created['result']['session_id']
        live = authority.sessions[sid]
        policy = policy_for_source(runner, live.source)
        assert not policy.follows_profile
        assert get_fallback_chain(policy.config(authority)) == [legacy]
        for i in range(4):
            seed_messages(authority.persistence, sid, {'role': 'user', 'content': f'question {i}'})
            seed_messages(authority.persistence, sid, {'role': 'assistant', 'content': f'answer {i}'})

        chain = [{'provider': backup, 'model': 'new-backup'}] if backup else []
        config_path.write_text(json.dumps({**primary, 'fallback_providers': chain}), encoding='utf-8')
        expected_attempts = [('openai-codex', 'gpt-5.6-sol')]
        if backup:
            expected_attempts.append((backup, 'new-backup'))

        attempted.clear()
        runner._pre_agent_fallback_notice = 'another turn notice'
        reply = await _compress(owner, authority.persistence, sid)
        assert runner._pre_agent_fallback_notice == 'another turn notice'
        assert attempted == expected_attempts
        attempted.clear()
        if backup:
            assert reply['result']['status'] == 'noop', reply
            assert compressor_routes == [('new-backup', backup)]
            with pytest.raises(_AgentRouteResolved) as caught:
                _run_turn_until_agent_config(runner, live)
            assert caught.value.args == compressor_routes[0]
        else:
            assert reply['error']['message'] == 'compression_provider_unavailable', reply
            assert compressor_routes == []
            assert _run_turn_until_agent_config(runner, live)['failed'] is True
        assert attempted == expected_attempts
    finally:
        await owner.close()
        store.close_all_db_handles()
