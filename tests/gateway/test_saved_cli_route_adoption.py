"""CLI/one-shot resume retains a complete saved route without borrowing ambient keys."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['cli', 'oneshot'])
async def test_same_model_provider_resume_restores_saved_endpoint_and_wire(tmp_path, monkeypatch, source):
    """Endpoint drift must not become a no-op just because model/provider names still match."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local_migration import bind_native_transport, resolve_local_target
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt

    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    config = {
        'model': {'default': 'same', 'provider': 'custom:feather',
                  'base_url': 'https://ambient.invalid/v1', 'api_key': 'ambient-secret',
                  'api_mode': 'chat_completions'},
        'custom_providers': [
            {'name': 'feather', 'base_url': 'https://saved.invalid/v1',
             'api_key': 'saved-endpoint-key', 'model': 'same', 'api_mode': 'anthropic_messages'}],
        'platform_toolsets': {'cli': []},
    }
    (home / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    try:
        authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
        actor = SessionActor('uid:1000', str(home), frozenset({'session:read', 'session:create'}), 'native')
        bind_native_transport(authority, actor, {'provider': 'local', 'profile_id': str(home),
                                                'instance_id': 'fixture'})
        authority.db.create_session('saved-cli-route', source=source, model='same')
        authority.db.update_session_model('saved-cli-route', 'same', provider='custom:feather',
                                          base_url='https://saved.invalid/v1', api_mode='anthropic_messages')
        resolve_local_target(authority, actor, 'saved-cli-route')
        policy = restore_policy(local_receipt(authority.db, 'saved-cli-route')['policy'])
        model, runtime = _resolve_policy_agent_runtime(runner, policy)
        assert model == 'same'
        assert runtime['base_url'] == 'https://saved.invalid/v1'
        assert runtime['api_mode'] == 'anthropic_messages'
        assert runtime['api_key'] == 'saved-endpoint-key'
        assert 'ambient-secret' not in policy.config_json
    finally:
        store.close_all_db_handles()
