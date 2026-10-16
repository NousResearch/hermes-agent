"""Deleting a local session drops the launch key and frozen config secrets it held in memory."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_delete_prunes_launch_and_config_secrets(tmp_path, monkeypatch):
    from gateway import run, session_mutations
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    config = {'platform_toolsets': {'cli': []}, 'model': {
        'provider': 'custom', 'api_key': 'config-secret', 'base_url': 'http://127.0.0.1:9/v1'}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: config)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    connection = AuthorityConnection(owner, object(), {'user_id': 'owner'})
    def create(request_id):
        return create_local_session(owner, connection.actor, {'request_id': request_id, 'source': 'cli',
            'cwd': str(tmp_path), 'model': 'm', 'toolsets': [], 'api_key': 'launch-' + request_id})
    try:
        doomed, kept = create('doomed'), create('kept')
        assert len(owner._local_launch_keys) == 2 and len(owner._local_config_secrets) == 2
        handle = owner._handle(doomed)
        await session_mutations.mutate_session(owner, connection.actor, doomed, dict(
            session_id=doomed.session_id, request_id='delete', operation='delete', payload={},
            expected_revision=handle.revision, expected_generation=handle.execution_generation))
        assert owner.db.get_session(doomed.session_id) is None
        assert 'launch-doomed' not in owner._local_launch_keys.values()
        assert list(owner._local_launch_keys.values()) == ['launch-kept']
        assert len(owner._local_config_secrets) == 1
        assert all(kept.session_id in ref for ref in owner._local_config_secrets)
    finally:
        await connection.close()
        owner.db.close()
