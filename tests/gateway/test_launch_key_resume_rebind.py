"""A launch-only key revoked by an owner restart can be re-supplied on session.resume, and only
the key the session was launched with (its durable fingerprint); never persisted."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_resume_rebinds_only_the_launch_key_after_restart(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from gateway.session_policy import launch_key, policy_for_source
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())

    def runner():
        owner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                                _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None)
        owner._adapter_for_source = lambda source: owner.adapters.get(source.platform)
        return owner

    raw = 'UNIQUE-LAUNCH-ONLY-KEY'
    first = await initialize_session_authority(runner(), profile_id=str(tmp_path), instance_id='first')
    actor = AuthorityConnection(first, object(), {'user_id': 'owner'}).actor
    keyed = create_local_session(first, actor, {'request_id': 'keyed', 'cwd': str(tmp_path), 'model': 'm',
                                                'toolsets': [], 'base_url': 'http://127.0.0.1:9/v1', 'api_key': raw})
    plain = create_local_session(first, actor, {'request_id': 'plain', 'cwd': str(tmp_path), 'model': 'm',
                                                'toolsets': []})
    with store._db._read_ctx() as conn:
        assert not [row for row in conn.execute('SELECT value FROM state_meta') if raw in row[0]]
    cold = await initialize_session_authority(runner(), profile_id=str(tmp_path), instance_id='restarted')
    connection = AuthorityConnection(cold, object(), {'user_id': 'owner'})

    async def resume(ref, **extra):
        reply = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                           'params': {'session_id': ref.session_id, **extra}})
        return reply.get('error', {}).get('message') or 'ok'

    def key_of(ref):
        return launch_key(cold, policy_for_source(cold.runner, cold.sessions[ref.session_id].source))

    try:
        assert await resume(keyed) == 'ok'
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
            key_of(keyed)
        assert await resume(keyed, api_key='a-different-key') == 'admission_conflict'
        assert await resume(plain, api_key=raw) == 'admission_conflict'  # no launch key to re-supply
        assert await resume(keyed, api_key=raw) == 'ok'
        assert key_of(keyed) == raw and key_of(plain) is None
        assert await resume(keyed, api_key='a-different-key') == 'admission_conflict'
    finally:
        await connection.close()
        store._db.close()
