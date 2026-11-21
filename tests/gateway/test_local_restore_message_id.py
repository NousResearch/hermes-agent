"""A live local session restores again after a turn stamps its source's ``message_id`` (#143537).

``message_id`` belongs to the triggering message, not the conversation; the receipt-built source never
carries it, so comparing whole source dicts refused every claim after the first one.
"""

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_live_restore_ignores_message_id_but_not_a_foreign_user(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from hermes_state_runtime import RuntimeStoreError
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path / 'fixture'), instance_id='owner')
    viewer = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': str(tmp_path / 'fixture')})
    ref = create_local_session(authority, viewer.actor, {'request_id': 'r', 'source': 'cli',
                                                         'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    restore_local_session(authority, ref.session_id)
    live = authority.sessions[ref.session_id]

    live.source.message_id = 'first-turn'
    assert restore_local_session(authority, ref.session_id).session_id == ref.session_id

    live.source.user_id = 'someone-else'
    with pytest.raises(RuntimeStoreError):
        restore_local_session(authority, ref.session_id)
