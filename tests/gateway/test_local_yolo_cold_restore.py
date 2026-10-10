"""An explicit /yolo OFF on a local session survives a cold owner restore of its creation receipt."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_cold_restore_keeps_the_persisted_yolo_toggle_and_untoggled_follows_launch(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_managed_worker import _session_yolo
    from tools.approval import clear_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    def runner():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        return SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    first = runner()
    authority = await initialize_session_authority(first, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    params = {'source': 'cli', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    revoked = create_local_session(authority, actor, {**params, 'request_id': 'revoked'})
    untouched = create_local_session(authority, actor, {**params, 'request_id': 'untouched'})
    revoked_route = authority.sessions[revoked.session_id].route
    untouched_route = authority.sessions[untouched.session_id].route
    assert first.session_store.set_session_yolo(revoked_route, False)  # owner `/yolo off` on a --yolo launch

    second = runner()  # idle exit, then `hermes -c`: a fresh owner over the same state.db
    cold = await initialize_session_authority(second, profile_id='fixture', instance_id='second')
    assert {revoked.session_id, untouched.session_id} <= set(cold.sessions)
    launch = SimpleNamespace(yolo=True)
    try:
        assert second.session_store.lookup_by_session_key(revoked_route).yolo is False
        assert _session_yolo(cold, revoked_route, launch) is False, 'cold restore revived a revoked --yolo'
        assert second.session_store.lookup_by_session_key(untouched_route).yolo is None
        assert _session_yolo(cold, untouched_route, launch) is True, 'an untoggled entry lost the launch --yolo'
    finally:
        clear_session(revoked_route)
        clear_session(untouched_route)
