"""An API session receipt minted under an older home of its store restores, under its own profile only.

A ``state.db`` that moved with its profile (another machine, a moved home) keeps API receipts whose
``profile_id`` names the old path. Restoring one is that store's own row, but the receipt's stored
origin still places the turn's runtime scope, so one naming another profile is refused.
"""
import json

import pytest

from hermes_state_local import API_BINDING_PREFIX
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


def _rehome(authority, session_id, **origin):
    key = API_BINDING_PREFIX + session_id
    receipt = json.loads(authority.db.get_meta(key))
    receipt['profile_id'] = '/departed/host/profiles/alpha'
    receipt['entry']['origin'].update(origin)
    authority.db.set_meta(key, json.dumps(receipt))


@pytest.mark.asyncio
async def test_moved_api_receipt_restores_only_under_its_own_profile(tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_api import bind_api_session, restore_api_session
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    runner = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        alpha = runner.session_authorities.for_home(homes[1][1])
        for sid in ('api-moved', 'api-sibling'):
            alpha.db.create_session(sid, source='api_server')
            bind_api_session(alpha, sid)
        _rehome(alpha, 'api-moved')
        _rehome(alpha, 'api-sibling', profile='beta')
        alpha.sessions.clear()

        restored = restore_api_session(alpha, 'api-moved')
        assert alpha.sessions[restored.session_id].source.profile == 'alpha'
        with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
            restore_api_session(alpha, 'api-sibling')
    finally:
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
