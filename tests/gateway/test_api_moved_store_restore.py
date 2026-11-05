"""An API session receipt minted under an older home of its store restores, under its own profile only.

A ``state.db`` that moved with its profile (another machine, a moved home) keeps API receipts whose
``profile_id`` names the old path. Restoring one is that store's own row, but the receipt's stored
origin still places the turn's runtime scope, so one naming another profile is refused.
"""
import json

import pytest

from gateway.session_persistence import authority_persistence
from persistence.conversation.sessions import NewSession
from persistence.core.errors import AuthorityRefusal
from persistence_assembly import close_stores_under
from tests.fixtures.storage_holder import api_binding_key
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


def _rehome(authority, session_id, **origin):
    meta, key = authority_persistence(authority).state_meta, api_binding_key(session_id)
    receipt = json.loads(meta.get(key))
    receipt['profile_id'] = '/departed/host/profiles/alpha'
    receipt['entry']['origin'].update(origin)
    meta.set(key, json.dumps(receipt))


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
            authority_persistence(alpha).sessions.create(NewSession(session_id=sid, source='api_server'))
            bind_api_session(alpha, sid)
        _rehome(alpha, 'api-moved')
        _rehome(alpha, 'api-sibling', profile='beta')
        alpha.sessions.clear()

        restored = restore_api_session(alpha, 'api-moved')
        assert alpha.sessions[restored.session_id].source.profile == 'alpha'
        with pytest.raises(AuthorityRefusal, match='profile_mismatch'):
            restore_api_session(alpha, 'api-sibling')
    finally:
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            close_stores_under(authority.persistence.scope.profile_dir)
        for _, home in homes:
            process_ownership.release(home)
