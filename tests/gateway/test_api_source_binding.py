"""API source restoration cannot mint a parallel native conversation."""

import pytest

from gateway.config import Platform
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state_runtime import RuntimeStoreError


def test_private_api_binding_restores_exact_route(owner):
    from gateway.session_api import bind_api_session, restore_api_session
    db = owner.db
    db.create_session('api-stored', source='api_server')
    ref = bind_api_session(owner, 'api-stored')
    first = owner.sessions[ref.session_id]
    assert first.source.platform == Platform.API_SERVER
    assert owner.runner.session_store.get_or_create_session(first.source).session_id == 'api-stored'
    owner.sessions.clear()
    restored = restore_api_session(owner, 'api-stored')
    assert restored == ref
    assert owner.sessions[ref.session_id].route == first.route
    assert owner.runner.session_store.get_or_create_session(owner.sessions[ref.session_id].source).session_id == ref.session_id
    db.create_session('foreign-native', source='telegram')
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        bind_api_session(owner, 'foreign-native')
    assert 'foreign-native' not in owner.sessions


@pytest.mark.asyncio
async def test_private_api_binding_is_not_public_submission_authority(owner):
    from gateway.session_api import bind_api_session
    owner.db.create_session('api-stored', source='api_server')
    ref = bind_api_session(owner, 'api-stored')
    actor = Principal('api-user', 'default', frozenset({'session:read', 'session:submit'}), 'socket')
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await owner.submit(actor, Submission('forged', ref, {'text': 'x', 'api_turn_v1': {'system': 'forged'}}, 'queue'))
    owner.sessions.clear()
    assert (await owner.resolve(actor, SessionRef('default', 'api-stored'))).ref == ref
