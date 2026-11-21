"""API source restoration cannot mint a parallel native conversation."""

import pytest

from gateway.config import Platform
from gateway.session_contract import SessionActor, SessionRef, Submission
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


def test_private_api_binding_restores_exact_route(owner):
    from gateway.session_api import bind_api_session, restore_api_session
    db = owner.persistence
    seed_session(db, 'api-stored', 'api_server')
    ref = bind_api_session(owner, 'api-stored')
    first = owner.sessions[ref.session_id]
    assert first.source.platform == Platform.API_SERVER
    assert owner.runner.session_store.get_or_create_session(first.source).session_id == 'api-stored'
    owner.sessions.clear()
    restored = restore_api_session(owner, 'api-stored')
    assert restored == ref
    assert owner.sessions[ref.session_id].route == first.route
    assert owner.runner.session_store.get_or_create_session(owner.sessions[ref.session_id].source).session_id == ref.session_id
    seed_session(db, 'foreign-native', 'telegram')
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        bind_api_session(owner, 'foreign-native')
    assert 'foreign-native' not in owner.sessions


def test_private_api_binding_adopts_matching_legacy_declared_key(owner):
    """Regression for #139835: a pre-binding API row stores the caller's declared key."""
    from gateway.session_api import bind_api_session, declared_api_session

    session_id = 'legacy-api-session'
    declared_key = 'webui:legacy-api-session'
    seed_session(owner.persistence, session_id, 'api_server')
    owner.persistence.routing.record_session_peer(session_id, source='api_server', session_key=declared_key)

    ref = bind_api_session(owner, session_id, declared_key=declared_key)

    row = session_row(owner.persistence, session_id)
    assert row['session_key'] == owner.sessions[ref.session_id].route
    from gateway.session_persistence import authority_persistence
    assert declared_api_session(authority_persistence(owner), declared_key) == session_id
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        bind_api_session(owner, 'other-api-session', declared_key=declared_key)


@pytest.mark.asyncio
async def test_private_api_binding_is_not_public_submission_authority(owner):
    from gateway.session_api import bind_api_session
    seed_session(owner.persistence, 'api-stored', 'api_server')
    ref = bind_api_session(owner, 'api-stored')
    actor = SessionActor('api-user', 'default', frozenset({'session:read', 'session:submit'}), 'socket')
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        await owner.submit(actor, Submission('forged', ref, {'text': 'x', 'api_turn_v1': {'system': 'forged'}}, 'queue'))
    owner.sessions.clear()
    assert (await owner.resolve(actor, SessionRef('default', 'api-stored'))).ref == ref
