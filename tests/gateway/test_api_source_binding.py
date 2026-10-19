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


@pytest.mark.parametrize('source', ['hermes_browser', 'hermes_web', 'hermes_mobile'])
def test_private_api_binding_takes_first_party_client_rows(owner, source):
    """The Web, Mobile and Browser Extension clients store their own label on the sessions they create;
    those are API conversations, so they bind, restore and re-bind like an ``api_server`` row."""
    from gateway.session_api import bind_api_session, restore_api_session
    owner.db.create_session('first-party', source=source)
    ref = bind_api_session(owner, 'first-party')
    assert bind_api_session(owner, 'first-party') == ref  # met live
    owner.sessions.clear()
    assert restore_api_session(owner, 'first-party') == ref  # met after a restart
    assert owner.db.get_session('first-party')['source'] == source


@pytest.mark.parametrize('source', ['cli', 'tui', 'acp', 'desktop', 'dashboard', 'discord', 'slack', 'cron'])
def test_private_api_binding_refuses_every_other_surface(owner, source):
    from gateway.session_api import bind_api_session
    owner.db.create_session('foreign-native', source=source)
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        bind_api_session(owner, 'foreign-native')
    assert 'foreign-native' not in owner.sessions


def test_private_api_binding_adopts_matching_legacy_declared_key(owner):
    """Regression for #139835: a pre-binding API row stores the caller's declared key."""
    from gateway.session_api import bind_api_session, declared_api_session

    session_id = 'legacy-api-session'
    declared_key = 'webui:legacy-api-session'
    owner.db.create_session(session_id, source='api_server')
    owner.db.record_gateway_session_peer(session_id, source='api_server', session_key=declared_key)

    ref = bind_api_session(owner, session_id, declared_key=declared_key)

    row = owner.db.get_session(session_id)
    assert row['session_key'] == owner.sessions[ref.session_id].route
    assert declared_api_session(owner.db, declared_key) == session_id
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        bind_api_session(owner, 'other-api-session', declared_key=declared_key)


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
