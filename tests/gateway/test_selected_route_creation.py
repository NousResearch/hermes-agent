"""Narrow absent-to-present canonical creation under an operation hold."""
import pytest
from tests.gateway.test_selected_route import selection

@pytest.mark.parametrize('foreign', [False, True])
def test_held_canonical_creation_is_only_own_transition(selection, foreign):
    from types import SimpleNamespace
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.session_api import hosted_session_id, bind_api_session
    from gateway import session_selected_route as sr
    t = selection
    t.boundary.fail = True
    from gateway.hosted_room_peer import HostedMemberDispatch
    from tests.gateway.test_canonical_peer_text_admission import dispatch
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    from gateway import hosted_rooms
    catalog = _local_room_catalog(t.adapter, 'default', hosted_rooms.local_authority_gateway_id())[1]
    d = dispatch({'catalog': catalog})
    sid = hosted_session_id(d)
    source = SessionSource(platform=Platform.API_SERVER, chat_id=sid, user_id='api', chat_type='dm')
    scope = sr.selection_scope(t.adapter, source=source,
        session_key=t.runner.session_store._generate_session_key(source), session_id=sid,
        request_identity=object(), purpose='admit')
    b = sr.prepare_selected_route(scope)
    if foreign:
        bind_api_session(t.authority, sid, hosted_dispatch=d.as_mapping())
        with pytest.raises(sr.SelectedRouteUnavailable):
            with sr.hold_selected_route(scope, b): pass
    else:
        with sr.hold_selected_route(scope, b):
            # The trusted binder must understand the held store guard. Never
            # reacquire its non-reentrant store lock or wildcard row revisions.
            bind_api_session(t.authority, sid, hosted_dispatch=d.as_mapping())
            assert sr.peek_selected_route(scope, b).supports_prepared_files
            t.db._execute_write(lambda c: c.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (sid,)))
            assert not sr.peek_selected_route(scope, b).supports_prepared_files
    assert b._material is None


def test_held_override_does_not_skip_unowned_session_store_lock(selection):
    from gateway import session_selected_route as sr
    from tests.gateway.test_selected_route import scoped
    t = selection
    t.runner._session_state(t.key).conversation.model_override = {'model': 'fixture', 'provider': 'anthropic'}
    scope = scoped(t)
    b = sr.prepare_selected_route(scope)
    with sr.hold_selected_route(scope, b):
        with sr.session_store_guard(t.runner):
            acquired = t.runner.session_store._lock.acquire(blocking=False)
            if acquired:
                t.runner.session_store._lock.release()
            assert acquired is False, 'only bypass a store lock actually owned by this hold'
