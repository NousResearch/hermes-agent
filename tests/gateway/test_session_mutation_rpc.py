"""Competing viewers must not overwrite a newer metadata revision."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import SessionRef
from gateway.session_controls import AuthorityConnection
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_mutation_rpc_fences_retries_and_competing_viewers(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='current')
        runner = SimpleNamespace(_draining=False)
        authority = SessionAuthority(runner, profile_id='owned', instance_id='current', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        viewer = AuthorityConnection(authority, object(), {'user_id': 'reader', 'capabilities': ['session:read']})
        request = {'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': 's', 'request_id': 'rename-1', 'expected_revision': session_row(db, 's')['runtime_revision'],
            'operation': 'rename', 'payload': {'title': 'Owned title'}}}
        try:
            denied = await viewer.dispatch(request)
            assert denied['error']['message'] == 'permission_denied', denied
            first = await owner.dispatch(request)
            assert first['result']['title'] == 'Owned title', first
            revision = session_row(db, 's')['runtime_revision']
            assert await owner.dispatch(request) == first
            assert session_row(db, 's')['runtime_revision'] == revision
            competing = {**request, 'params': {**request['params'], 'request_id': 'rename-2',
                                                'payload': {'title': 'Stale title'}}}
            conflict = await owner.dispatch(competing)
            assert conflict['error']['message'] == 'revision_conflict', conflict
            assert session_row(db, 's')['title'] == 'Owned title'
            archived = await owner.dispatch({**request, 'params': {**request['params'],
                'request_id': 'archive-1', 'expected_revision': revision,
                'operation': 'archive', 'payload': {'archived': True}}})
            assert archived['result']['archived'] is True, archived
            runner._draining = True
            stopped = await owner.dispatch(request)
            assert stopped['error']['message'] == 'runtime_draining', stopped
        finally:
            await owner.close()
            await viewer.close()


@pytest.mark.asyncio
async def test_pending_fanout_carries_the_revision_a_later_mutation_must_present(tmp_path):
    """A turn bumps runtime_revision; viewers only ever see session.info, so it must carry it."""
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='current')
        runner = SimpleNamespace(_draining=False)
        authority = SessionAuthority(runner, profile_id='owned', instance_id='current', epoch=epoch, persistence=storage_handle(tmp_path))
        live = authority.sessions['s'] = LiveSession(None, 'route')
        published = []
        live.event_stream.publish = lambda sid, payload, *, event_type='message.complete': published.append((event_type, payload))
        raw_holder(db)._execute_write(lambda conn: conn.execute("UPDATE sessions SET runtime_revision=runtime_revision+3 WHERE id='s'"))
        authority._publish_pending(SessionRef('owned', 's'))
        assert published and published[-1][0] == 'session.info'
        assert published[-1][1]['revision'] == session_row(db, 's')['runtime_revision']


@pytest.mark.asyncio
@pytest.mark.parametrize('guard', ['turn_lease', 'compression_lock'])
async def test_ws_mutation_refused_by_live_write_guard_is_session_busy(tmp_path, guard):
    """The WS path answers a live turn lease / compression lock like REST's 409: a retryable
    session_busy error frame with the row intact, never an exception that becomes an internal error."""
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['s'] = LiveSession(None, 'route')
        if guard == 'turn_lease':
            assert db.coordination.try_acquire_turn_lease(session_id='s', holder='foreign-turn')
        else:
            assert db.coordination.try_acquire_compression_lock(session_id='s', holder='foreign-compressor')
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        try:
            refused = await owner.dispatch({'id': 7, 'method': 'session.mutate', 'params': {
                'session_id': 's', 'request_id': 'delete-1',
                'expected_revision': session_row(db, 's')['runtime_revision'],
                'expected_generation': session_row(db, 's')['runtime_generation'],
                'operation': 'delete', 'payload': {}}})
            assert refused['error']['message'] == 'session_busy', refused
            assert db.sessions.get('s') is not None
        finally:
            await owner.close()
