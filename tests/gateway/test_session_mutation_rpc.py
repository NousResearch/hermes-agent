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


async def _rename(db, owner, session_id, title):
    return await owner.dispatch({'id': 9, 'method': 'session.mutate', 'params': {
        'session_id': session_id, 'request_id': f'rename-{session_id}-{title}',
        'expected_revision': session_row(db, session_id)['runtime_revision'],
        'operation': 'rename', 'payload': {'title': title}}})


@pytest.mark.asyncio
async def test_ws_rename_onto_a_held_title_is_a_readable_refusal(tmp_path):
    """#160463: a rename onto another session's title answered ``-32603 internal error`` (the store's
    plain ValueError crashed the dispatch). It is an ``invalid_title`` refusal whose message names
    the holder, and nothing is written."""
    from tests.fixtures.storage_seed import seed_messages
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 'holder', 'desktop', title='Marketplace Watch')
        seed_messages(db, 'holder', {'role': 'user', 'content': 'a real chat, not an empty ghost that yields its name'})
        seed_session(db, 's', 'desktop')
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        revision = session_row(db, 's')['runtime_revision']
        try:
            refused = await _rename(db, owner, 's', 'Marketplace Watch')
            error = refused['error']
            assert error['code'] == 4001, refused
            assert error['message'] == "Title 'Marketplace Watch' is already in use by session holder", refused
            assert error['data']['reason'] == 'invalid_title' and error['data']['field'] == 'title', refused
            row = session_row(db, 's')
            assert row['title'] is None and row['runtime_revision'] == revision
            assert session_row(db, 'holder')['title'] == 'Marketplace Watch'
            renamed = await _rename(db, owner, 's', 'Marketplace Watch 2')  # a free name still renames
            assert renamed['result']['title'] == 'Marketplace Watch 2', renamed
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_ws_rename_of_the_canonical_bot_chat_is_a_readable_refusal(tmp_path):
    """#160463: the canonical Bot Chat's name refusal took the same crash path."""
    from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 'bot', 'desktop', title=CANONICAL_BOT_CHAT_TITLE, hidden=True)
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['bot'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        revision = session_row(db, 'bot')['runtime_revision']
        try:
            refused = await _rename(db, owner, 'bot', 'Something else')
            error = refused['error']
            assert error['code'] == 4001, refused
            assert error['message'].startswith("This is the bot's canonical Bot Chat"), refused
            assert error['data']['reason'] == 'invalid_title', refused
            row = session_row(db, 'bot')
            assert row['title'] == CANONICAL_BOT_CHAT_TITLE and row['runtime_revision'] == revision
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_ws_rename_to_a_too_long_title_is_a_readable_refusal(tmp_path):
    """#160463 review: the store's other title refusal (past MAX_TITLE_LENGTH) took the same crash path."""
    from persistence.conversation.sessions import MAX_TITLE_LENGTH
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'desktop')
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['s'] = LiveSession(None, 'route')
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        revision = session_row(db, 's')['runtime_revision']
        try:
            refused = await _rename(db, owner, 's', 'x' * (MAX_TITLE_LENGTH + 1))
            error = refused['error']
            assert error['message'] == f'Title too long ({MAX_TITLE_LENGTH + 1} chars, max {MAX_TITLE_LENGTH})', refused
            assert error['data']['reason'] == 'invalid_title' and error['data']['field'] == 'title', refused
            row = session_row(db, 's')
            assert row['title'] is None and row['runtime_revision'] == revision
        finally:
            await owner.close()
