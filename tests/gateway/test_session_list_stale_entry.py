"""``session.list`` skips a live entry whose durable row is gone instead of dying on it.

``hermes sessions delete`` removes the row but the authority's live registry keeps the
routing entry, and ``authorize`` only refuses an id absent from ``self.sessions`` entirely.
``_handle`` then indexed the missing row and raised ``TypeError``, which is not a
``RuntimeStoreError``, so the lister's ``except`` could not skip it: every ``session.list``
failed until the gateway restarted, taking the whole picker down with one stale entry.

The same entry poisons the Desktop feeder: ``session.active_list`` reads the same live
registry and its ``except`` also only tolerates ``RuntimeStoreError``. An out-of-band row
deletion (a maintenance script against ``state.db``, not the gateway's ``sessions delete``)
leaves the same shape behind, so both feeders are pinned here.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_contract import SessionRef

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='live', instance_id='owner')
    for sid in ('chat', 'kept'):
        store._db.create_session(sid, source='telegram')
        authority.sessions[sid] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id=sid), sid)
    connection = AuthorityConnection(authority, object(), {'user_id': 'viewer'})

    async def listing():
        reply = await connection.dispatch({'id': 1, 'method': 'session.list', 'params': {'limit': 200}})
        assert 'result' in reply, reply
        return {row['session_id']: row for row in reply['result']['sessions']}

    async def live_listing():
        reply = await connection.dispatch({'id': 1, 'method': 'session.active_list', 'params': {}})
        assert 'result' in reply, reply
        return {row['id'] for row in reply['result']['sessions']}

    async def resolve(session_id):
        return await authority.resolve(connection.actor, SessionRef('live', session_id))

    yield SimpleNamespace(authority=authority, listing=listing, live_listing=live_listing,
                          resolve=resolve, db=store._db)


@pytest.mark.asyncio
async def test_a_live_entry_whose_row_was_deleted_is_skipped_not_fatal(owner):
    assert set(await owner.listing()) == {'chat', 'kept'}

    owner.db.delete_session('chat')

    assert set(await owner.listing()) == {'kept'}


@pytest.mark.asyncio
async def test_the_desktop_live_feed_skips_the_same_entry(owner):
    assert await owner.live_listing() == {'chat', 'kept'}

    owner.db.delete_session('chat')

    assert await owner.live_listing() == {'kept'}


@pytest.mark.asyncio
async def test_a_stale_live_entry_resolves_as_not_found(owner):
    from hermes_state_runtime import RuntimeStoreError

    owner.db.delete_session('chat')

    with pytest.raises(RuntimeStoreError) as excinfo:
        await owner.resolve('chat')
    assert excinfo.value.reason == 'not_found'
