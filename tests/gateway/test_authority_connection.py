"""Connection membership against the real authority and temporary SQLite."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.mark.asyncio
async def test_repeated_resume_and_close_preserve_only_live_memberships(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test',
                                     instance_id='test', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        first = AuthorityConnection(authority, object(), {'user_id': 'human'})
        peer = AuthorityConnection(authority, object(), {'user_id': 'human'})
        request = {'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}}
        for connection in (first, peer, first, first):
            result = (await connection.dispatch(request))['result']
            assert result['info'] == {'desktop_protocol': CANONICAL_GATEWAY_PROTOCOL}
        members = authority.sessions['s'].subscribers
        assert len(members) == 2
        await first.close()
        assert list(members.values()) == [peer.actor]
        assert first.actor.transport_id not in authority.events
        await first.close()
        assert list(members.values()) == [peer.actor]
        await peer.close()
        assert not members and not authority.events
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('deleted_first', [True, False])
async def test_close_after_a_subscribed_session_was_deleted_releases_every_membership(tmp_path, deleted_first):
    """Deletion evicts the live session before the viewer disconnects; teardown must still
    release the other subscription and the transport instead of raising not_found."""
    import tui_gateway.ws  # noqa: F401 — the fanout drain thread imports it lazily during teardown
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        for sid in ('doomed', 'kept'):
            db.create_session(sid, source='test')
        epoch = begin_runtime_epoch(db, instance_id='test')
        store = SimpleNamespace(_routing_db=db, retire_runtime_sessions=lambda ids: None)
        runner = SimpleNamespace(_draining=False, session_store=store, _evict_cached_agent=lambda route: None)
        authority = SessionAuthority(runner, profile_id='test', instance_id='test', db=db, epoch=epoch)
        for sid in ('doomed', 'kept'):
            authority.sessions[sid] = LiveSession(None, sid + '-route')
        viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})
        order = ['doomed', 'kept'] if deleted_first else ['kept', 'doomed']
        for sid in order:
            reply = await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
            assert 'result' in reply, reply
        kept_members = authority.sessions['kept'].subscribers
        reply = await viewer.dispatch({'id': 2, 'method': 'session.mutate', 'params': {
            'session_id': 'doomed', 'request_id': 'delete', 'expected_revision': 0,
            'expected_generation': 0, 'operation': 'delete', 'payload': {}}})
        assert reply['result']['deleted_ids'] == ['doomed'], reply
        assert 'doomed' not in authority.sessions
        await viewer.close()
        assert not kept_members
        assert not viewer.subscriptions
        assert viewer.actor.transport_id not in authority.events
    finally:
        db.close()


@pytest.mark.asyncio
async def test_detach_releases_only_the_paired_subscription_without_stopping_execution(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    running = None
    viewer = observer = None
    try:
        for sid in ('a', 'b'):
            db.create_session(sid, source='test')
        epoch = begin_runtime_epoch(db, instance_id='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test',
                                     instance_id='test', db=db, epoch=epoch)
        for sid in ('a', 'b'):
            authority.sessions[sid] = LiveSession(None, sid + '-route')
        running = asyncio.create_task(asyncio.Event().wait())
        authority.sessions['a'].task = running
        viewer = AuthorityConnection(authority, object(), {
            'user_id': 'human', 'capabilities': ['session:read']})
        observer = AuthorityConnection(authority, object(), {
            'user_id': 'peer', 'capabilities': ['session:read']})

        a = (await viewer.dispatch({'id': 1, 'method': 'session.resume',
                                    'params': {'session_id': 'a'}}))['result']
        peer_a = (await observer.dispatch({'id': 1, 'method': 'session.resume',
                                           'params': {'session_id': 'a'}}))['result']
        b = (await viewer.dispatch({'id': 2, 'method': 'session.resume',
                                    'params': {'session_id': 'b'}}))['result']
        request = {'id': 3, 'method': 'session.detach', 'params': {
            'session_id': 'a', 'subscription_id': a['subscription_id']}}

        detached = await viewer.dispatch(request)

        assert detached['result'] == {
            'session_id': 'a', 'subscription_id': a['subscription_id'], 'detached': True}
        assert authority.sessions['a'].task is running and not running.done()
        assert list(authority.sessions['a'].subscribers) == [peer_a['subscription_id']]
        assert list(authority.sessions['b'].subscribers) == [b['subscription_id']]
        assert viewer.subscriptions == {'b': b['subscription_id']}

        duplicate = await viewer.dispatch(request)
        stale_pair = await viewer.dispatch({'id': 4, 'method': 'session.detach', 'params': {
            'session_id': 'b', 'subscription_id': a['subscription_id']}})
        caller_mismatch = await viewer.dispatch({'id': 5, 'method': 'session.detach', 'params': {
            'session_id': 'a', 'subscription_id': peer_a['subscription_id']}})
        assert duplicate['result'] == {
            'session_id': 'a', 'subscription_id': a['subscription_id'], 'detached': False}
        assert stale_pair['result'] == {
            'session_id': 'b', 'subscription_id': a['subscription_id'], 'detached': False}
        assert caller_mismatch['result'] == {
            'session_id': 'a', 'subscription_id': peer_a['subscription_id'], 'detached': False}
        assert list(authority.sessions['a'].subscribers) == [peer_a['subscription_id']]
        assert list(authority.sessions['b'].subscribers) == [b['subscription_id']]
    finally:
        if viewer is not None:
            await viewer.close()
        if observer is not None:
            await observer.close()
        if running is not None:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_detach_rejects_malformed_params_through_dispatcher(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    viewer = None
    try:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test',
                                     instance_id='test', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})

        malformed = [
            ['unexpected'],
            'unexpected',
            1,
            {},
            {'session_id': 's'},
            {'session_id': '', 'subscription_id': 'token'},
            {'session_id': 's', 'subscription_id': ''},
            {'session_id': 1, 'subscription_id': 'token'},
            {'session_id': 's', 'subscription_id': 1},
            {'session_id': 's', 'subscription_id': 'token', 'extra': True},
        ]
        for index, params in enumerate(malformed):
            reply = await viewer.dispatch({
                'id': index, 'method': 'session.detach', 'params': params})
            assert reply['error']['data']['reason'] == 'invalid_params', reply
    finally:
        if viewer is not None:
            await viewer.close()
        db.close()


@pytest.mark.asyncio
async def test_stale_same_session_token_cannot_detach_newer_subscription(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    viewer = None
    try:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test',
                                     instance_id='test', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})

        first = (await viewer.dispatch({'id': 1, 'method': 'session.resume',
                                        'params': {'session_id': 's'}}))['result']
        # Model an already-retired attachment before the same connection resumes S.
        await authority.detach(viewer.actor, first['subscription_id'])
        second = (await viewer.dispatch({'id': 2, 'method': 'session.resume',
                                         'params': {'session_id': 's'}}))['result']
        assert second['subscription_id'] != first['subscription_id']

        stale = await viewer.dispatch({'id': 3, 'method': 'session.detach', 'params': {
            'session_id': 's', 'subscription_id': first['subscription_id']}})
        duplicate_stale = await viewer.dispatch({'id': 4, 'method': 'session.detach', 'params': {
            'session_id': 's', 'subscription_id': first['subscription_id']}})
        assert stale['result']['detached'] is False
        assert duplicate_stale['result']['detached'] is False
        assert viewer.subscriptions == {'s': second['subscription_id']}
        assert list(authority.sessions['s'].subscribers) == [second['subscription_id']]
    finally:
        if viewer is not None:
            await viewer.close()
        db.close()
