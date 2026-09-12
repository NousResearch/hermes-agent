"""Connection membership against the real authority and temporary SQLite."""
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
