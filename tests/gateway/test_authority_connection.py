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
