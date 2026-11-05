"""Owed revokes never decide coverage from a grant's unsigned claims: the home cannot check a member
gateway's signature, so bad-signature grants filling one member's owed-revoke slots never make a real
grant handed in after them go unrevoked."""

import time

import pytest

from persistence.gateway import hosted_rooms
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.gateway.test_hosted_room_owed_revoke_bounds import _grant, _owe, _rows
from tests.gateway.test_session_group_peers import _grant_error, call, invite
from tests.gateway.test_session_group_peers import gateway as gateway_fixture
from tests.gateway.test_session_group_peers import linked_peer as linked_peer_fixture

gateway, linked_peer = gateway_fixture, linked_peer_fixture


def test_a_full_member_refuses_instead_of_dropping_a_grant(tmp_path):
    db, now = storage_handle(tmp_path), time.time()
    for index in range(hosted_rooms.MAX_OWED_REVOKES_PER_MEMBER):
        _owe(db, _grant(now + 86400 + index, f"forged{index}"))  # claims to outlive the real one
    before = _rows(db)
    with pytest.raises(hosted_rooms.OwedRevokesFullError):
        _owe(db, _grant(now + 3600, "real"))
    with pytest.raises(hosted_rooms.OwedRevokesFullError):
        _owe(db, _grant(now + 10**6, "longer"))  # nor is an owed row evicted for an unsigned claim
    assert _rows(db) == before


@pytest.mark.asyncio
async def test_forged_grants_in_every_slot_do_not_keep_a_real_grant_alive(gateway, linked_peer):
    """The room ended before its invitation was registered, and its member's owed-revoke slots hold
    bad-signature grants claiming longer lifetimes: the real grant is revoked at its member now."""
    _server, url, catalog, room = linked_peer
    grant = (await invite(gateway, room))['grant']
    assert 'tombstone' in await call(gateway.owner, 'groups.disband', room_id='linked')
    now = time.time()
    for index in range(hosted_rooms.MAX_OWED_REVOKES_PER_MEMBER):
        _owe(gateway.db, _grant(now + 86400 + index, f"forged{index}"), room='linked', member='reviewer')
    refused = await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                         target_url=url, target_profile='default', grant=grant, catalog=catalog)
    assert not isinstance(refused, dict), refused
    assert await _grant_error(url, grant) == 'room_reauthorization_required'
