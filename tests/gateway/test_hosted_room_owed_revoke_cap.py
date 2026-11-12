"""At the owed-revoke ceiling a grant this home was handed is never dropped while it can still pass."""

import sqlite3
import time

import pytest

from gateway import hosted_rooms
from tests.gateway.test_hosted_room_owed_revoke_bounds import _grant, _owe, _rows
from tests.gateway.test_session_group_peers import _grant_error, call, invite
from tests.gateway.test_session_group_peers import gateway as gateway_fixture
from tests.gateway.test_session_group_peers import linked_peer as linked_peer_fixture

gateway, linked_peer = gateway_fixture, linked_peer_fixture


def _fill(db, *, expires_at):
    """Fill the owed-revoke table to its ceiling with other members' rows."""
    _owe(db, _grant(expires_at, "seed"), member="seed")
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO hosted_room_owed_revokes(room_id, member_id, grant_digest, target_url, grant, expires_at,"
            " next_attempt_at, created_at, held) VALUES ('full', ?, ?, 'http://127.0.0.1:9', 'g', ?, ?, 0, 0)",
            [(f"m{i}", f"d{i}", expires_at, 10**12) for i in range(hosted_rooms.MAX_OWED_REVOKES - 1)])


def test_at_the_ceiling_an_expired_row_makes_room(tmp_path):
    db, now = tmp_path / "state.db", time.time()
    _fill(db, expires_at=now - 1)
    _owe(db, _grant(now + 3600, "live"), member="new")
    assert [member for member, *_ in _rows(db)] == ["new"]


@pytest.mark.asyncio
async def test_a_grant_handed_in_at_the_ceiling_is_revoked_at_once(gateway, monkeypatch, linked_peer):
    """The room ended before its invitation was registered, and no room is left to record the
    revoke owed for that grant: it is revoked at its member's gateway now, never dropped live."""
    _server, url, catalog, room = linked_peer
    grant = (await invite(gateway, room))['grant']
    assert 'tombstone' in await call(gateway.owner, 'groups.disband', room_id='linked')
    _fill(gateway.db.db_path, expires_at=time.time() + 3600)
    assert await _grant_error(url, grant) is None
    refused = await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                         target_url=url, target_profile='default', grant=grant, catalog=catalog)
    assert not isinstance(refused, dict), refused
    assert await _grant_error(url, grant) == 'room_reauthorization_required'
