"""Owed RoomLink grant revokes stay bounded however many grants a caller hands in."""

import base64
import json
import time

from gateway import hosted_rooms


def _grant(expires_at: float, nonce: str) -> str:
    """A grant-shaped token whose unsigned payload claims ``expires_at`` (the signature is never read)."""
    claims = json.dumps({"expires_at": expires_at, "status_expires_at": expires_at, "nonce": nonce}).encode()
    return f"{base64.urlsafe_b64encode(claims).decode().rstrip('=')}.c2ln"


def _rows(db):
    import sqlite3
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT member_id, expires_at, attempts FROM hosted_room_owed_revokes "
                            "ORDER BY member_id, expires_at").fetchall()


def _owe(db, grant, *, member="peer", room="ended", expires_at=None):
    from gateway.hosted_room_peer import room_grant_hard_expiry
    hosted_rooms.owe_room_grant_revoke(
        db, room_id=room, member_id=member, target_url="http://127.0.0.1:9", grant=grant,
        expires_at=room_grant_hard_expiry(grant) if expires_at is None else expires_at)


def test_an_owed_revoke_lives_no_longer_than_the_longest_signed_horizon(tmp_path):
    db, now = tmp_path / "state.db", time.time()
    _owe(db, _grant(now + 10**9, "forever"))  # an unsigned payload claiming 30 years
    (_member, expires_at, _attempts), = _rows(db)
    assert expires_at <= now + hosted_rooms.MAX_OWED_REVOKE_SECONDS + 60


def test_repeated_and_many_grants_for_one_member_keep_a_few_rows(tmp_path):
    db, now = tmp_path / "state.db", time.time()
    same = _grant(now + 3600, "same")
    for _ in range(5):
        _owe(db, same)
    assert len(_rows(db)) == 1
    for index in range(50):
        _owe(db, _grant(now + 3600 + index, f"g{index}"))
    rows = _rows(db)
    assert len(rows) == hosted_rooms.MAX_OWED_REVOKES_PER_MEMBER
    # The longest-lived grants are kept: revoking the scope at the member covers the older ones.
    assert rows[-1][1] == now + 3600 + 49


def test_an_owed_revoke_that_never_lands_is_dropped_after_its_attempts(tmp_path):
    db = tmp_path / "state.db"
    _owe(db, _grant(time.time() + 3600, "unreachable"))
    for _ in range(hosted_rooms.MAX_OWED_REVOKE_ATTEMPTS):
        owed, = hosted_rooms.due_room_grant_revokes(db, limit=5, now=time.time() + 10)
        hosted_rooms.settle_room_grant_revoke(
            db, room_id=owed["room_id"], member_id=owed["member_id"], grant_digest=owed["grant_digest"],
            retry_at=time.time())
    assert _rows(db) == []
