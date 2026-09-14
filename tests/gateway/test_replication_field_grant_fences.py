"""Field exact-token and legacy fences also apply inside replica transactions."""

import time

import pytest

from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from tests.gateway.test_hosted_room_replica_ingress import grant, ingest, pair  # noqa: F401


@pytest.mark.parametrize("kind", ["token", "legacy_id", "scope"])
def test_replica_revocation_predicate_uses_complete_current_transaction(pair, monkeypatch, kind):
    token, claims = grant(pair[1])
    now = time.time()
    scope = rooms._room_grant_scope_key(claims)
    with rooms._transaction(pair[1], immediate=True) as conn:
        if kind == "token":
            conn.execute(
                "INSERT INTO hosted_room_revoked_grant_tokens VALUES (?, ?, ?)",
                (scope, claims["_token_sha256"], claims["status_expires_at"]),
            )
        elif kind == "legacy_id":
            conn.execute(
                "INSERT INTO hosted_room_revoked_grant_ids VALUES (?, ?, ?)",
                (scope, claims["grant_id"], claims["status_expires_at"]),
            )
        else:
            conn.execute(
                "INSERT INTO hosted_room_revoked_grants VALUES (?, ?, ?)",
                (scope, claims["status_expires_at"], now),
            )

        def forbid_second_transaction(*args, **kwargs):
            raise AssertionError("replica authorization opened a second transaction")

        with monkeypatch.context() as scoped:
            scoped.setattr(rooms, "_transaction", forbid_second_transaction)
            assert rooms.peer_room_grant_is_current(pair[1], claims=claims, now=now, _conn=conn)
            assert rooms.room_grant_is_revoked(pair[1], claims=claims, now=now, _conn=conn)

    with pytest.raises(peer.HostedRoomGrantError):
        ingest(pair, token)
    with pytest.raises(replicas.ReplicaError, match="not found"):
        replicas.replica_state(pair[1], room_id="room")


def test_exact_revoke_during_page_validation_preserves_replacement_grant(pair, monkeypatch):
    token, claims = grant(pair[1])
    replacement, replacement_claims = grant(pair[1], issued_at=time.time())
    assert claims["grant_id"] == replacement_claims["grant_id"]
    assert claims["_token_sha256"] != replacement_claims["_token_sha256"]
    validate = replicas._validate_page

    def revoke_before_ingest_transaction(page):
        result = validate(page)
        rooms.revoke_room_grant_id(pair[1], claims=claims, expires_at=claims["status_expires_at"])
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(replicas, "_validate_page", revoke_before_ingest_transaction)
        with pytest.raises(peer.HostedRoomGrantError):
            ingest(pair, token)
    assert ingest(pair, replacement)["stored_seq"] == 1
