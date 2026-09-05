from __future__ import annotations

import sqlite3

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_peer import decode_room_grant, issue_room_grant


def claims(grant_id: str, issued_at: float) -> dict:
    fields = {
        "grant_id": grant_id,
        "room_id": "room-1",
        "home_install_id": "install:home",
        "authority_gateway_id": "install:home",
        "authority_epoch": 1,
        "member_id": "builder",
        "target_install_id": "install:peer",
        "target_profile": "builder",
        "issued_at": issued_at,
    }
    secret = b"exact-grant-storage-test-secret-only"
    token = issue_room_grant(secret, **fields, ttl_seconds=200)
    return decode_room_grant(secret, token, permission="status", now=issued_at + 1)


@pytest.mark.parametrize("winning_id", ["grant-winning", "grant-losing"])
def test_exact_revoke_preserves_a_concurrent_replacement(tmp_path, winning_id):
    db = tmp_path / "state.db"
    losing = claims("grant-losing", 100.0)
    winning = claims(winning_id, 101.0)
    other_scope = {**losing, "member_id": "reviewer"}

    hosted_rooms.revoke_room_grant_id(
        db,
        claims=losing,
        expires_at=300.0,
        now=110.0,
    )

    assert hosted_rooms.room_grant_is_revoked(db, claims=losing, now=120.0)
    assert not hosted_rooms.room_grant_is_revoked(db, claims=winning, now=120.0)
    assert not hosted_rooms.room_grant_is_revoked(db, claims=other_scope, now=120.0)


def test_scope_revoke_still_fences_all_older_grants(tmp_path):
    db = tmp_path / "state.db"
    first = claims("grant-first", 100.0)
    second = claims("grant-second", 101.0)

    hosted_rooms.revoke_room_grant_scope(
        db,
        claims=first,
        expires_at=300.0,
        now=110.0,
    )

    assert hosted_rooms.room_grant_is_revoked(db, claims=first, now=120.0)
    assert hosted_rooms.room_grant_is_revoked(db, claims=second, now=120.0)


def test_legacy_deny_survives_token_table_migration_without_allowing_claim_only_revoke(tmp_path):
    db = tmp_path / "state.db"
    denied = claims("legacy-grant", 100.0)
    assert not hosted_rooms.room_grant_is_revoked(db, claims=denied, now=110.0)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_revoked_grant_tokens")
        conn.execute("INSERT INTO hosted_room_revoked_grant_ids VALUES (?, ?, ?)",
                     (hosted_rooms._room_grant_scope_key(denied), denied["grant_id"], 300.0))
    assert hosted_rooms.room_grant_is_revoked(db, claims=denied, now=120.0)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM hosted_room_revoked_grant_tokens").fetchone()[0] == 0
    with pytest.raises(hosted_rooms.HostedRoomError, match="signed-token digest"):
        hosted_rooms.revoke_room_grant_id(
            db, claims={k: v for k, v in denied.items() if k != "_token_sha256"}, expires_at=300, now=120)
