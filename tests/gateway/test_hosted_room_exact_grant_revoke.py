from __future__ import annotations

from gateway import hosted_rooms


def claims(grant_id: str, issued_at: float) -> dict:
    return {
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


def test_exact_revoke_preserves_a_concurrent_replacement(tmp_path):
    db = tmp_path / "state.db"
    losing = claims("grant-losing", 100.0)
    winning = claims("grant-winning", 101.0)
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
