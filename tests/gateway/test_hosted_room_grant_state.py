"""Dual-store RoomLink grant state invariants."""

import sqlite3
from pathlib import Path

import pytest

from gateway import hosted_rooms
from gateway import hosted_room_grant_state as grant_state


def _claims() -> dict:
    return {
        "grant_id": "grant-test",
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "install-home",
        "authority_epoch": 1,
        "member_id": "member-reviewer",
        "target_install_id": "install-target",
        "target_profile": "reviewer",
        "issued_at": 1,
    }


def test_grant_state_paths_keep_shared_and_named_profile_db(tmp_path, monkeypatch):
    root = tmp_path / "install"
    profile_home = root / "profiles" / "reviewer"
    monkeypatch.setenv("HERMES_HOME", str(root))

    assert grant_state.grant_state_db_paths(profile_home) == (
        root / "state.db",
        profile_home / "state.db",
    )


def test_grant_state_paths_follow_the_active_profile_scope(tmp_path, monkeypatch):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    root = tmp_path / "install"
    profile_home = root / "profiles" / "reviewer"
    monkeypatch.setenv("HERMES_HOME", str(root))
    token = set_hermes_home_override(str(profile_home))
    try:
        assert grant_state.grant_state_db_paths() == (
            root / "state.db",
            profile_home / "state.db",
        )
    finally:
        reset_hermes_home_override(token)


def test_reservation_failure_compensates_the_shared_store(tmp_path, monkeypatch):
    shared = tmp_path / "shared.db"
    profile = tmp_path / "profile.db"
    claims = _claims()
    original = hosted_rooms.reserve_peer_room

    def fail_profile(db_path, **kwargs):
        if Path(db_path) == profile:
            raise RuntimeError("profile write failed")
        return original(db_path, **kwargs)

    monkeypatch.setattr(
        "gateway.hosted_rooms.reserve_peer_room",
        fail_profile,
    )

    with pytest.raises(RuntimeError, match="profile write failed"):
        grant_state.reserve_grant_state(
            (shared, profile),
            claims=claims,
            expires_at=10_000_000_000,
        )

    assert not hosted_rooms.peer_room_is_reserved(
        shared,
        room_id="room-1",
        target_profile="reviewer",
    )
    assert not hosted_rooms.room_grant_is_revoked(shared, claims=claims)


def test_failed_repeat_reservation_preserves_existing_grant(tmp_path, monkeypatch):
    shared = tmp_path / "shared.db"
    profile = tmp_path / "profile.db"
    claims = _claims()
    expires_at = 10_000_000_000
    for db_path in (shared, profile):
        hosted_rooms.reserve_peer_room(
            db_path,
            claims=claims,
            expires_at=expires_at - 100,
        )
    original = hosted_rooms.reserve_peer_room

    def fail_profile(db_path, **kwargs):
        if Path(db_path) == profile:
            raise RuntimeError("profile write failed")
        return original(db_path, **kwargs)

    monkeypatch.setattr(
        "gateway.hosted_rooms.reserve_peer_room",
        fail_profile,
    )

    with pytest.raises(RuntimeError, match="profile write failed"):
        grant_state.reserve_grant_state(
            (shared, profile),
            claims=claims,
            expires_at=expires_at,
        )

    assert hosted_rooms.peer_room_grant_is_current(shared, claims=claims)
    assert hosted_rooms.peer_room_grant_is_current(profile, claims=claims)
    assert not hosted_rooms.room_grant_is_revoked(shared, claims=claims)


def test_failed_authority_change_restores_the_previous_epoch(tmp_path, monkeypatch):
    shared = tmp_path / "shared.db"
    profile = tmp_path / "profile.db"
    old_claims = _claims()
    new_claims = {
        **old_claims,
        "authority_gateway_id": "install-new-home",
        "authority_epoch": 2,
    }
    expires_at = 10_000_000_000
    for db_path in (shared, profile):
        hosted_rooms.reserve_peer_room(
            db_path,
            claims=old_claims,
            expires_at=expires_at - 100,
        )
    original = hosted_rooms.reserve_peer_room

    def fail_profile(db_path, **kwargs):
        if Path(db_path) == profile:
            raise RuntimeError("profile write failed")
        return original(db_path, **kwargs)

    monkeypatch.setattr(
        "gateway.hosted_rooms.reserve_peer_room",
        fail_profile,
    )

    with pytest.raises(RuntimeError, match="profile write failed"):
        grant_state.reserve_grant_state(
            (shared, profile),
            claims=new_claims,
            expires_at=expires_at,
        )

    assert hosted_rooms.peer_room_grant_is_current(shared, claims=old_claims)
    assert hosted_rooms.peer_room_grant_is_current(profile, claims=old_claims)
    assert not hosted_rooms.peer_room_grant_is_current(shared, claims=new_claims)


def test_rollback_refuses_to_overwrite_a_later_reservation(tmp_path):
    shared = tmp_path / "shared.db"
    claims = _claims()
    snapshot = hosted_rooms.reserve_peer_room(
        shared,
        claims=claims,
        expires_at=10_000_000_000,
        now=100,
    )
    later = {
        **claims,
        "authority_gateway_id": "install-new-home",
        "authority_epoch": 2,
    }
    hosted_rooms.reserve_peer_room(
        shared,
        claims=later,
        expires_at=10_000_000_000,
        now=100,
    )

    with pytest.raises(
        hosted_rooms.AuthorityConflictError,
        match="changed during rollback",
    ):
        hosted_rooms.restore_peer_room_reservations(
            shared,
            claims=claims,
            snapshot=snapshot,
        )

    assert hosted_rooms.peer_room_grant_is_current(shared, claims=later, now=101)


def test_rollback_refuses_an_identical_later_reservation(tmp_path):
    shared = tmp_path / "shared.db"
    claims = _claims()
    snapshot = hosted_rooms.reserve_peer_room(
        shared,
        claims=claims,
        expires_at=10_000_000_000,
        now=100,
    )
    hosted_rooms.reserve_peer_room(
        shared,
        claims=claims,
        expires_at=10_000_000_000,
        now=100,
    )

    with pytest.raises(
        hosted_rooms.AuthorityConflictError,
        match="changed during rollback",
    ):
        hosted_rooms.restore_peer_room_reservations(
            shared,
            claims=claims,
            snapshot=snapshot,
        )

    assert hosted_rooms.peer_room_grant_is_current(shared, claims=claims, now=101)


def test_fresh_schema_accepts_and_stamps_an_older_writer(tmp_path):
    shared = tmp_path / "shared.db"
    claims = _claims()
    hosted_rooms.peer_room_is_reserved(
        shared,
        room_id="room-1",
        target_profile="reviewer",
    )
    with sqlite3.connect(shared) as conn:
        conn.execute(
            """INSERT INTO hosted_room_peer_reservations(
                   room_id, member_id, target_profile, authority_gateway_id,
                   authority_epoch, expires_at, revoked_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
            (
                claims["room_id"],
                claims["member_id"],
                claims["target_profile"],
                claims["authority_gateway_id"],
                claims["authority_epoch"],
                10_000_000_000,
                100,
                100,
            ),
        )
        mutation_id = conn.execute(
            "SELECT mutation_id FROM hosted_room_peer_reservations"
        ).fetchone()[0]

    assert mutation_id != "legacy"
    assert hosted_rooms.peer_room_grant_is_current(shared, claims=claims, now=101)


def test_rollback_refuses_an_identical_later_older_writer(tmp_path):
    shared = tmp_path / "shared.db"
    claims = _claims()
    snapshot = hosted_rooms.reserve_peer_room(
        shared,
        claims=claims,
        expires_at=10_000_000_000,
        now=100,
    )
    with sqlite3.connect(shared) as conn:
        conn.execute(
            """INSERT INTO hosted_room_peer_reservations(
                   room_id, member_id, target_profile, authority_gateway_id,
                   authority_epoch, expires_at, revoked_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
               ON CONFLICT(room_id, member_id, target_profile) DO UPDATE SET
                   authority_gateway_id=excluded.authority_gateway_id,
                   authority_epoch=excluded.authority_epoch,
                   expires_at=MAX(hosted_room_peer_reservations.expires_at,
                                  excluded.expires_at),
                   revoked_at=NULL,
                   updated_at=excluded.updated_at""",
            (
                claims["room_id"],
                claims["member_id"],
                claims["target_profile"],
                claims["authority_gateway_id"],
                claims["authority_epoch"],
                10_000_000_000,
                100,
                100,
            ),
        )

    with pytest.raises(
        hosted_rooms.AuthorityConflictError,
        match="changed during rollback",
    ):
        hosted_rooms.restore_peer_room_reservations(
            shared,
            claims=claims,
            snapshot=snapshot,
        )

    assert hosted_rooms.peer_room_grant_is_current(shared, claims=claims, now=101)


def test_revoke_attempts_profile_store_after_shared_failure(tmp_path, monkeypatch):
    shared = tmp_path / "shared.db"
    profile = tmp_path / "profile.db"
    claims = _claims()
    expires_at = 10_000_000_000
    for db_path in (shared, profile):
        hosted_rooms.reserve_peer_room(
            db_path,
            claims=claims,
            expires_at=expires_at,
        )
    original = hosted_rooms.revoke_room_grant_scope

    def fail_shared(db_path, **kwargs):
        if Path(db_path) == shared:
            raise RuntimeError("shared revoke failed")
        return original(db_path, **kwargs)

    monkeypatch.setattr(
        "gateway.hosted_rooms.revoke_room_grant_scope",
        fail_shared,
    )

    with pytest.raises(RuntimeError, match="shared revoke failed"):
        grant_state.revoke_grant_state(
            (shared, profile),
            claims=claims,
            expires_at=expires_at,
        )

    assert hosted_rooms.room_grant_is_revoked(profile, claims=claims)
