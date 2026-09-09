"""Storage and credential tests for reciprocal hosted-room control."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway import hosted_room_controls as controls
from gateway import hosted_rooms


HOME = "install:gateway-a"


def _create_room(db, room_id="room-1", *, authority=HOME, epoch=1):
    room = hosted_rooms.create_room(
        db,
        room_id=room_id,
        name="Release room",
        members=[
            {
                "member_id": "member-1",
                "profile": "reviewer",
                "handle": "reviewer",
            }
        ],
        authority_gateway_id=authority,
        now=10,
    )
    assert room["authority_epoch"] == epoch
    return room


def _issue(db, *, room_id="room-1", member_id="member-1", now=20):
    return controls.issue_home_control_token(
        db,
        room_id=room_id,
        member_id=member_id,
        authority_gateway_id=HOME,
        authority_epoch=1,
        expires_at=now + 600,
        now=now,
    )


def _save(db, token, *, room_id="room-1", member_id="member-1", **overrides):
    values = {
        "target_profile": "reviewer",
        "home_url": "https://home.example.test",
        "authority_gateway_id": HOME,
        "authority_epoch": 1,
        "room_name": "Planning",
        "member_count": 2,
        "expires_at": 620,
        "now": 20,
        **overrides,
    }
    return controls.save_peer_control_link(
        db,
        room_id=room_id,
        member_id=member_id,
        control_token=token,
        **values,
    )


def test_home_stores_only_sha256_and_never_exposes_token(tmp_path):
    db = tmp_path / "state.db"
    _create_room(db)
    issued = _issue(db)

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT token_hash, status FROM hosted_room_control_tokens"
        ).fetchone()
    assert isinstance(row[0], bytes)
    assert len(row[0]) == 32
    assert row[0] != issued.control_token.encode()
    assert row[1] == "active"
    assert issued.control_token not in repr(issued)
    assert issued.control_token not in repr(issued.as_status())
    assert issued.control_token.encode() not in db.read_bytes()


def test_home_invitation_replay_recovers_the_same_opaque_token(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _create_room(db)
    monkeypatch.setattr(
        controls,
        "gateway_room_grant_secret",
        lambda: b"s" * 32,
    )
    common = {
        "room_id": "room-1",
        "member_id": "member-1",
        "authority_gateway_id": HOME,
        "authority_epoch": 1,
        "expires_at": controls.ROOM_LIFETIME_EXPIRES_AT,
        "request_id": "desktop-room-1-member-1-v1",
    }

    first = controls.issue_home_control_token(db, now=20, **common)
    replay = controls.issue_home_control_token(db, now=30, **common)

    assert replay.control_token == first.control_token
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM hosted_room_control_tokens").fetchone()[
                0
            ]
            == 1
        )
    with pytest.raises(controls.HostedRoomControlConflictError):
        controls.issue_home_control_token(
            db,
            now=40,
            **{**common, "request_id": "different-request"},
        )


def test_previous_control_schema_adds_request_id_before_use(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _create_room(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE hosted_room_control_tokens (
                room_id TEXT NOT NULL,
                member_id TEXT NOT NULL,
                authority_gateway_id TEXT NOT NULL,
                authority_epoch INTEGER NOT NULL,
                token_hash BLOB NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                revoked_at REAL,
                PRIMARY KEY (
                    room_id, member_id, authority_gateway_id, authority_epoch
                )
            )"""
        )
    monkeypatch.setattr(
        controls,
        "gateway_room_grant_secret",
        lambda: b"s" * 32,
    )

    controls.issue_home_control_token(
        db,
        room_id="room-1",
        member_id="member-1",
        authority_gateway_id=HOME,
        authority_epoch=1,
        expires_at=controls.ROOM_LIFETIME_EXPIRES_AT,
        request_id="desktop-room-1-member-1-v1",
        now=20,
    )

    with sqlite3.connect(db) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(hosted_room_control_tokens)")
        }
        request_id = conn.execute(
            "SELECT request_id FROM hosted_room_control_tokens"
        ).fetchone()[0]
    assert "request_id" in columns
    assert request_id == "desktop-room-1-member-1-v1"


def test_verify_is_exactly_room_member_authority_epoch_and_active_room_scoped(tmp_path):
    db = tmp_path / "state.db"
    _create_room(db)
    _create_room(db, "room-2")
    issued = _issue(db)
    base = {
        "room_id": "room-1",
        "member_id": "member-1",
        "authority_gateway_id": HOME,
        "authority_epoch": 1,
        "control_token": issued.control_token,
        "now": 30,
    }

    assert controls.verify_home_control_token(db, **base)
    for changed in (
        {"room_id": "room-2"},
        {"member_id": "member-2"},
        {"authority_gateway_id": "install:gateway-b"},
        {"authority_epoch": 2},
        {"control_token": "A" * 43},
    ):
        assert not controls.verify_home_control_token(db, **{**base, **changed})

    hosted_rooms.disband_room(
        db,
        room_id="room-1",
        expected_gateway_id=HOME,
        expected_epoch=1,
        now=40,
    )
    assert not controls.verify_home_control_token(db, **{**base, "now": 50})


def test_verify_always_uses_constant_time_digest_comparison(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _create_room(db)
    issued = _issue(db)
    seen = []
    original = hmac.compare_digest

    def record(left, right):
        seen.append((left, right))
        return original(left, right)

    monkeypatch.setattr(controls.hmac, "compare_digest", record)
    assert (
        controls.verify_home_control_token(
            db,
            room_id="room-1",
            member_id="missing-member",
            authority_gateway_id=HOME,
            authority_epoch=1,
            control_token=issued.control_token,
            now=30,
        )
        is False
    )
    assert len(seen) == 1
    assert all(isinstance(value, bytes) and len(value) == 32 for value in seen[0])


def test_home_expiry_and_revocation_fail_closed_and_revoke_is_idempotent(tmp_path):
    db = tmp_path / "state.db"
    _create_room(db)
    issued = _issue(db)
    kwargs = {
        "room_id": "room-1",
        "member_id": "member-1",
        "authority_gateway_id": HOME,
        "authority_epoch": 1,
        "control_token": issued.control_token,
    }

    assert not controls.verify_home_control_token(db, **kwargs, now=620)
    assert (
        controls.revoke_home_control_tokens(
            db, room_id="room-1", member_id="member-1", now=100
        )
        == 1
    )
    assert (
        controls.revoke_home_control_tokens(
            db, room_id="room-1", member_id="member-1", now=101
        )
        == 0
    )
    assert not controls.verify_home_control_token(db, **kwargs, now=110)


def test_malformed_home_row_fails_closed_without_skipping_digest_comparison(
    tmp_path, monkeypatch
):
    db = tmp_path / "state.db"
    _create_room(db)
    issued = _issue(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_control_tokens SET expires_at='not-a-time'")
        conn.commit()
    compared = 0
    original = hmac.compare_digest

    def record(left, right):
        nonlocal compared
        compared += 1
        return original(left, right)

    monkeypatch.setattr(controls.hmac, "compare_digest", record)
    assert not controls.verify_home_control_token(
        db,
        room_id="room-1",
        member_id="member-1",
        authority_gateway_id=HOME,
        authority_epoch=1,
        control_token=issued.control_token,
        now=30,
    )
    assert compared == 1


def test_revoked_or_expired_scope_can_issue_a_new_opaque_token(tmp_path):
    db = tmp_path / "state.db"
    _create_room(db)
    first = _issue(db)
    controls.revoke_home_control_tokens(db, room_id="room-1", now=30)
    second = _issue(db, now=40)

    assert first.control_token != second.control_token
    common = {
        "room_id": "room-1",
        "member_id": "member-1",
        "authority_gateway_id": HOME,
        "authority_epoch": 1,
        "now": 50,
    }
    assert not controls.verify_home_control_token(
        db, control_token=first.control_token, **common
    )
    assert controls.verify_home_control_token(
        db, control_token=second.control_token, **common
    )


def test_peer_link_save_load_restart_and_private_repr(tmp_path):
    home_db = tmp_path / "home.db"
    peer_db = tmp_path / "peer.db"
    _create_room(home_db)
    issued = _issue(home_db)

    created = _save(peer_db, issued.control_token)
    repeated = _save(peer_db, issued.control_token)
    loaded = controls.load_peer_control_links(peer_db, now=30)

    assert created.idempotent is False
    assert repeated.idempotent is True
    assert loaded.links == (created.link,)
    assert loaded.quarantined == 0
    assert loaded.truncated is False
    assert issued.control_token not in repr(created)
    assert issued.control_token not in repr(created.link.as_status())
    assert created.link.transport_security == "tls"


@pytest.mark.parametrize(
    "change",
    [
        {"home_url": "https://other.example.test"},
        {"authority_gateway_id": "install:gateway-b"},
        {"authority_epoch": 2},
    ],
)
def test_peer_link_conflicts_on_changed_authority_or_url(tmp_path, change):
    home_db = tmp_path / "home.db"
    peer_db = tmp_path / "peer.db"
    _create_room(home_db)
    issued = _issue(home_db)
    _save(peer_db, issued.control_token)

    with pytest.raises(
        controls.HostedRoomControlConflictError, match="stored authority"
    ):
        _save(peer_db, issued.control_token, **change)


def test_peer_link_conflicts_on_changed_token_without_leaking_it(tmp_path):
    home_db = tmp_path / "home.db"
    peer_db = tmp_path / "peer.db"
    _create_room(home_db)
    first = _issue(home_db)
    controls.revoke_home_control_tokens(home_db, room_id="room-1", now=30)
    second = _issue(home_db, now=40)
    _save(peer_db, first.control_token)

    with pytest.raises(controls.HostedRoomControlConflictError) as error:
        _save(peer_db, second.control_token)
    assert first.control_token not in str(error.value)
    assert second.control_token not in str(error.value)


def test_peer_link_requires_https_or_loopback(tmp_path):
    strong = secrets.token_urlsafe(controls.TOKEN_BYTES)

    with pytest.raises(controls.HostedRoomControlError, match="endpoint"):
        _save(tmp_path / "state.db", strong, home_url="http://peer.example.test")
    saved = _save(tmp_path / "state.db", strong, home_url="http://127.0.0.1:8080")
    assert saved.link.transport_security == "loopback"


def test_peer_control_link_requires_the_exact_live_roomlink_reservation(tmp_path):
    db = tmp_path / "state.db"
    claims = {
        "room_id": "room-1",
        "member_id": "member-1",
        "target_profile": "reviewer",
        "authority_gateway_id": HOME,
        "authority_epoch": 1,
    }
    hosted_rooms.reserve_peer_room(
        db,
        claims=claims,
        expires_at=100,
        now=20,
    )
    assert controls.peer_reservation_matches(
        db,
        **claims,
        now=30,
    )
    assert not controls.peer_reservation_matches(
        db,
        **{**claims, "member_id": "member-2"},
        now=30,
    )
    hosted_rooms.reserve_peer_room(
        db,
        claims={**claims, "target_profile": "writer"},
        expires_at=100,
        now=20,
    )
    assert not controls.peer_reservation_matches(db, **claims, now=30)
    assert not controls.peer_reservation_matches(
        db,
        **claims,
        now=100,
    )


def test_peer_target_profile_persists_and_migrates_only_exact_unrevoked_binding(
    tmp_path,
):
    token = secrets.token_urlsafe(controls.TOKEN_BYTES)
    fresh = tmp_path / "fresh.db"
    assert _save(fresh, token).link.target_profile == "reviewer"
    with pytest.raises(controls.HostedRoomControlConflictError):
        _save(
            fresh,
            token,
            target_profile="writer",
            allow_rotation=True,
            now=30,
        )

    cases = (
        ("expired", (("reviewer", 20),), False, "reviewer"),
        ("revoked", (("reviewer", 20),), True, None),
        (
            "ambiguous",
            (("reviewer", 20), ("writer", 20)),
            False,
            None,
        ),
        ("reassigned", (("writer", 30),), False, None),
        ("missing", (), False, None),
    )
    for name, reservations, revoked, expected in cases:
        db = tmp_path / f"{name}.db"
        _save(db, token)
        with sqlite3.connect(db) as conn:
            conn.execute(
                "ALTER TABLE hosted_room_peer_controls DROP COLUMN target_profile"
            )
        for profile, created_at in reservations:
            hosted_rooms.reserve_peer_room(
                db,
                claims={
                    "room_id": "room-1",
                    "member_id": "member-1",
                    "target_profile": profile,
                    "authority_gateway_id": HOME,
                    "authority_epoch": 1,
                },
                expires_at=100,
                now=created_at,
            )
        if revoked:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE hosted_room_peer_reservations SET revoked_at=30"
                )

        loaded = controls.load_peer_control_links(db, now=200).links
        assert len(loaded) == 1
        assert loaded[0].target_profile == expected


def test_peer_expiry_and_revocation_are_durable_and_idempotent(tmp_path):
    db = tmp_path / "state.db"
    token = secrets.token_urlsafe(controls.TOKEN_BYTES)
    _save(db, token, expires_at=40)

    expired = controls.load_peer_control_links(db, now=40, include_inactive=True)
    assert expired.links[0].status == "expired"
    assert controls.load_peer_control_links(db, now=41).links == ()
    assert (
        controls.revoke_peer_control_links(
            db, room_id="room-1", member_id="member-1", now=50
        )
        == 1
    )
    assert (
        controls.revoke_peer_control_links(
            db, room_id="room-1", member_id="member-1", now=51
        )
        == 0
    )
    reloaded = controls.load_peer_control_links(db, now=60, include_inactive=True)
    assert reloaded.links[0].status == "revoked"


def test_malformed_peer_row_is_quarantined_without_secret_in_diagnostics(tmp_path):
    db = tmp_path / "state.db"
    token = secrets.token_urlsafe(controls.TOKEN_BYTES)
    _save(db, token)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_peer_controls
                  SET home_url='http://public.example.test'
                WHERE room_id='room-1'"""
        )
        conn.commit()

    loaded = controls.load_peer_control_links(db, now=30)
    assert loaded.links == ()
    assert loaded.quarantined == 1
    assert token not in repr(loaded)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """SELECT status, quarantine_reason
                 FROM hosted_room_peer_controls WHERE room_id='room-1'"""
        ).fetchone()
    assert row == ("quarantined", "invalid_stored_link")
    repeated = controls.load_peer_control_links(db, now=31)
    assert repeated.links == ()
    assert repeated.quarantined == 0


def test_peer_load_is_bounded_and_reports_truncation(tmp_path):
    db = tmp_path / "state.db"
    token = secrets.token_urlsafe(controls.TOKEN_BYTES)
    for index in range(3):
        _save(db, token, room_id=f"room-{index}", member_id=f"member-{index}")

    loaded = controls.load_peer_control_links(db, limit=2, now=30)
    assert len(loaded.links) == 2
    assert loaded.truncated is True


def test_remote_retry_plan_and_result_are_idempotent_and_conflict_safe(tmp_path):
    db = tmp_path / "state.db"
    first = controls.begin_control_retry(
        db,
        command_id="retry-1",
        room_id="room-1",
        member_id="member-1",
        task_ids=["task-1", "task-2"],
        now=20,
    )
    replay = controls.begin_control_retry(
        db,
        command_id="retry-1",
        room_id="room-1",
        member_id="member-1",
        task_ids=["task-1", "task-2"],
        now=21,
    )
    assert first.idempotent is False
    assert replay.idempotent is True
    assert replay.task_ids == ("task-1", "task-2")
    pending = controls.load_pending_control_retries(db, room_id="room-1")
    assert [(item.command_id, item.member_id, item.task_ids) for item in pending] == [
        ("retry-1", "member-1", ("task-1", "task-2"))
    ]

    completed = controls.complete_control_retry(
        db,
        command_id="retry-1",
        result={"retried": 2},
        now=22,
    )
    assert completed == {"retried": 2}
    assert controls.load_pending_control_retries(db, room_id="room-1") == ()
    final = controls.begin_control_retry(
        db,
        command_id="retry-1",
        room_id="room-1",
        member_id="member-1",
        task_ids=["task-1", "task-2"],
        now=23,
    )
    assert final.result == {"retried": 2}

    with pytest.raises(controls.HostedRoomControlConflictError):
        controls.begin_control_retry(
            db,
            command_id="retry-1",
            room_id="room-1",
            member_id="member-1",
            task_ids=["task-3"],
            now=24,
        )


def test_completed_control_retries_make_room_at_the_cap(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(controls, "MAX_CONTROL_COMMANDS", 2)
    controls.begin_control_retry(
        db,
        command_id="completed",
        room_id="room-1",
        member_id="member-1",
        task_ids=["task-1"],
        now=1,
    )
    controls.complete_control_retry(
        db,
        command_id="completed",
        result={"retried": 1},
        now=2,
    )
    controls.begin_control_retry(
        db,
        command_id="pending",
        room_id="room-1",
        member_id="member-1",
        task_ids=["task-2"],
        now=3,
    )

    admitted = controls.begin_control_retry(
        db,
        command_id="new",
        room_id="room-1",
        member_id="member-1",
        task_ids=["task-3"],
        now=4,
    )

    assert admitted.task_ids == ("task-3",)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT command_id, state FROM hosted_room_control_commands "
            "ORDER BY command_id"
        ).fetchall()
    assert rows == [("new", "pending"), ("pending", "pending")]


def test_pending_control_retries_fail_closed_at_the_cap(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(controls, "MAX_CONTROL_COMMANDS", 2)
    for index in range(2):
        controls.begin_control_retry(
            db,
            command_id=f"pending-{index}",
            room_id="room-1",
            member_id="member-1",
            task_ids=[f"task-{index}"],
            now=index + 1,
        )

    with pytest.raises(controls.HostedRoomControlError, match="limit reached"):
        controls.begin_control_retry(
            db,
            command_id="pending-3",
            room_id="room-1",
            member_id="member-1",
            task_ids=["task-3"],
            now=3,
        )


@pytest.mark.parametrize(
    "stored_task_ids",
    ["not-json", '{"task-real":true}', '"task-real"'],
)
def test_pending_retry_loader_quarantines_malformed_rows(
    tmp_path,
    stored_task_ids,
):
    db = tmp_path / "state.db"
    controls.begin_control_retry(
        db,
        command_id="retry-invalid",
        room_id="room-1",
        member_id="member-1",
        task_ids=["task-1"],
        now=20,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_control_commands
                  SET task_ids_json=?
                WHERE command_id='retry-invalid'""",
            (stored_task_ids,),
        )
        conn.commit()

    assert controls.load_pending_control_retries(db, room_id="room-1") == ()
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """SELECT state, result_json
                 FROM hosted_room_control_commands
                WHERE command_id='retry-invalid'"""
        ).fetchone()
    assert row[0] == "completed"
    assert json.loads(row[1]) == {
        "action": "retry",
        "error": "invalid_stored_plan",
        "retried": 0,
    }


def test_failed_retry_rotation_does_not_starve_newer_commands(tmp_path):
    db = tmp_path / "state.db"
    for index in range(9):
        controls.begin_control_retry(
            db,
            command_id=f"retry-{index}",
            room_id="room-1",
            member_id="member-1",
            task_ids=[f"task-{index}"],
            now=20 + index,
        )
    first = controls.load_pending_control_retries(db, room_id="room-1", limit=8)
    assert [item.command_id for item in first] == [
        f"retry-{index}" for index in range(8)
    ]
    for item in first:
        assert controls.defer_control_retry(
            db,
            command_id=item.command_id,
            now=100,
        )

    rotated = controls.load_pending_control_retries(db, room_id="room-1", limit=8)
    assert rotated[0].command_id == "retry-8"


def test_concurrent_home_issuance_allows_only_one_token_per_scope(tmp_path):
    db = tmp_path / "state.db"
    _create_room(db)

    def issue(_index):
        try:
            return _issue(db).control_token
        except controls.HostedRoomControlConflictError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(issue, range(16)))
    assert len([token for token in results if token is not None]) == 1


def test_concurrent_peer_link_saves_do_not_lose_records(tmp_path):
    db = tmp_path / "state.db"
    token = secrets.token_urlsafe(controls.TOKEN_BYTES)

    def save(index):
        return _save(
            db,
            token,
            room_id=f"room-{index}",
            member_id=f"member-{index}",
            home_url=f"https://home-{index}.example.test",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(save, range(24)))
    assert all(not result.idempotent for result in results)
    assert len(controls.load_peer_control_links(db, now=30).links) == 24


def test_state_database_permissions_are_owner_only_best_effort(tmp_path):
    db = tmp_path / "state.db"
    token = secrets.token_urlsafe(controls.TOKEN_BYTES)
    _save(db, token)
    if os.name == "posix":
        assert stat.S_IMODE(db.stat().st_mode) == 0o600
