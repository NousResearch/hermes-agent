"""An authenticated setup can recover current control access without rotating it."""

import sqlite3

import pytest

from gateway import hosted_room_controls as controls, hosted_rooms
from tests.gateway.test_hosted_room_controls import HOME, _create_room


@pytest.fixture
def control_home(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "gateway_room_grant_secret", lambda: b"s" * 32)
    db = tmp_path / "state.db"
    _create_room(db)
    return db


def issue(db, request_id="first", **overrides):
    return controls.issue_home_control_token(db, **{
        "room_id": "room-1", "member_id": "member-1", "authority_gateway_id": HOME,
        "authority_epoch": 1, "request_id": request_id, "expires_at": 620, "now": 20,
        **overrides,
    })


def test_reuse_preserves_current_token_horizon_and_journal(control_home):
    original = issue(control_home)
    reused = issue(control_home, "reconnect", reuse_existing=True, now=30, expires_at=999999)
    assert reused == original
    assert reused.control_token == original.control_token
    assert reused.expires_at == 620
    assert original.control_token.encode() not in control_home.read_bytes()
    with sqlite3.connect(control_home) as conn:
        row = conn.execute("SELECT request_id,created_at,updated_at FROM hosted_room_control_tokens").fetchone()
    assert row == ("first", 20, 20)


def test_default_invitation_conflict_contract_is_unchanged(control_home):
    issue(control_home)
    with pytest.raises(controls.HostedRoomControlConflictError):
        issue(control_home, "different")


@pytest.mark.parametrize("reason", ["revoked", "expired"])
def test_reuse_never_resurrects_an_old_credential(control_home, reason):
    original = issue(control_home)
    assert issue(control_home, "consumed-reuse", reuse_existing=True, now=25).control_token == original.control_token
    now = 700 if reason == "expired" else 30
    if reason == "revoked":
        controls.revoke_home_control_tokens(control_home, room_id="room-1", now=now)
    for request in ("first", "consumed-reuse", "different"):
        with pytest.raises(controls.HostedRoomControlConflictError, match="explicit authorization"):
            issue(control_home, request, reuse_existing=True, now=now, expires_at=now+600)
    fresh = issue(control_home, "explicit-reconnect", reuse_existing=False, now=now, expires_at=now+600)
    assert fresh.control_token != original.control_token


@pytest.mark.parametrize("reason", ["secret-changed", "legacy-random", "disbanded", "wrong-member", "wrong-epoch"])
def test_reuse_requires_recoverable_current_exact_scope(control_home, monkeypatch, reason):
    original = issue(control_home, None if reason == "legacy-random" else "first")
    extra = {}
    if reason == "secret-changed":
        monkeypatch.setattr(controls, "gateway_room_grant_secret", lambda: b"z" * 32)
    elif reason == "disbanded":
        hosted_rooms.disband_room(control_home, room_id="room-1",
                                 expected_gateway_id=HOME, expected_epoch=1, now=30)
    elif reason == "wrong-member":
        extra["member_id"] = "other"
    elif reason == "wrong-epoch":
        extra["authority_epoch"] = 2
    with pytest.raises(controls.HostedRoomControlError):
        issue(control_home, "reconnect", reuse_existing=True, now=40, **extra)
    with sqlite3.connect(control_home) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_control_tokens").fetchone()[0] == 1
    assert original.control_token.encode() not in control_home.read_bytes()
