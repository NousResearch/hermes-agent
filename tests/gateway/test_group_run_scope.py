"""The owner input boundary must preserve the existing participant Runs key."""

from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.platforms import api_server_runs
from gateway.platforms.api_server_run_scope import (
    ROOM_RUN_SCOPE_FIELDS, room_run_scope_key, validate_room_run_scope,
)

IDENTITY = dict(room_id="room-one", home_install_id="install-home", authority_gateway_id="authority",
                authority_epoch=3, member_id="member-one", target_install_id="participant", target_profile="research")


def test_scope_codec_matches_actual_runs_owner_and_not_grant_scope():
    class Request(dict):
        method = "POST"
        path = "/v1/runs"

    claims = {**IDENTITY, "status_expires_at": 1234, "grant_id": "not-part-of-run-scope"}
    adapter = SimpleNamespace(_room_grant_token=lambda request: "synthetic",
                              _room_grant_claims=lambda request, **kwargs: claims)
    expected = api_server_runs._run_idempotency_scope(adapter, Request(), _api_server=SimpleNamespace())
    # Persisted protocol vector stays independent when the API adopts the shared codec.
    assert expected == "47b455fd52d0e02d48e39e8ee2c2dfcf15d3d829d4091ae1ad039c9429f05725"
    assert room_run_scope_key(claims) == expected
    assert room_run_scope_key(IDENTITY) != hosted_rooms._room_grant_scope_key(IDENTITY)
    assert list(validate_room_run_scope(dict(reversed(list(IDENTITY.items()))))) == list(ROOM_RUN_SCOPE_FIELDS)
    for field in ROOM_RUN_SCOPE_FIELDS:
        changed = {**IDENTITY, field: 4 if field == "authority_epoch" else IDENTITY[field] + "-other"}
        assert room_run_scope_key(changed) != expected


@pytest.mark.parametrize("identity", [
    None, "opaque-hash", {}, {**IDENTITY, "extra": "PRIVATE"},
    {key: value for key, value in IDENTITY.items() if key != "member_id"},
    *({**IDENTITY, "authority_epoch": value} for value in (True, 0, -1, "3", 3.0, 2**63)),
    *({**IDENTITY, "target_profile": value} for value in (True, 7, "", " default", "default ", "x\0y", "x" * 257)),
])
def test_owner_scope_rejects_ambiguous_or_noncanonical_values(identity):
    with pytest.raises(ValueError) as failure:
        validate_room_run_scope(identity)
    assert "PRIVATE" not in str(failure.value)


def test_codec_preserves_legacy_string_conversion_without_relaxing_validation():
    legacy = {**IDENTITY, "authority_epoch": "3"}
    assert room_run_scope_key(legacy) == room_run_scope_key(IDENTITY)
    with pytest.raises(ValueError):
        validate_room_run_scope(legacy)
