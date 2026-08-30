"""Contracts for autonomous cross-gateway hosted-room members."""

from __future__ import annotations

import hashlib
import json
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gateway.hosted_room_execution_policy import execution_policy_mapping
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    HostedMemberDispatch,
    HostedRoomGrantError,
    HostedRoomPeerError,
    PROTOCOL_VERSION,
    RoomLinkProbe,
    catalog_mapping,
    derive_room_grant_secret,
    gateway_room_grant_secret,
    issue_room_grant,
    local_room_link_endpoint,
    select_room_link,
    verify_room_grant,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


SECRET = b"s" * 32
EXECUTION_POLICY = execution_policy_mapping(target_profile="reviewer")


def test_gateway_room_grant_secret_is_private_persistent_and_not_an_api_key(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    profile_home = home / "profiles" / "reviewer"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    first = gateway_room_grant_secret()
    token = set_hermes_home_override(str(profile_home))
    try:
        second = gateway_room_grant_secret()
    finally:
        reset_hermes_home_override(token)

    secret_path = home / ".room-link-grant-secret"
    assert first == second
    assert len(first) == 32
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
    assert secret_path.read_bytes() != first
    assert first != derive_room_grant_secret("gateway-api-key-1234567890")


def test_gateway_room_grant_secret_is_atomic_across_concurrent_workers(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    with ThreadPoolExecutor(max_workers=8) as pool:
        secrets = list(pool.map(lambda _index: gateway_room_grant_secret(), range(8)))

    assert len(set(secrets)) == 1
    assert (home / ".room-link-grant-secret").stat().st_size == 32


def test_gateway_room_grant_secret_is_cached_by_installation_root(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    first = gateway_room_grant_secret()
    original_read = Path.read_bytes

    def reject_secret_reread(path):
        if path == home / ".room-link-grant-secret":
            raise AssertionError("grant secret was read again")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", reject_secret_reread)
    assert gateway_room_grant_secret() == first


def test_room_link_protocol_fixture_matches_backend_contract():
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "room_link_protocol_v2.json").read_text(
            encoding="utf-8"
        )
    )

    assert fixture["protocol_version"] == PROTOCOL_VERSION
    assert fixture["catalog"]["protocol_versions"] == [PROTOCOL_VERSION]


def test_room_link_endpoint_reads_supported_config_with_env_override(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "gateway:\n  room_link_url: https://configured.example.test/hermes\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_ROOM_LINK_URL", raising=False)
    assert local_room_link_endpoint() == {
        "available": True,
        "url": "https://configured.example.test/hermes",
        "transport_security": "tls",
    }

    monkeypatch.setenv(
        "HERMES_ROOM_LINK_URL", "https://override.example.test/hermes"
    )
    assert local_room_link_endpoint()["url"] == (
        "https://override.example.test/hermes"
    )


def test_named_profile_inherits_gateway_room_link_endpoint(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "reviewer"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "gateway:\n  room_link_url: https://gateway.example.test/hermes\n",
        encoding="utf-8",
    )
    (profile / "config.yaml").write_text("gateway: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_ROOM_LINK_URL", raising=False)

    token = set_hermes_home_override(profile)
    try:
        assert local_room_link_endpoint() == {
            "available": True,
            "url": "https://gateway.example.test/hermes",
            "transport_security": "tls",
        }
    finally:
        reset_hermes_home_override(token)


def test_named_profile_room_link_override_wins_over_gateway_root(
    tmp_path, monkeypatch
):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "reviewer"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "gateway:\n  room_link_url: https://gateway.example.test/hermes\n",
        encoding="utf-8",
    )
    (profile / "config.yaml").write_text(
        "gateway:\n  room_link_url: https://profile.example.test/hermes\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_ROOM_LINK_URL", raising=False)

    token = set_hermes_home_override(profile)
    try:
        assert local_room_link_endpoint()["url"] == (
            "https://profile.example.test/hermes"
        )
    finally:
        reset_hermes_home_override(token)


def _dispatch(**overrides):
    prompt = overrides.pop("prompt", "Review the current room state.")
    value = {
        "protocol_version": 2,
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "gateway-home",
        "authority_epoch": 2,
        "member_id": "member-reviewer",
        "target_install_id": "install-peer",
        "target_profile": "reviewer",
        "task_id": "task-1",
        "execution_generation": 1,
        "source_event_seq": 9,
        "cancellation_scope_id": "cancel-1",
        "prompt": prompt,
        "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        "capability_digest": "a" * 64,
        "execution_policy_digest": EXECUTION_POLICY["policy_digest"],
        "trace_id": "trace-1",
        **overrides,
    }
    return HostedMemberDispatch.from_mapping(value)


def test_catalog_digest_is_canonical_and_tamper_evident():
    value = catalog_mapping(
        installation_id="install-peer",
        protocol_versions=(2,),
        link_modes=("direct", "pull"),
        persistent_process=True,
    )
    catalog = GatewayRoomCatalog.from_mapping(value)
    assert catalog.text is True
    assert catalog.attachments is False

    value["attachments"] = True
    with pytest.raises(HostedRoomPeerError, match="catalog_digest"):
        GatewayRoomCatalog.from_mapping(value)


def test_dispatch_rejects_unknown_fields_and_prompt_digest_mismatch():
    with pytest.raises(HostedRoomPeerError, match="unknown fields"):
        _dispatch(extra=True)
    with pytest.raises(HostedRoomPeerError, match="prompt_digest"):
        _dispatch(prompt_digest="0" * 64)


def test_room_grant_is_scoped_to_exact_room_home_target_and_profile():
    dispatch = _dispatch()
    token = issue_room_grant(
        SECRET,
        grant_id="grant-1",
        room_id=dispatch.room_id,
        home_install_id=dispatch.home_install_id,
        authority_gateway_id=dispatch.authority_gateway_id,
        authority_epoch=dispatch.authority_epoch,
        member_id=dispatch.member_id,
        target_install_id=dispatch.target_install_id,
        target_profile=dispatch.target_profile,
        execution_policy_digest=dispatch.execution_policy_digest,
        issued_at=100,
        ttl_seconds=60,
    )
    claims = verify_room_grant(SECRET, token, dispatch, now=120)
    assert claims["grant_id"] == "grant-1"

    wrong_target = _dispatch(target_profile="other")
    with pytest.raises(HostedRoomGrantError, match="scope"):
        verify_room_grant(SECRET, token, wrong_target, now=120)
    with pytest.raises(HostedRoomGrantError, match="scope"):
        verify_room_grant(
            SECRET, token, _dispatch(member_id="member-other"), now=120
        )
    with pytest.raises(HostedRoomGrantError, match="scope"):
        verify_room_grant(
            SECRET, token, _dispatch(authority_epoch=999), now=120
        )


def test_room_grant_fails_closed_for_tamper_expiry_and_permission():
    dispatch = _dispatch()
    token = issue_room_grant(
        SECRET,
        grant_id="grant-1",
        room_id=dispatch.room_id,
        home_install_id=dispatch.home_install_id,
        authority_gateway_id=dispatch.authority_gateway_id,
        authority_epoch=dispatch.authority_epoch,
        member_id=dispatch.member_id,
        target_install_id=dispatch.target_install_id,
        target_profile=dispatch.target_profile,
        execution_policy_digest=dispatch.execution_policy_digest,
        permissions=("status",),
        issued_at=100,
        ttl_seconds=10,
        status_expires_at=120,
    )
    with pytest.raises(HostedRoomGrantError, match="allow"):
        verify_room_grant(SECRET, token, dispatch, now=105)
    # Dispatch expires quickly, while observation/stop remains available for
    # bounded headless recovery after the Desktop has closed.
    assert (
        verify_room_grant(SECRET, token, dispatch, permission="status", now=111)[
            "grant_id"
        ]
        == "grant-1"
    )
    with pytest.raises(HostedRoomGrantError, match="expired"):
        verify_room_grant(
            SECRET,
            token,
            dispatch,
            permission="status",
            now=100 + 30 * 24 * 60 * 60,
        )
    with pytest.raises(HostedRoomGrantError, match="signature"):
        verify_room_grant(SECRET, token[:-1] + "A", dispatch, now=105)


def test_link_selection_prefers_safe_direct_then_overlay_then_relay_then_pull():
    selected = select_room_link(
        [
            RoomLinkProbe("relay", True, True, 10),
            RoomLinkProbe("direct", True, True, 50),
            RoomLinkProbe("overlay", True, True, 5),
            RoomLinkProbe("pull", True, True, 1),
        ],
        desktop_available=False,
    )
    assert selected is not None
    assert selected.mode == "direct"


def test_link_selection_never_falls_back_to_unencrypted_route():
    assert (
        select_room_link(
            [RoomLinkProbe("direct", True, False, 1)],
            desktop_available=False,
        )
        is None
    )
    fallback = select_room_link([], desktop_available=True)
    assert fallback is not None
    assert fallback.mode == "desktop"


def test_local_catalog_is_honest_for_app_managed_process(monkeypatch):
    from gateway.hosted_room_peer import local_catalog_mapping

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    catalog = local_catalog_mapping(installation_id="install-desktop")
    assert catalog["persistent_process"] is False
    assert catalog["link_modes"] == ["direct"]


@pytest.mark.parametrize(
    ("configured", "available", "reason", "security"),
    [
        (None, False, "not_configured", None),
        ("http://peer.example.test:8000", False, "invalid_configuration", None),
        ("http://127.0.0.1:8000", True, None, "loopback"),
        ("https://peer.example.test", True, None, "tls"),
    ],
)
def test_self_advertised_endpoint_is_explicit_and_validated(
    monkeypatch, configured, available, reason, security
):
    from gateway.hosted_room_peer import local_catalog_mapping

    if configured is None:
        monkeypatch.delenv("HERMES_ROOM_LINK_URL", raising=False)
    else:
        monkeypatch.setenv("HERMES_ROOM_LINK_URL", configured)
    endpoint = local_catalog_mapping(installation_id="install-peer")["endpoint"]
    assert endpoint["available"] is available
    if reason is not None:
        assert endpoint == {"available": False, "reason": reason}
    else:
        assert endpoint["transport_security"] == security
