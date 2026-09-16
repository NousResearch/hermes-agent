"""Local cooperative attach must be advertised and fenced without opening the OAuth gate."""

from fastapi.testclient import TestClient

from tui_gateway import server

from hermes_cli.active_sessions import active_session_registry_snapshot, try_acquire_active_session
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS


def _bind_loopback_runtime(*, auth_required=False, host="127.0.0.1", port=8234):
    from hermes_cli import web_server

    previous = {
        "auth_required": getattr(web_server.app.state, "auth_required", False),
        "bound_host": getattr(web_server.app.state, "bound_host", None),
        "bound_port": getattr(web_server.app.state, "bound_port", None),
    }
    web_server.app.state.auth_required = auth_required
    web_server.app.state.bound_host = host
    web_server.app.state.bound_port = port
    return web_server, previous


def _restore_runtime(web_server, previous):
    for name, value in previous.items():
        setattr(web_server.app.state, name, value)


def test_session_attach_is_not_on_the_shared_public_api_allowlist():
    assert "/api/session-attach" not in PUBLIC_API_PATHS


def test_desktop_owner_advertises_origin_and_loopback_handshake_returns_ws(tmp_path):
    from hermes_cli.web_server_chat import _build_gateway_ws_url

    web_server, previous = _bind_loopback_runtime()
    try:
        lease, refusal = server._claim_active_session_slot(
            "live-chat", live_session_id="live", surface="desktop", profile_home=tmp_path,
        )
        assert refusal is None
        entry = active_session_registry_snapshot(tmp_path)[0]
        origin = entry["metadata"]["shared_runtime_url"]
        assert origin == "http://127.0.0.1:8234"
        try:
            client = TestClient(web_server.app, base_url=origin, client=("127.0.0.1", 4321))
            response = client.get(
                "/api/session-attach",
                params={
                    "session_id": "live-chat",
                    "lease_id": lease.lease_id,
                    "profile_home": str(tmp_path.resolve()),
                },
            )
            assert response.status_code == 200
            body = response.json()
            expected_ws = _build_gateway_ws_url()
            assert expected_ws
            assert body == {
                "session_id": "live-chat",
                "lease_id": lease.lease_id,
                "profile_home": str(tmp_path.resolve()),
                "websocket_url": expected_ws,
            }
            assert body["websocket_url"].startswith("ws://127.0.0.1:8234/api/ws")

            mismatch = client.get(
                "/api/session-attach",
                params={
                    "session_id": "live-chat",
                    "lease_id": "not-this-lease",
                    "profile_home": str(tmp_path.resolve()),
                },
            )
            assert mismatch.status_code == 403
            assert active_session_registry_snapshot(tmp_path)[0]["lease_id"] == lease.lease_id
        finally:
            lease.release()
    finally:
        _restore_runtime(web_server, previous)


def test_non_loopback_peer_cannot_mint_the_gateway_url(tmp_path):
    web_server, previous = _bind_loopback_runtime()
    try:
        lease, refusal = server._claim_active_session_slot(
            "live-chat", live_session_id="live", surface="desktop", profile_home=tmp_path,
        )
        assert refusal is None
        try:
            client = TestClient(
                web_server.app, base_url="http://127.0.0.1:8234", client=("8.8.8.8", 4321),
            )
            response = client.get(
                "/api/session-attach",
                params={
                    "session_id": "live-chat",
                    "lease_id": lease.lease_id,
                    "profile_home": str(tmp_path.resolve()),
                },
            )
            assert response.status_code == 403
            assert active_session_registry_snapshot(tmp_path)[0]["lease_id"] == lease.lease_id
        finally:
            lease.release()
    finally:
        _restore_runtime(web_server, previous)


def test_oauth_gate_still_requires_auth_for_session_attach(tmp_path):
    web_server, previous = _bind_loopback_runtime(auth_required=True)
    try:
        lease, refusal = server._claim_active_session_slot(
            "live-chat", live_session_id="live", surface="desktop", profile_home=tmp_path,
        )
        assert refusal is None
        try:
            entry = active_session_registry_snapshot(tmp_path)[0]
            assert "shared_runtime_url" not in (entry.get("metadata") or {})
            client = TestClient(
                web_server.app, base_url="http://127.0.0.1:8234", client=("127.0.0.1", 4321),
            )
            response = client.get(
                "/api/session-attach",
                params={
                    "session_id": "live-chat",
                    "lease_id": lease.lease_id,
                    "profile_home": str(tmp_path.resolve()),
                },
            )
            assert response.status_code == 401
        finally:
            lease.release()
    finally:
        _restore_runtime(web_server, previous)


def test_cli_and_gateway_owners_keep_leases_without_advertising(tmp_path):
    from hermes_cli.shared_session_attach import discover_attach_url

    web_server, previous = _bind_loopback_runtime()
    try:
        cli_lease, cli_refusal = try_acquire_active_session(
            session_id="cli-chat",
            surface="cli",
            config={},
            registry_home=tmp_path,
            metadata={"live_session_id": "cli-chat"},
        )
        assert cli_refusal is None
        gateway_lease, gateway_refusal = try_acquire_active_session(
            session_id="gw-chat",
            surface="gateway:telegram",
            config={},
            registry_home=tmp_path,
            metadata={"platform": "telegram", "chat_id": "1", "user_id": "2", "live_session_id": "gw-chat"},
        )
        assert gateway_refusal is None
        try:
            import pytest

            with pytest.raises(ValueError, match="does not advertise"):
                discover_attach_url("cli-chat", registry_home=tmp_path)
            with pytest.raises(ValueError, match="does not advertise"):
                discover_attach_url("gw-chat", registry_home=tmp_path)
            snapshot = {row["session_id"]: row["lease_id"] for row in active_session_registry_snapshot(tmp_path)}
            assert snapshot["cli-chat"] == cli_lease.lease_id
            assert snapshot["gw-chat"] == gateway_lease.lease_id
        finally:
            cli_lease.release()
            gateway_lease.release()
    finally:
        _restore_runtime(web_server, previous)
