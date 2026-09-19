"""The cooperative attach handshake must be reachable from a live Desktop owner."""

from fastapi.testclient import TestClient

from hermes_cli.active_sessions import active_session_registry_snapshot, try_acquire_active_session


def test_session_attach_returns_the_live_gateway_url_for_its_exact_lease(tmp_path, monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.web_server_chat import cooperative_session_origin

    previous = {
        "auth_required": getattr(web_server.app.state, "auth_required", False),
        "bound_host": getattr(web_server.app.state, "bound_host", None),
        "bound_port": getattr(web_server.app.state, "bound_port", None),
    }
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8234
    origin = cooperative_session_origin()
    assert origin == "http://127.0.0.1:8234"
    monkeypatch.setattr(
        "hermes_cli.web_server_chat._build_gateway_ws_url",
        lambda: "ws://127.0.0.1:8234/api/ws?token=attached",
    )
    monkeypatch.setattr("hermes_cli.web_routers.chat_ws._is_loopback_peer", lambda request: True)
    lease, refusal = try_acquire_active_session(
        session_id="attached-session", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "live", "shared_runtime_url": origin},
    )
    assert refusal is None
    try:
        client = TestClient(web_server.app, base_url=origin)
        response = client.get("/api/session-attach", params={
            "session_id": "attached-session", "lease_id": lease.lease_id,
            "profile_home": str(tmp_path.resolve()),
        })
        assert response.status_code == 200
        assert response.json() == {
            "session_id": "attached-session",
            "lease_id": lease.lease_id,
            "profile_home": str(tmp_path.resolve()),
            "websocket_url": "ws://127.0.0.1:8234/api/ws?token=attached",
        }

        mismatch = client.get("/api/session-attach", params={
            "session_id": "attached-session", "lease_id": "different-lease",
            "profile_home": str(tmp_path.resolve()),
        })
        assert mismatch.status_code == 403
        assert active_session_registry_snapshot(tmp_path)[0]["lease_id"] == lease.lease_id
    finally:
        lease.release()
        for name, value in previous.items():
            setattr(web_server.app.state, name, value)


def test_tui_owner_advertises_only_the_loopback_runtime(tmp_path, monkeypatch):
    import tui_gateway.server as server

    monkeypatch.setattr(
        "hermes_cli.web_server_chat.cooperative_session_origin",
        lambda: "http://127.0.0.1:8234",
    )
    lease, refusal = server._claim_active_session_slot(
        "attached-session", live_session_id="live", surface="desktop", profile_home=tmp_path,
    )
    assert refusal is None
    try:
        entry = active_session_registry_snapshot(tmp_path)[0]
        assert entry["metadata"]["shared_runtime_url"] == "http://127.0.0.1:8234"
    finally:
        lease.release()
