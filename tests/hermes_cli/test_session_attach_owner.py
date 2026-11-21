"""Owner half of cooperative session attach: the lease advertises us, the handshake answers only
for leases THIS process holds, from loopback, never to a browser — and the end-to-end pair with the
client half (``shared_session_attach``) hands a local ``hermes --tui`` our live ``/api/ws``."""

from __future__ import annotations

import os

import pytest

from hermes_cli.active_sessions import try_acquire_active_session


@pytest.fixture
def advertised(monkeypatch):
    from tui_gateway import shared_runtime

    shared_runtime.advertise_shared_runtime("http://127.0.0.1:4567")
    yield "http://127.0.0.1:4567"
    shared_runtime.advertise_shared_runtime(None)


@pytest.fixture
def client(monkeypatch):
    from starlette.testclient import TestClient

    import hermes_cli.web_server as ws
    import hermes_cli.web_server_chat as chat

    monkeypatch.setattr(chat, "_build_gateway_ws_url", lambda: "ws://127.0.0.1:4567/api/ws?internal=cred")
    # Starlette's default peer is the non-address "testclient"; the handshake must see a loopback IP.
    return TestClient(ws.app, client=("127.0.0.1", 50000))


def _lease(home, session_id="sess-1", origin="http://127.0.0.1:4567"):
    lease, error = try_acquire_active_session(
        session_id=session_id, surface="desktop", config={}, registry_home=home,
        metadata={"live_session_id": "live", **({"shared_runtime_url": origin} if origin else {})},
    )
    assert error is None
    return lease


def test_advertised_origin_must_be_loopback_http():
    from tui_gateway import shared_runtime

    for bad in ["http://192.168.1.5:9119", "https://127.0.0.1:1", "http://127.0.0.1", "http://127.0.0.1:1/x",
                "ws://127.0.0.1:1"]:
        with pytest.raises(ValueError):
            shared_runtime.advertise_shared_runtime(bad)
    shared_runtime.advertise_shared_runtime("http://127.0.0.1:4567/")
    assert shared_runtime.shared_runtime_origin() == "http://127.0.0.1:4567"
    shared_runtime.advertise_shared_runtime(None)
    assert shared_runtime.shared_runtime_origin() is None


def test_lease_metadata_carries_the_advertised_origin(advertised):
    from tui_gateway.session_lifecycle import _lease_metadata

    assert _lease_metadata("live") == {"live_session_id": "live", "shared_runtime_url": advertised}


def test_lease_metadata_is_unchanged_when_nothing_is_advertised():
    from tui_gateway.session_lifecycle import _lease_metadata

    assert _lease_metadata("live") == {"live_session_id": "live"}


def test_handshake_answers_for_an_owned_lease_from_loopback(tmp_path, advertised, client):
    home = tmp_path.resolve()
    lease = _lease(home)
    params = {"session_id": "sess-1", "lease_id": lease.lease_id, "profile_home": str(home)}
    try:
        response = client.get("/api/session-attach", params=params)
        assert response.status_code == 200
        assert response.json() == {**params, "websocket_url": "ws://127.0.0.1:4567/api/ws?internal=cred"}
    finally:
        lease.release()


def test_handshake_refuses_everything_that_is_not_our_lease(tmp_path, advertised, client, monkeypatch):
    home = tmp_path.resolve()
    lease = _lease(home)
    good = {"session_id": "sess-1", "lease_id": lease.lease_id, "profile_home": str(home)}
    try:
        # A browser page fetching us cross-origin, even from loopback.
        assert client.get("/api/session-attach", params=good, headers={"Origin": "http://127.0.0.1:4567"}).status_code == 404
        assert client.get("/api/session-attach", params={**good, "lease_id": "guess"}).status_code == 404
        assert client.get("/api/session-attach", params={**good, "session_id": "other"}).status_code == 404
        assert client.get("/api/session-attach", params={**good, "profile_home": str(tmp_path / "x")}).status_code == 404
        assert client.get("/api/session-attach", params={}).status_code == 404
        # A lease that is not ours (another process) — same registry, foreign pid.
        from hermes_cli.active_sessions import _read_entries, _write_entries
        registry = home / "runtime" / "active_sessions.json"
        entries = _read_entries(registry)
        own_pid = entries[0]["pid"]
        entries[0]["pid"] = os.getpid() + 2**22
        _write_entries(registry, entries)
        assert client.get("/api/session-attach", params=good).status_code == 404
        entries[0]["pid"] = own_pid
        _write_entries(registry, entries)
        # A lease we hold but did not advertise on (written before the bind).
        entries[0]["metadata"].pop("shared_runtime_url")
        _write_entries(registry, entries)
        assert client.get("/api/session-attach", params=good).status_code == 404
    finally:
        lease.release()


def test_handshake_is_closed_when_nothing_is_advertised(tmp_path, client):
    home = tmp_path.resolve()
    lease = _lease(home, origin=None)
    try:
        params = {"session_id": "sess-1", "lease_id": lease.lease_id, "profile_home": str(home)}
        assert client.get("/api/session-attach", params=params).status_code == 404
    finally:
        lease.release()


def test_handshake_refuses_a_non_loopback_peer(tmp_path, advertised, monkeypatch):
    from starlette.testclient import TestClient

    import hermes_cli.web_server as ws
    import hermes_cli.web_server_chat as chat

    monkeypatch.setattr(chat, "_build_gateway_ws_url", lambda: "ws://127.0.0.1:4567/api/ws?internal=cred")
    home = tmp_path.resolve()
    lease = _lease(home)
    try:
        params = {"session_id": "sess-1", "lease_id": lease.lease_id, "profile_home": str(home)}
        remote = TestClient(ws.app, client=("192.168.1.22", 51000))
        assert remote.get("/api/session-attach", params=params).status_code == 404
    finally:
        lease.release()


def test_client_half_attaches_through_the_real_handshake(tmp_path, advertised, client, monkeypatch):
    """The pair: the TUI launcher's discovery reads the lease, calls our handshake over a real socket
    and ends up with our ``/api/ws`` in ``HERMES_TUI_GATEWAY_URL``."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit

    from hermes_cli.shared_session_attach import configure_tui_attachment
    from tui_gateway import shared_runtime

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # relay into the FastAPI app under test
            parts = urlsplit(self.path)
            response = client.get(parts.path + "?" + parts.query)
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    shared_runtime.advertise_shared_runtime(origin)
    monkeypatch.setattr("hermes_cli.web_server_chat._build_gateway_ws_url",
                        lambda: origin.replace("http:", "ws:") + "/api/ws?internal=cred")
    home = tmp_path.resolve()
    lease = _lease(home, origin=origin)
    try:
        env: dict[str, str] = {}
        configure_tui_attachment(env, "sess-1", registry_home=home)
        assert env["HERMES_TUI_GATEWAY_URL"] == origin.replace("http:", "ws:") + "/api/ws?internal=cred"
    finally:
        lease.release()
        server.shutdown()
        server.server_close()
