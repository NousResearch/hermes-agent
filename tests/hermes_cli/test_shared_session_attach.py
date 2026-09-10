"""Local owner discovery must fence profile and lease identity."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli.active_sessions import try_acquire_active_session


def test_discovery_uses_exact_profile_and_owner_handshake(tmp_path):
    from hermes_cli.shared_session_attach import discover_attach_url

    home = tmp_path / "profile"
    other = tmp_path / "other"
    reply = {}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    lease, error = try_acquire_active_session(
        session_id="same-id", surface="desktop", config={}, registry_home=home,
        metadata={"live_session_id": "live", "shared_runtime_url": origin},
    )
    assert error is None
    reply.update(session_id="same-id", lease_id=lease.lease_id,
                 profile_home=str(home.resolve()), websocket_url=origin.replace("http:", "ws:") + "/api/ws?token=real-token")
    try:
        assert discover_attach_url("same-id", registry_home=other) is None
        assert requests == []
        assert discover_attach_url("same-id", registry_home=home) == reply["websocket_url"]
        assert len(requests) == 1
        from hermes_cli.shared_session_attach import configure_tui_attachment
        env = {"HERMES_TUI_GATEWAY_URL": "   "}
        configure_tui_attachment(env, "same-id", registry_home=home)
        assert env["HERMES_TUI_GATEWAY_URL"] == reply["websocket_url"]
        reply["profile_home"] = str(other.resolve())
        with pytest.raises(ValueError, match="identity"):
            discover_attach_url("same-id", registry_home=home)
        reply["profile_home"] = str(home.resolve())
        reply["websocket_url"] = "ws://example.com/api/ws?token=secret"
        with pytest.raises(ValueError, match="endpoint"):
            discover_attach_url("same-id", registry_home=home)
    finally:
        lease.release()
        server.shutdown()
        server.server_close()
        thread.join()


def test_discovery_refuses_unsupported_owner_without_releasing_lease(tmp_path, monkeypatch):
    from hermes_cli.shared_session_attach import discover_attach_url
    from hermes_cli.active_sessions import active_session_registry_snapshot

    lease, error = try_acquire_active_session(
        session_id="old", surface="desktop", config={}, registry_home=tmp_path,
    )
    assert error is None
    try:
        with pytest.raises(ValueError, match="does not advertise"):
            discover_attach_url("old", registry_home=tmp_path)
        assert active_session_registry_snapshot(tmp_path)[0]["lease_id"] == lease.lease_id
        registry = tmp_path / "runtime" / "active_sessions.json"
        before = registry.read_bytes()
        with monkeypatch.context() as patch:
            def denied(pid):
                raise PermissionError("process inspection denied")
            patch.setattr("gateway.status._pid_exists", denied)
            with pytest.raises(RuntimeError, match="liveness is unknown"):
                discover_attach_url("old", registry_home=tmp_path)
        assert registry.read_bytes() == before
    finally:
        lease.release()
