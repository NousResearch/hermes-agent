"""Local owner discovery must fence profile and lease identity."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli.active_sessions import try_acquire_active_session


def test_discovery_uses_exact_profile_and_owner_handshake(tmp_path):
    import hashlib
    import hmac as hmac_mod
    from urllib.parse import parse_qs, urlsplit

    from hermes_cli.shared_session_attach import discover_attach_url

    home = tmp_path / "profile"
    other = tmp_path / "other"
    reply = {}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            params = parse_qs(urlsplit(self.path).query)
            proof = hmac_mod.new(
                lease.lease_id.encode(),
                f"hermes-session-attach:{params['session_id'][0]}:{params['nonce'][0]}".encode(),
                hashlib.sha256,
            ).hexdigest()
            payload = json.dumps(
                {**reply, "nonce": params["nonce"][0], "attach_proof": proof}
            ).encode()
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
    reply.update(session_id="same-id",
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
        with pytest.raises(ValueError, match="not available in this build") as caught:
            discover_attach_url("old", registry_home=tmp_path)
        first, details = str(caught.value).splitlines()
        assert "hermes --resume old" in first
        assert details.startswith("Details: ")
        assert active_session_registry_snapshot(tmp_path)[0]["lease_id"] == lease.lease_id
        registry = tmp_path / "runtime" / "active_sessions.json"
        with monkeypatch.context() as patch:
            # Our own pid is never probed (#108005), so model a FOREIGN owner whose inspection is denied.
            from hermes_cli.active_sessions import _read_entries, _write_entries
            entries = _read_entries(registry)
            entries[0]["pid"] = os.getpid() + 2**22
            _write_entries(registry, entries)
            before = registry.read_bytes()

            def denied(pid):
                raise PermissionError("process inspection denied")
            patch.setattr("gateway.status._pid_exists", denied)
            with pytest.raises(RuntimeError, match="liveness is unknown"):
                discover_attach_url("old", registry_home=tmp_path)
        assert registry.read_bytes() == before
    finally:
        lease.release()


def test_discovery_failure_message_names_state_and_resume_path(tmp_path):
    """A refused handshake keeps the lease and points at the working alternative."""
    from hermes_cli.shared_session_attach import discover_attach_url

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    lease, error = try_acquire_active_session(
        session_id="held", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "live", "shared_runtime_url": origin},
    )
    assert error is None
    try:
        with pytest.raises(ValueError, match="just failed") as caught:
            discover_attach_url("held", registry_home=tmp_path)
        first, details = str(caught.value).splitlines()
        assert "hermes --resume held" in first
        assert details.startswith("Details: ")
    finally:
        lease.release()
        server.shutdown()
        server.server_close()
        thread.join()


def test_discovery_refuses_a_listener_that_only_echoes_the_request(tmp_path):
    """A rogue loopback listener can echo every value the client sent; the handshake
    must demand proof of a secret the listener cannot learn from the request."""
    from urllib.parse import parse_qs, urlsplit

    from hermes_cli.shared_session_attach import discover_attach_url

    extra = {}

    class RogueHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = parse_qs(urlsplit(self.path).query)
            body = json.dumps({
                **{k: v[0] for k, v in params.items()},
                **extra,
                "websocket_url": (
                    f"ws://127.0.0.1:{self.server.server_port}/api/ws?token=stolen"
                ),
            }).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RogueHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    lease, error = try_acquire_active_session(
        session_id="prey", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "live", "shared_runtime_url": origin},
    )
    assert error is None
    try:
        with pytest.raises(ValueError, match="identity"):
            discover_attach_url("prey", registry_home=tmp_path)
        # A forged proof still refuses, and a non-ASCII one cannot crash the check.
        extra["attach_proof"] = "0" * 64
        with pytest.raises(ValueError, match="identity"):
            discover_attach_url("prey", registry_home=tmp_path)
        extra["attach_proof"] = "zéro forgé"
        with pytest.raises(ValueError, match="identity"):
            discover_attach_url("prey", registry_home=tmp_path)
    finally:
        lease.release()
        server.shutdown()
        server.server_close()
        thread.join()


def test_the_request_never_discloses_the_lease_id(tmp_path):
    """The lease id is the handshake secret: it must not travel in the request."""
    import hashlib
    import hmac as hmac_mod
    from urllib.parse import parse_qs, urlsplit

    from hermes_cli.shared_session_attach import discover_attach_url

    requests = []
    reply = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            params = parse_qs(urlsplit(self.path).query)
            proof = hmac_mod.new(
                lease.lease_id.encode(),
                f"hermes-session-attach:{params['session_id'][0]}:{params['nonce'][0]}".encode(),
                hashlib.sha256,
            ).hexdigest()
            body = json.dumps({
                **reply, "nonce": params["nonce"][0], "attach_proof": proof,
            }).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "profile"
    origin = f"http://127.0.0.1:{server.server_port}"
    lease, error = try_acquire_active_session(
        session_id="quiet", surface="desktop", config={}, registry_home=home,
        metadata={"live_session_id": "live", "shared_runtime_url": origin},
    )
    assert error is None
    reply.update(session_id="quiet", profile_home=str(home.resolve()),
                 websocket_url=origin.replace("http:", "ws:") + "/api/ws?token=real-token")
    try:
        assert discover_attach_url("quiet", registry_home=home) == reply["websocket_url"]
        sent = parse_qs(urlsplit(requests[0]).query)
        assert "lease_id" not in sent
        assert lease.lease_id not in requests[0]
        assert len(sent["client_proof"][0]) == 64  # hex MAC: proof without disclosure
    finally:
        lease.release()
        server.shutdown()
        server.server_close()
        thread.join()


def _rogue_server():
    """A listener that answers /api/session-attach by echoing the request's own
    parameters and offering its own websocket: the registry's port, hijacked."""
    from urllib.parse import parse_qs, urlsplit

    class RogueHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = parse_qs(urlsplit(self.path).query)
            body = json.dumps({
                **{k: v[0] for k, v in params.items()},
                "websocket_url": (
                    f"ws://127.0.0.1:{self.server.server_port}/api/ws?token=stolen"
                ),
            }).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RogueHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def test_e2e_tui_attachment_refuses_a_rogue_owner_and_keeps_the_env(tmp_path):
    """configure_tui_attachment is the exact seam _launch_tui calls: a rogue
    listener must leave HERMES_TUI_GATEWAY_URL unset and raise."""
    from hermes_cli.shared_session_attach import configure_tui_attachment

    server, thread, origin = _rogue_server()
    lease, error = try_acquire_active_session(
        session_id="prey", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "live", "shared_runtime_url": origin},
    )
    assert error is None
    try:
        env = {}
        with pytest.raises(ValueError, match="identity"):
            configure_tui_attachment(env, "prey", registry_home=tmp_path)
        assert "HERMES_TUI_GATEWAY_URL" not in env
    finally:
        lease.release()
        server.shutdown()
        server.server_close()
        thread.join()


def test_e2e_launch_tui_exits_cleanly_against_a_rogue_listener(tmp_path, monkeypatch, capsys):
    """hermes --resume against a hijacked port: the refusal reaches the user as
    Error + exit 1, before any TUI child is spawned."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    server, thread, origin = _rogue_server()
    lease, error = try_acquire_active_session(
        session_id="prey", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "live", "shared_runtime_url": origin},
    )
    assert error is None
    try:
        from hermes_cli.main_tui_launch import _launch_tui
        with pytest.raises(SystemExit) as caught:
            _launch_tui(resume_session_id="prey")
        assert caught.value.code == 1
        assert "identity" in capsys.readouterr().err
    finally:
        lease.release()
        server.shutdown()
        server.server_close()
        thread.join()
