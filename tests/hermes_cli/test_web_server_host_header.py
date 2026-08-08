"""Tests for GHSA-ppp5-vxwm-4cf7 — Host-header validation.

DNS rebinding defence: a victim browser that has the dashboard open
could be tricked into fetching from an attacker-controlled hostname
that TTL-flips to 127.0.0.1. Same-origin / CORS checks won't help —
the browser now treats the attacker origin as same-origin. Validating
the Host header at the application layer rejects the attack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


class TestHostHeaderValidator:
    """Unit test the _is_accepted_host helper directly — cheaper and
    more thorough than spinning up the full FastAPI app."""



    def test_zero_zero_bind_accepts_anything(self):
        """0.0.0.0 means operator explicitly opted into all-interfaces
        (requires --insecure). No Host-layer defence is possible — rely
        on operator network controls."""
        from hermes_cli.web_server import _is_accepted_host

        for host in ("10.0.0.5", "evil.example", "my-server.corp.net"):
            assert _is_accepted_host(host, frozenset({"0.0.0.0"}))
            assert _is_accepted_host(host + ":9119", frozenset({"0.0.0.0"}))

    def test_explicit_non_loopback_bind_requires_exact_match(self):
        """If the operator bound to a specific non-loopback hostname,
        the Host header must match exactly."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("my-server.corp.net", frozenset({"my-server.corp.net"}))
        assert _is_accepted_host("my-server.corp.net:9119", frozenset({"my-server.corp.net"}))
        # Different host — reject
        assert not _is_accepted_host("evil.example", frozenset({"my-server.corp.net"}))
        # Loopback — reject (we bound to a specific non-loopback name)
        assert not _is_accepted_host("localhost", frozenset({"my-server.corp.net"}))

    def test_loopback_bind_accepts_any_loopback(self):
        """Loopback alias bind: 127.0.0.1, localhost, and ::1 are all
        treated as equivalent, so any loopback Host header is accepted."""
        from hermes_cli.web_server import _is_accepted_host

        for loopback in ("127.0.0.1", "localhost", "::1"):
            for target in ("127.0.0.1", "localhost", "::1"):
                assert _is_accepted_host(target, frozenset({loopback}))
                assert _is_accepted_host(f"{target}:9119", frozenset({loopback}))

    def test_ipv6_bracketed_host(self):
        """IPv6 Host headers with brackets are parsed correctly."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("[::1]:9119", frozenset({"::1"}))
        assert _is_accepted_host("[::1]", frozenset({"::1"}))
        assert not _is_accepted_host("[::1]:9119", frozenset({"192.168.1.1"}))

    def test_dual_stack_accepts_either_host(self):
        """When bound to both an IPv4 and IPv6 host, a request targeting
        EITHER address is legitimate."""
        from hermes_cli.web_server import _is_accepted_host

        hosts = frozenset({"0.0.0.0", "::"})
        assert _is_accepted_host("10.0.0.5", hosts)
        assert _is_accepted_host("[::1]", hosts)

    def test_dual_stack_loopback_accepts_any_loopback(self):
        """Dual loopback bind (127.0.0.1 + ::1) accepts any loopback Host."""
        from hermes_cli.web_server import _is_accepted_host

        hosts = frozenset({"127.0.0.1", "::1"})
        for target in ("127.0.0.1", "localhost", "::1"):
            assert _is_accepted_host(target, hosts)
        assert not _is_accepted_host("10.0.0.5", hosts)
        assert not _is_accepted_host("evil.example", hosts)

    def test_mixed_loopback_public_accepts_either(self):
        """Mixed loopback + non-loopback bind: accept if the Host matches
        ANY bound host."""
        from hermes_cli.web_server import _is_accepted_host

        hosts = frozenset({"127.0.0.1", "192.168.1.100"})
        assert _is_accepted_host("127.0.0.1", hosts)
        assert _is_accepted_host("192.168.1.100", hosts)
        assert _is_accepted_host("localhost", hosts)  # loopback alias
        assert not _is_accepted_host("10.0.0.5", hosts)  # non-loopback, not bound



class TestHostHeaderMiddleware:
    """End-to-end test via the FastAPI app — verify the middleware
    rejects bad Host headers with 400."""

    def test_rebinding_request_rejected(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Simulate start_server having set the bound_hosts
        app.state.bound_hosts = frozenset({"127.0.0.1"})
        try:
            client = TestClient(app)
            # The TestClient sends Host: testserver by default — which is
            # NOT a loopback alias, so the middleware must reject it.
            resp = client.get(
                "/api/status",
                headers={"Host": "evil.example"},
            )
            assert resp.status_code == 400
            assert "Invalid Host header" in resp.json()["detail"]
        finally:
            # Clean up so other tests don't inherit the bound_hosts
            if hasattr(app.state, "bound_hosts"):
                del app.state.bound_hosts


    def test_no_bound_host_skips_validation(self):
        """If app.state.bound_hosts isn't set (e.g. running under test
        infra without calling start_server), middleware must pass through
        rather than crash."""
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Make sure bound_hosts isn't set
        if hasattr(app.state, "bound_hosts"):
            del app.state.bound_hosts

        client = TestClient(app)
        resp = client.get("/api/status")
        # Should get through to the route (even if it errors for other reasons)
        assert resp.status_code != 400


class TestWebSocketHostOriginGuard:
    """WebSocket upgrades must enforce the same dashboard boundary as HTTP."""

    def test_rebinding_websocket_host_is_rejected(self, monkeypatch):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_hosts", frozenset({"127.0.0.1"}), raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "evil.example",
                    "Origin": "http://evil.example",
                },
            ):
                pass

        assert exc.value.code == 4403


    def test_loopback_websocket_host_and_origin_are_accepted(self, monkeypatch):
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_hosts", frozenset({"127.0.0.1"}), raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "localhost:9119",
                "Origin": "http://localhost:9119",
            },
        ):
            pass
