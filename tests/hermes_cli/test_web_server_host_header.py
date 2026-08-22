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
            assert _is_accepted_host(host, "0.0.0.0")
            assert _is_accepted_host(host + ":9119", "0.0.0.0")

    def test_explicit_non_loopback_bind_requires_exact_match(self):
        """If the operator bound to a specific non-loopback hostname,
        the Host header must match exactly."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("my-server.corp.net", "my-server.corp.net")
        assert _is_accepted_host("my-server.corp.net:9119", "my-server.corp.net")
        # Different host — reject
        assert not _is_accepted_host("evil.example", "my-server.corp.net")
        # Loopback — reject (we bound to a specific non-loopback name)
        assert not _is_accepted_host("localhost", "my-server.corp.net")



class TestHostHeaderMiddleware:
    """End-to-end test via the FastAPI app — verify the middleware
    rejects bad Host headers with 400."""

    def test_rebinding_request_rejected(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Simulate start_server having set the bound_host
        app.state.bound_host = "127.0.0.1"
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
            # Clean up so other tests don't inherit the bound_host
            if hasattr(app.state, "bound_host"):
                del app.state.bound_host


    def test_no_bound_host_skips_validation(self):
        """If app.state.bound_host isn't set (e.g. running under test
        infra without calling start_server), middleware must pass through
        rather than crash."""
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Make sure bound_host isn't set
        if hasattr(app.state, "bound_host"):
            del app.state.bound_host

        client = TestClient(app)
        resp = client.get("/api/status")
        # Should get through to the status endpoint, not a 400
        assert resp.status_code != 400


class TestWebSocketHostOriginGuard:
    """WebSocket upgrades must enforce the same dashboard boundary as HTTP."""

    def test_rebinding_websocket_host_is_rejected(self, monkeypatch):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

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

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

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


class TestWebSocketPublicUrlOrigin:
    """Reverse-proxy topology: loopback-bound dashboard fronted by something
    (e.g. cloudflared) that rewrites Host but not Origin. ``dashboard.public_url``
    is the operator's explicit opt-in to also accept that public host — see
    PR description for the cloudflared+Cloudflare-Access scenario this fixes.
    """

    @staticmethod
    def _patch_public_url(monkeypatch, url: str) -> None:
        import hermes_cli.dashboard_auth.prefix as prefix_mod

        monkeypatch.setattr(prefix_mod, "resolve_public_url", lambda: url)

    def test_public_url_with_explicit_port_is_accepted(self, monkeypatch):
        """Regression test: public_host used to be derived via .netloc
        (keeps the port) while the incoming value had its port stripped
        before comparison — a public_url with an explicit port could never
        match. .hostname (+ reusing _is_accepted_host) fixes this."""
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        self._patch_public_url(monkeypatch, "https://hermes.example.com:8443")

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "localhost",
                "Origin": "https://hermes.example.com:8443",
            },
        ):
            pass

    def test_public_url_ipv6_host_is_accepted(self, monkeypatch):
        """Regression test: the old value.split(':', 1)[0] comparison broke
        on IPv6 literals (splits on every colon, not just a port separator).
        _is_accepted_host already handles bracket notation correctly.

        Uses a documentation-range IPv6 literal (RFC 3849, 2001:db8::/32)
        rather than ::1 — a loopback address would be accepted via the
        bound_host check alone and never actually exercise the public_host
        comparison this test targets.
        """
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        self._patch_public_url(monkeypatch, "http://[2001:db8::1]:8443")

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "localhost",
                "Origin": "http://[2001:db8::1]:8443",
            },
        ):
            pass

    def test_evil_origin_still_rejected_when_public_url_is_set(self, monkeypatch):
        """public_url is an explicit opt-in for one specific host — it must
        not become a general Origin bypass."""
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        self._patch_public_url(monkeypatch, "https://hermes.example.com:8443")

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "localhost",
                    "Origin": "https://evil.example.com:8443",
                },
            ):
                pass

        assert exc.value.code == 4403

    def test_public_host_header_still_rejected_with_bound_origin(self, monkeypatch):
        """The public_url relaxation applies to Origin only, per @Kinkoolino-Hermes's
        review — Host stays gated to bound_host with no widening at all. A Host
        header claiming to be the public host must still be rejected even when
        Origin is the ordinary bound-loopback value, since Host reflects what the
        server itself believes it's bound to (unlike Origin, which just reflects
        what the browser sent) — widening that check would be a materially bigger
        relaxation than the one this PR makes."""
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        self._patch_public_url(monkeypatch, "https://hermes.example.com:8443")

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "hermes.example.com:8443",
                    "Origin": "http://localhost",
                },
            ):
                pass

        assert exc.value.code == 4403

    def test_public_host_header_still_rejected_with_public_origin(self, monkeypatch):
        """Same as above, but with Origin also set to the public host -- proves
        rejection is driven purely by the unwidened Host check, not by Origin
        happening to mismatch too."""
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
        self._patch_public_url(monkeypatch, "https://hermes.example.com:8443")

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "hermes.example.com:8443",
                    "Origin": "https://hermes.example.com:8443",
                },
            ):
                pass

        assert exc.value.code == 4403
