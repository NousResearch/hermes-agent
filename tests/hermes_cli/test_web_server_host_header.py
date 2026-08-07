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

class TestExtraAcceptedHosts:
    """``dashboard.extra_hosts`` (config.yaml) — operator-trusted
    reverse-proxy / tunnel hostnames in front of a loopback bind (#70059).

    The proxy rewrites the Host header, but the browser-set WebSocket
    Origin header carries the public hostname and cannot be rewritten by
    any proxy — without this opt-in every WS upgrade is refused with
    ``origin_mismatch``."""

    @staticmethod
    def _set_extra_hosts(monkeypatch, value):
        """Point the mtime-cached config read at an in-memory config."""
        import hermes_cli.web_server as ws

        monkeypatch.setattr(
            ws,
            "load_config_readonly",
            lambda: {"dashboard": {"extra_hosts": value}},
        )

    def test_extra_host_accepted_on_loopback_bind(self, monkeypatch):
        from hermes_cli.web_server import _is_accepted_host

        self._set_extra_hosts(monkeypatch, ["atlas.example.com"])
        assert _is_accepted_host("atlas.example.com", "127.0.0.1")
        assert _is_accepted_host("atlas.example.com:9119", "127.0.0.1")
        # Case-insensitive, like every other host comparison here
        assert _is_accepted_host("Atlas.Example.COM", "127.0.0.1")

    def test_other_hosts_still_rejected(self, monkeypatch):
        from hermes_cli.web_server import _is_accepted_host

        self._set_extra_hosts(monkeypatch, ["atlas.example.com"])
        assert not _is_accepted_host("evil.example", "127.0.0.1")
        # Loopback aliases keep working on a loopback bind
        assert _is_accepted_host("localhost:9119", "127.0.0.1")

    def test_unset_config_keeps_strict_behaviour(self, monkeypatch):
        import hermes_cli.web_server as ws
        from hermes_cli.web_server import _is_accepted_host

        monkeypatch.setattr(ws, "load_config_readonly", lambda: {})
        assert not _is_accepted_host("atlas.example.com", "127.0.0.1")

    def test_multiple_entries_and_whitespace(self, monkeypatch):
        from hermes_cli.web_server import _is_accepted_host

        self._set_extra_hosts(
            monkeypatch, [" atlas.example.com ", "hermes.tail1234.ts.net"]
        )
        assert _is_accepted_host("atlas.example.com", "127.0.0.1")
        assert _is_accepted_host("hermes.tail1234.ts.net", "127.0.0.1")
        assert not _is_accepted_host("other.example.com", "127.0.0.1")

    def test_single_string_tolerated(self, monkeypatch):
        """A bare string instead of a one-element list is a natural YAML
        mistake — accept it rather than silently disabling the opt-in."""
        from hermes_cli.web_server import _is_accepted_host

        self._set_extra_hosts(monkeypatch, "atlas.example.com")
        assert _is_accepted_host("atlas.example.com", "127.0.0.1")

    def test_entry_port_suffix_stripped(self, monkeypatch):
        from hermes_cli.web_server import _is_accepted_host

        self._set_extra_hosts(monkeypatch, ["atlas.example.com:9119"])
        assert _is_accepted_host("atlas.example.com", "127.0.0.1")
        assert _is_accepted_host("atlas.example.com:8443", "127.0.0.1")

    def test_malformed_entries_fail_closed(self, monkeypatch):
        """Exact-match only: wildcards, schemes/paths, and non-string
        entries are ignored — never partially honoured."""
        from hermes_cli.web_server import _is_accepted_host

        self._set_extra_hosts(
            monkeypatch,
            ["*.example.com", "https://x.example", "a b.example", 42, None,
             "", "atlas.example.com"],
        )
        assert _is_accepted_host("atlas.example.com", "127.0.0.1")
        assert not _is_accepted_host("x.example", "127.0.0.1")
        assert not _is_accepted_host("sub.example.com", "127.0.0.1")

    def test_empty_entries_ignored(self, monkeypatch):
        from hermes_cli.web_server import _is_accepted_host

        self._set_extra_hosts(monkeypatch, [" ", ""])
        assert not _is_accepted_host("", "127.0.0.1")
        assert not _is_accepted_host("evil.example", "127.0.0.1")

    # -- positive regression coverage through the real request paths --

    def test_http_middleware_accepts_extra_host(self, monkeypatch):
        """The HTTP middleware (host_header_middleware) must accept a
        configured extra host — not just the unit-level helper."""
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        self._set_extra_hosts(monkeypatch, ["atlas.example.com"])
        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)

        client = TestClient(ws.app)
        resp = client.get(
            "/api/status",
            headers={"Host": "atlas.example.com"},
        )
        assert resp.status_code != 400

    def test_http_middleware_still_rejects_unlisted_host(self, monkeypatch):
        """The opt-in must not widen the accept set beyond its entries."""
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        self._set_extra_hosts(monkeypatch, ["atlas.example.com"])
        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)

        client = TestClient(ws.app)
        resp = client.get(
            "/api/status",
            headers={"Host": "evil.example"},
        )
        assert resp.status_code == 400

    def test_websocket_accepts_extra_host_and_origin(self, monkeypatch):
        """The WS Host/Origin guard (_ws_host_origin_reason) must accept a
        configured extra host on both the Host and Origin headers — the
        exact reverse-proxy failure this opt-in exists for (#70059)."""
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        self._set_extra_hosts(monkeypatch, ["atlas.example.com"])
        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "atlas.example.com",
                "Origin": "https://atlas.example.com",
            },
        ):
            pass

    def test_websocket_still_rejects_unlisted_origin(self, monkeypatch):
        """With the opt-in configured, an unlisted Origin keeps failing
        with the same 4403 as before — the DNS-rebinding defence stays."""
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        self._set_extra_hosts(monkeypatch, ["atlas.example.com"])
        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "atlas.example.com",
                    "Origin": "http://evil.example",
                },
            ):
                pass

        assert exc.value.code == 4403
