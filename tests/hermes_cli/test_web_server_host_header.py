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

    def test_loopback_bind_accepts_loopback_names(self):
        from hermes_cli.web_server import _is_accepted_host

        for bound in ("127.0.0.1", "localhost", "::1"):
            for host_header in (
                "127.0.0.1", "127.0.0.1:9119",
                "localhost", "localhost:9119",
                "[::1]", "[::1]:9119",
            ):
                assert _is_accepted_host(host_header, bound), (
                    f"bound={bound} must accept host={host_header}"
                )

    def test_loopback_bind_rejects_attacker_hostnames(self):
        """The core rebinding defence: attacker-controlled hosts that
        TTL-flip to 127.0.0.1 must be rejected."""
        from hermes_cli.web_server import _is_accepted_host

        for bound in ("127.0.0.1", "localhost"):
            for attacker in (
                "evil.example",
                "evil.example:9119",
                "rebind.attacker.test:80",
                "localhost.attacker.test",  # subdomain trick
                "127.0.0.1.evil.test",  # lookalike IP prefix
                "",  # missing Host
            ):
                assert not _is_accepted_host(attacker, bound), (
                    f"bound={bound} must reject attacker host={attacker!r}"
                )

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

    def test_case_insensitive_comparison(self):
        """Host headers are case-insensitive per RFC — accept variations."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("LOCALHOST", "127.0.0.1")
        assert _is_accepted_host("LocalHost:9119", "127.0.0.1")


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

    def test_legit_loopback_request_accepted(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        app.state.bound_host = "127.0.0.1"
        try:
            client = TestClient(app)
            # /api/status is in _PUBLIC_API_PATHS — passes auth — so the
            # only thing that can reject is the host header middleware
            resp = client.get(
                "/api/status",
                headers={"Host": "localhost:9119"},
            )
            # Either 200 (endpoint served) or some other non-400 —
            # just not the host-rejection 400
            assert resp.status_code != 400 or (
                "Invalid Host header" not in resp.json().get("detail", "")
            )
        finally:
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


class _FakeWS:
    """Minimal WebSocket stand-in exposing just the .headers mapping the
    Host/Origin guard reads. Avoids spinning up the full ASGI app."""

    def __init__(self, headers: dict[str, str]):
        # Header lookups in the guard are lowercase; mirror that.
        self.headers = {k.lower(): v for k, v in headers.items()}


class TestWebSocketHostOriginGuard:
    """Tests for GHSA-4pqm-j46f-795x — DNS-rebinding bypass via WebSocket
    endpoints. FastAPI HTTP middleware does NOT run on WS upgrades, so the
    Host (and, when present, Origin) header must be validated inside the WS
    handlers. _ws_host_origin_is_allowed is that guard."""

    def setup_method(self):
        from hermes_cli.web_server import app

        app.state.bound_host = "127.0.0.1"

    def teardown_method(self):
        from hermes_cli.web_server import app

        if hasattr(app.state, "bound_host"):
            del app.state.bound_host

    def test_unbound_host_skips_guard(self):
        from hermes_cli.web_server import app, _ws_host_origin_is_allowed

        if hasattr(app.state, "bound_host"):
            del app.state.bound_host
        # No bound_host → nothing to compare against → allow (HTTP layer
        # behaves the same).
        assert _ws_host_origin_is_allowed(_FakeWS({"host": "evil.example"}))

    def test_loopback_host_allowed(self):
        from hermes_cli.web_server import _ws_host_origin_is_allowed

        assert _ws_host_origin_is_allowed(_FakeWS({"host": "localhost:9119"}))
        assert _ws_host_origin_is_allowed(_FakeWS({"host": "127.0.0.1:9119"}))

    def test_rebinding_host_rejected(self):
        """An attacker hostname that TTL-flips to 127.0.0.1 passes the peer-IP
        check but must be rejected by the Host guard on the WS upgrade."""
        from hermes_cli.web_server import _ws_host_origin_is_allowed

        assert not _ws_host_origin_is_allowed(_FakeWS({"host": "evil.example"}))
        assert not _ws_host_origin_is_allowed(
            _FakeWS({"host": "rebind.attacker.test:9119"})
        )

    def test_cross_origin_rejected(self):
        """Browser sends Origin on the WS handshake; a mismatched web Origin
        (the cross-site rebinding driver) must be rejected even if Host is
        spoofed to loopback."""
        from hermes_cli.web_server import _ws_host_origin_is_allowed

        assert not _ws_host_origin_is_allowed(
            _FakeWS({"host": "127.0.0.1:9119", "origin": "https://evil.example"})
        )

    def test_matching_origin_allowed(self):
        from hermes_cli.web_server import _ws_host_origin_is_allowed

        assert _ws_host_origin_is_allowed(
            _FakeWS({"host": "127.0.0.1:9119", "origin": "http://127.0.0.1:9119"})
        )

    def test_non_web_origin_allowed(self):
        """Packaged Electron / native clients send file://, null, or app://
        origins. The token credential is the auth boundary there — don't
        reject on Origin scheme."""
        from hermes_cli.web_server import _ws_host_origin_is_allowed

        for origin in ("file://", "null", "app://hermes"):
            assert _ws_host_origin_is_allowed(
                _FakeWS({"host": "127.0.0.1:9119", "origin": origin})
            )

    def test_missing_origin_allowed(self):
        """Non-browser WS clients omit Origin entirely — allowed (Host still
        validated)."""
        from hermes_cli.web_server import _ws_host_origin_is_allowed

        assert _ws_host_origin_is_allowed(_FakeWS({"host": "127.0.0.1:9119"}))
