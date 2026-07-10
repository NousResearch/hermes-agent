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

    def test_allowed_host_accepted_on_non_loopback_bind(self):
        """An allowlisted host is accepted even though it isn't the bound host."""
        from hermes_cli.web_server import _is_accepted_host

        allowed = frozenset({"box.tailnet.ts.net"})
        assert _is_accepted_host("box.tailnet.ts.net", "100.64.0.1", allowed)
        assert _is_accepted_host("box.tailnet.ts.net:9119", "100.64.0.1", allowed)
        # The bound host itself still works.
        assert _is_accepted_host("100.64.0.1", "100.64.0.1", allowed)

    def test_allowed_host_accepted_on_loopback_bind(self):
        """The `tailscale serve` case: loopback bind reached via MagicDNS name."""
        from hermes_cli.web_server import _is_accepted_host

        allowed = frozenset({"box.tailnet.ts.net"})
        assert _is_accepted_host("box.tailnet.ts.net", "127.0.0.1", allowed)
        # Loopback names still work alongside the allowlist.
        assert _is_accepted_host("localhost:9119", "127.0.0.1", allowed)

    def test_non_listed_host_still_rejected(self):
        """A host that is neither the bound host nor allowlisted is rejected."""
        from hermes_cli.web_server import _is_accepted_host

        allowed = frozenset({"box.tailnet.ts.net"})
        assert not _is_accepted_host("evil.example", "127.0.0.1", allowed)
        assert not _is_accepted_host("evil.example", "100.64.0.1", allowed)

    def test_empty_allowed_set_is_unchanged_behavior(self):
        """The default empty allowlist keeps the original accept/reject rules."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("localhost", "127.0.0.1", frozenset())
        assert not _is_accepted_host("box.tailnet.ts.net", "127.0.0.1", frozenset())
        # Same as calling without the argument at all.
        assert _is_accepted_host("localhost", "127.0.0.1")
        assert not _is_accepted_host("box.tailnet.ts.net", "127.0.0.1")


class TestNormalizeAllowedHost:
    """The entry normalizer must reduce every accepted form to host_only."""

    def test_port_case_bracket_fqdn_dot_normalization(self):
        from hermes_cli.web_server import _normalize_allowed_host

        cases = {
            "  Box.Tailnet.TS.net  ": "box.tailnet.ts.net",
            "box.tailnet.ts.net:9119": "box.tailnet.ts.net",
            "box.tailnet.ts.net.": "box.tailnet.ts.net",  # FQDN trailing dot
            "https://box.tailnet.ts.net/path": "box.tailnet.ts.net",  # URL form
            "https://box.tailnet.ts.net:8443": "box.tailnet.ts.net",
            "[::1]": "::1",  # IPv6 brackets stripped
            "[fe80::1]:9119": "fe80::1",
            "": "",  # empty dropped by caller
            "   ": "",
        }
        for raw, expected in cases.items():
            assert _normalize_allowed_host(raw) == expected, raw

    def test_normalized_entry_matches_is_accepted_host(self):
        """A normalized entry compares equal to _is_accepted_host's host_only."""
        from hermes_cli.web_server import _is_accepted_host, _normalize_allowed_host

        allowed = frozenset({_normalize_allowed_host("https://Box.Tailnet.TS.net:8443/")})
        assert _is_accepted_host("box.tailnet.ts.net:9119", "127.0.0.1", allowed)


class TestResolveAllowedHosts:
    """The union resolver merges CLI + env + config and normalizes."""

    def test_union_of_cli_env_config(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setenv(
            "HERMES_DASHBOARD_ALLOWED_HOSTS", "Env-Host.example:9119, ,dup.example"
        )
        monkeypatch.setattr(
            ws, "load_config", lambda: {"dashboard": {"allowed_hosts": ["Cfg-Host.example."]}}
        )
        result = ws._resolve_allowed_hosts(["Cli-Host.example", "dup.example"])
        assert result == frozenset(
            {"cli-host.example", "env-host.example", "cfg-host.example", "dup.example"}
        )

    def test_empty_sources_give_empty_set(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_HOSTS", raising=False)
        monkeypatch.setattr(ws, "load_config", lambda: {"dashboard": {}})
        assert ws._resolve_allowed_hosts(None) == frozenset()
        assert ws._resolve_allowed_hosts([]) == frozenset()

    def test_non_empty_forces_auth_gate(self, monkeypatch):
        """The auth-forcing rule: a non-empty allowlist ORs into auth_required
        even on a loopback bind (should_require_auth stays pure)."""
        import hermes_cli.web_server as ws

        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_HOSTS", raising=False)
        monkeypatch.setattr(ws, "load_config", lambda: {"dashboard": {}})

        allowed = ws._resolve_allowed_hosts(["box.tailnet.ts.net"])
        # Loopback bind: should_require_auth is False, but the allowlist forces it.
        assert ws.should_require_auth("127.0.0.1") is False
        assert (ws.should_require_auth("127.0.0.1") or bool(allowed)) is True
        # Empty allowlist leaves the loopback bind ungated.
        empty = ws._resolve_allowed_hosts([])
        assert (ws.should_require_auth("127.0.0.1") or bool(empty)) is False


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

    def test_allowed_host_request_accepted(self):
        """With app.state.allowed_hosts set, a MagicDNS Host header passes
        while a non-listed host still 400s."""
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        app.state.bound_host = "127.0.0.1"
        app.state.allowed_hosts = frozenset({"box.tailnet.ts.net"})
        try:
            client = TestClient(app)
            ok = client.get(
                "/api/status", headers={"Host": "box.tailnet.ts.net"}
            )
            assert ok.status_code != 400 or (
                "Invalid Host header" not in ok.json().get("detail", "")
            )
            bad = client.get("/api/status", headers={"Host": "evil.example"})
            assert bad.status_code == 400
            assert "Invalid Host header" in bad.json()["detail"]
        finally:
            if hasattr(app.state, "bound_host"):
                del app.state.bound_host
            if hasattr(app.state, "allowed_hosts"):
                del app.state.allowed_hosts


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

    def test_rebinding_websocket_origin_is_rejected(self, monkeypatch):
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
                    "Host": "localhost:9119",
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

    def test_allowed_host_websocket_is_accepted(self, monkeypatch):
        """An allowlisted MagicDNS host is accepted for the WS upgrade."""
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(
            ws.app.state, "allowed_hosts",
            frozenset({"box.tailnet.ts.net"}), raising=False,
        )
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "box.tailnet.ts.net",
                "Origin": "http://box.tailnet.ts.net",
            },
        ):
            pass

    def test_non_listed_websocket_host_still_rejected(self, monkeypatch):
        """A non-allowlisted host is still rejected with 4403."""
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(
            ws.app.state, "allowed_hosts",
            frozenset({"box.tailnet.ts.net"}), raising=False,
        )
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
