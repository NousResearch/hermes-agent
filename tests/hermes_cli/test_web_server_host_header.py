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

    def test_multi_host_frozenset_back_compat_single_str(self):
        """_is_accepted_host accepts BOTH a bare str (legacy single-bind) and a
        frozenset of bound hosts (dual-stack). Existing str callers are intact."""
        from hermes_cli.web_server import _is_accepted_host

        # str form still works
        assert _is_accepted_host("127.0.0.1", "127.0.0.1")
        assert not _is_accepted_host("evil.example", "127.0.0.1")
        # frozenset form with one member is equivalent
        assert _is_accepted_host("127.0.0.1", frozenset({"127.0.0.1"}))
        assert not _is_accepted_host("evil.example", frozenset({"127.0.0.1"}))

    def test_dual_stack_wildcard_accepts_either_family(self):
        """Bound to both IPv4 + IPv6 wildcard: any Host targeting either is legit."""
        from hermes_cli.web_server import _is_accepted_host

        hosts = frozenset({"0.0.0.0", "::"})
        assert _is_accepted_host("10.0.0.5", hosts)
        assert _is_accepted_host("[::1]", hosts)
        assert _is_accepted_host("[::1]:9119", hosts)

    def test_dual_stack_loopback_accepts_any_loopback(self):
        """Dual loopback bind (127.0.0.1 + ::1) accepts any loopback Host,
        rejects non-loopback."""
        from hermes_cli.web_server import _is_accepted_host

        hosts = frozenset({"127.0.0.1", "::1"})
        # NB: a bare "::1" Host value is rejected by the upstream parser as an
        # ambiguous IPv6 authority (RFC Host headers bracket IPv6 literals);
        # the canonical spelling is "[::1]" / "[::1]:port".
        for target in ("127.0.0.1", "localhost", "[::1]", "[::1]:9119"):
            assert _is_accepted_host(target, hosts)
        assert not _is_accepted_host("10.0.0.5", hosts)
        assert not _is_accepted_host("evil.example", hosts)

    def test_mixed_loopback_public_accepts_either_bound_host(self):
        """Mixed loopback + specific LAN bind: accept if Host matches ANY bound
        host; reject anything else."""
        from hermes_cli.web_server import _is_accepted_host

        hosts = frozenset({"127.0.0.1", "192.168.1.100"})
        assert _is_accepted_host("127.0.0.1", hosts)
        assert _is_accepted_host("192.168.1.100", hosts)
        assert _is_accepted_host("localhost", hosts)  # loopback alias
        assert not _is_accepted_host("10.0.0.5", hosts)  # non-loopback, not bound

    def test_explicit_multi_non_loopback_requires_exact_match(self):
        """Two explicit non-loopback binds: only those exact hosts pass; a
        loopback name must NOT sneak through when nothing loopback was bound."""
        from hermes_cli.web_server import _is_accepted_host

        hosts = frozenset({"10.0.0.5", "10.0.0.6"})
        assert _is_accepted_host("10.0.0.5", hosts)
        assert _is_accepted_host("10.0.0.6:9119", hosts)
        assert not _is_accepted_host("10.0.0.7", hosts)
        assert not _is_accepted_host("localhost", hosts)



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


    def test_trusted_public_host_is_exact_match_only(self):
        """A declared proxy host is accepted without weakening rebinding checks."""
        from hermes_cli.web_server import _is_accepted_host

        trusted = frozenset({"dashboard.example.test"})
        assert _is_accepted_host(
            "dashboard.example.test:9443", "127.0.0.1", trusted
        )
        assert not _is_accepted_host(
            "dashboard.example.test.evil.test", "127.0.0.1", trusted
        )
        assert not _is_accepted_host("evil.test", "127.0.0.1", trusted)

    def test_malformed_host_authorities_fail_closed(self):
        """Ports, IPv6 brackets, and authority syntax must be unambiguous."""
        from hermes_cli.web_server import _is_accepted_host

        trusted = frozenset({"dashboard.example.test"})
        for malformed in (
            "http://dashboard.example.test:9443",
            "dashboard.example.test:",
            "dashboard.example.test:notaport",
            "[::1].evil.test",
            "[::1]:notaport",
            "[localhost]",
        ):
            assert not _is_accepted_host(malformed, "127.0.0.1", trusted)


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
        finally:
            # Clean up so other tests don't inherit the bound_host
            if hasattr(app.state, "bound_host"):
                del app.state.bound_host


    def test_trusted_public_host_request_accepted(self):
        """A loopback backend may accept its declared reverse-proxy host."""
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        app.state.bound_host = "127.0.0.1"
        app.state.trusted_public_hosts = frozenset({"dashboard.example.test"})
        try:
            client = TestClient(app)
            resp = client.get(
                "/api/status",
                headers={"Host": "dashboard.example.test:9443"},
            )
            assert resp.status_code != 400
        finally:
            del app.state.bound_host
            del app.state.trusted_public_hosts

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
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
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
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
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

    def test_trusted_public_websocket_host_and_origin_are_accepted(self, monkeypatch):
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(
            ws.app.state,
            "trusted_public_hosts",
            frozenset({"dashboard.example.test"}),
            raising=False,
        )
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "dashboard.example.test:9443",
                "Origin": "https://dashboard.example.test:9443",
            },
        ):
            pass

    def test_trusted_public_websocket_rejects_cross_site_origin(self, monkeypatch):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(
            ws.app.state,
            "trusted_public_hosts",
            frozenset({"dashboard.example.test"}),
            raising=False,
        )
        monkeypatch.setattr(ws.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "dashboard.example.test:9443",
                    "Origin": "https://evil.test",
                },
            ):
                pass

        assert exc.value.code == 4403


class TestAuthPredicateSingleSource:
    """The Host-header defence (GHSA-ppp5-vxwm-4cf7) must have ONE implementation.

    ``web_server`` re-exports the predicate from ``web_server_boundary`` and
    ``dashboard_auth.request_utils`` imports it through the seam; a local copy in
    ``web_server`` would silently shadow the boundary version, so boundary's own
    internal callers and the request-time middleware would validate against two
    divergent rulesets.
    """

    def test_is_accepted_host_seam_identity(self):
        from hermes_cli import web_server, web_server_boundary

        assert web_server._is_accepted_host is web_server_boundary._is_accepted_host

    def test_mixed_bind_serves_its_own_public_member(self):
        """A loopback + explicit-address bind must accept a Host naming the
        public member: accepting loopback aliases may not short-circuit into
        rejecting our own bound address, while a pure-loopback bind must still
        refuse every external Host (the DNS-rebinding defence unchanged)."""
        from hermes_cli.web_server import _is_accepted_host

        mixed = frozenset({"127.0.0.1", "203.0.113.9"})
        assert _is_accepted_host("203.0.113.9", mixed)
        assert _is_accepted_host("203.0.113.9:9119", mixed)
        assert _is_accepted_host("localhost", mixed)
        assert not _is_accepted_host("198.51.100.7", mixed)

        pure = frozenset({"127.0.0.1"})
        assert not _is_accepted_host("203.0.113.9", pure)
        assert not _is_accepted_host("evil.example", pure)


class TestWsAuthModeMultiHost:
    """``_ws_auth_mode`` logs the active WS auth mode on every connection. Since
    a dual-stack bind can be loopback-only or partly public, the label must be
    decided across the whole bound set — a single-value read would report
    ``loopback`` for a ``127.0.0.1`` + public bind and hide the public member.
    """

    def test_gated_wins_over_bind_membership(self, monkeypatch):
        from hermes_cli import web_server_chat as chat
        from hermes_cli.web_server import app

        monkeypatch.setattr(app.state, "auth_required", True, raising=False)
        monkeypatch.setattr(app.state, "bound_hosts", frozenset({"127.0.0.1"}), raising=False)
        assert chat._ws_auth_mode() == "gated"

    def test_any_public_member_reports_insecure(self, monkeypatch):
        from hermes_cli import web_server_chat as chat
        from hermes_cli.web_server import app

        monkeypatch.setattr(app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(
            app.state, "bound_hosts", frozenset({"127.0.0.1", "203.0.113.9"}), raising=False
        )
        monkeypatch.setattr(app.state, "bound_host", "127.0.0.1", raising=False)
        assert chat._ws_auth_mode() == "insecure"

    def test_all_loopback_members_report_loopback(self, monkeypatch):
        from hermes_cli import web_server_chat as chat
        from hermes_cli.web_server import app

        monkeypatch.setattr(app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(
            app.state, "bound_hosts", frozenset({"127.0.0.1", "::1", "localhost"}), raising=False
        )
        assert chat._ws_auth_mode() == "loopback"

    def test_explicit_none_bound_host_degrades_instead_of_crashing(self, monkeypatch):
        """``app.state.bound_host = None`` is a real shape — upstream's console
        websocket fixture clears it to disable bind assertions. A fallback chain
        that ends in ``getattr(..., "")`` still yields ``None`` there (an existing
        attribute holding None is not a missing one), so the collection branch
        would iterate None; the label must degrade to ``loopback`` instead."""
        from hermes_cli import web_server_chat as chat
        from hermes_cli.web_server import app

        monkeypatch.setattr(app.state, "auth_required", False, raising=False)
        monkeypatch.delattr(app.state, "bound_hosts", raising=False)
        monkeypatch.setattr(app.state, "bound_host", None, raising=False)
        assert chat._ws_auth_mode() == "loopback"

    def test_legacy_single_bound_host_still_decides(self, monkeypatch):
        """Callers and tests that only set the singular ``bound_host`` keep the
        legacy behaviour (fallback chain, no AttributeError)."""
        from hermes_cli import web_server_chat as chat
        from hermes_cli.web_server import app

        monkeypatch.setattr(app.state, "auth_required", False, raising=False)
        monkeypatch.delattr(app.state, "bound_hosts", raising=False)
        monkeypatch.setattr(app.state, "bound_host", "203.0.113.9", raising=False)
        assert chat._ws_auth_mode() == "insecure"

        monkeypatch.setattr(app.state, "bound_host", "127.0.0.1", raising=False)
        assert chat._ws_auth_mode() == "loopback"


class TestSetAwareConsumers:
    """Every consumer of the bind state must decide over the whole bound set.

    The reviewer's original objection was that reading only the first bound
    address leaves the second listener unvalidated. Two request-time consumers
    sit outside the middleware and need the same treatment: the cookie-write Host
    guard in ``dashboard_auth.request_utils`` and the ungated loopback decision
    behind the host-terminal gate.
    """

    def _request(self, host_header: str):
        from starlette.requests import Request

        from hermes_cli.web_server import app
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/x",
            "root_path": "",
            "query_string": b"",
            "raw_path": b"/api/x",
            "scheme": "http",
            "server": ("127.0.0.1", 9119),
            "headers": [(b"host", host_header.encode())],
            "client": ("127.0.0.1", 50000),
            "app": app,
        }
        return Request(scope)

    def test_cookie_host_guard_accepts_the_second_listener(self, monkeypatch):
        """A two-address public bind must accept a request naming the member that
        is not ``bound_host`` — validating only the primary rejects a legitimate
        cookie write against the other listener."""
        from hermes_cli.dashboard_auth import request_utils
        from hermes_cli.web_server import app

        monkeypatch.setattr(app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(
            app.state, "bound_hosts", frozenset({"203.0.113.9", "198.51.100.7"}), raising=False
        )
        monkeypatch.setattr(app.state, "bound_host", "203.0.113.9", raising=False)
        monkeypatch.setattr(app.state, "trusted_public_hosts", frozenset(), raising=False)

        assert request_utils.cookie_origin_is_allowed(self._request("198.51.100.7:9119")) is True
        # and an unrelated Host is still refused
        assert request_utils.cookie_origin_is_allowed(self._request("evil.example")) is False

    def test_host_terminal_gate_requires_every_member_loopback(self, monkeypatch):
        """The ungated host-terminal permission is loopback-only across the whole
        bind: a public member alongside a loopback primary must not read as local."""
        from types import SimpleNamespace

        from hermes_cli import web_host_terminal, web_server, web_server_surface

        monkeypatch.setattr(web_server_surface, "policy",
                            lambda state: SimpleNamespace(host_terminal=True))
        monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(
            web_server.app.state, "bound_hosts", frozenset({"127.0.0.1", "203.0.113.9"}), raising=False
        )
        assert web_host_terminal.request_allowed() is False

        monkeypatch.setattr(
            web_server.app.state, "bound_hosts", frozenset({"127.0.0.1", "::1"}), raising=False
        )
        assert web_host_terminal.request_allowed() is True

    def test_host_terminal_gate_still_honours_legacy_single_bound_host(self, monkeypatch):
        """Callers/tests that only set ``bound_host`` keep today's behaviour."""
        from types import SimpleNamespace

        from hermes_cli import web_host_terminal, web_server, web_server_surface

        monkeypatch.setattr(web_server_surface, "policy",
                            lambda state: SimpleNamespace(host_terminal=True))
        monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
        monkeypatch.delattr(web_server.app.state, "bound_hosts", raising=False)
        monkeypatch.setattr(web_server.app.state, "bound_host", "localhost", raising=False)
        assert web_host_terminal.request_allowed() is True

        monkeypatch.setattr(web_server.app.state, "bound_host", "0.0.0.0", raising=False)
        assert web_host_terminal.request_allowed() is False
