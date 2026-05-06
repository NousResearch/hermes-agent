"""Tests for the dashboard auth policy added in the Tailscale Serve change.

Unit-tests the pure decision function ``_dashboard_auth_decision`` across the
matrix of (peer, identity header, host, allowlist) and exercises the HTTP
middleware end-to-end with a TestClient. WebSocket-side coverage uses the
same helper through ``_ws_auth_ok``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# _dashboard_auth_decision — unit tests on the pure policy function
# ---------------------------------------------------------------------------


@pytest.fixture
def loopback_bind():
    """Force ``_is_public_bind`` False by setting a loopback bound_host."""
    from hermes_cli.web_server import app

    app.state.bound_host = "127.0.0.1"
    try:
        yield
    finally:
        if hasattr(app.state, "bound_host"):
            del app.state.bound_host


@pytest.fixture
def public_bind():
    """Force ``_is_public_bind`` True — operator used --insecure."""
    from hermes_cli.web_server import app

    app.state.bound_host = "0.0.0.0"
    try:
        yield
    finally:
        if hasattr(app.state, "bound_host"):
            del app.state.bound_host


def _decide(*, client_host, ts_login="", host_header="", token=False, allowlist=()):
    """Helper: call ``_dashboard_auth_decision`` with allowlist patched."""
    from hermes_cli import web_server

    with patch.object(
        web_server,
        "_dashboard_tailscale_allowlist",
        return_value=frozenset(allowlist),
    ):
        return web_server._dashboard_auth_decision(
            client_host=client_host,
            ts_login=ts_login,
            host_header=host_header,
            token_matches=lambda: token,
        )


class TestPublicBind:
    """--insecure / 0.0.0.0 bind: token-only, identity ignored."""

    def test_token_match_allows(self, public_bind):
        assert _decide(client_host="203.0.113.5", token=True) is True

    def test_token_mismatch_denies(self, public_bind):
        assert _decide(client_host="203.0.113.5", token=False) is False

    def test_identity_header_is_ignored(self, public_bind):
        # Public bind cannot trust Tailscale-User-Login (no tailscaled
        # in front), so allowlist hit must NOT bypass token check.
        assert _decide(
            client_host="203.0.113.5",
            ts_login="alice@example.com",
            allowlist={"alice@example.com"},
            token=False,
        ) is False

    def test_loopback_peer_under_public_bind_still_token_only(self, public_bind):
        assert _decide(client_host="127.0.0.1", token=True) is True
        assert _decide(client_host="127.0.0.1", token=False) is False


class TestLoopbackPeerNoIdentity:
    """Standard local-browser case: peer=127.0.0.1, no Tailscale header."""

    def test_token_match_allows(self, loopback_bind):
        assert _decide(client_host="127.0.0.1", token=True) is True

    def test_token_mismatch_denies(self, loopback_bind):
        assert _decide(client_host="127.0.0.1", token=False) is False

    def test_testclient_peer_treated_as_loopback(self, loopback_bind):
        assert _decide(client_host="testclient", token=True) is True

    def test_ipv6_loopback_treated_as_loopback(self, loopback_bind):
        assert _decide(client_host="::1", token=True) is True


class TestTailscaleIdentityFromLoopback:
    """tailscale serve proxies tailnet traffic to the local app over
    loopback and injects Tailscale-User-Login. The allowlist gates it."""

    def test_allowlisted_login_passes_without_token(self, loopback_bind):
        assert _decide(
            client_host="127.0.0.1",
            ts_login="alice@example.com",
            allowlist={"alice@example.com"},
            token=False,
        ) is True

    def test_login_not_in_allowlist_denied(self, loopback_bind):
        assert _decide(
            client_host="127.0.0.1",
            ts_login="mallory@example.com",
            allowlist={"alice@example.com"},
            token=True,  # token would have passed; identity branch wins
        ) is False

    def test_empty_allowlist_fails_closed(self, loopback_bind):
        # Even a valid-looking login is denied when the allowlist is empty.
        # No fall-through to token check on the identity path.
        assert _decide(
            client_host="127.0.0.1",
            ts_login="alice@example.com",
            allowlist=(),
            token=True,
        ) is False

    def test_allowlist_match_is_case_sensitive(self, loopback_bind):
        # Documenting current behaviour: the allowlist is an exact
        # string match. Tailscale logins are normalised by tailscaled,
        # so this is acceptable but worth pinning.
        assert _decide(
            client_host="127.0.0.1",
            ts_login="Alice@Example.com",
            allowlist={"alice@example.com"},
            token=False,
        ) is False


class TestTailscaleCgnatPeer:
    """Direct tailnet bind (no tailscale serve): peer is in 100.64/10."""

    def test_cgnat_with_allowlisted_identity_passes(self, loopback_bind):
        assert _decide(
            client_host="100.112.255.106",
            ts_login="alice@example.com",
            allowlist={"alice@example.com"},
            token=False,
        ) is True

    def test_cgnat_with_unknown_identity_denied(self, loopback_bind):
        assert _decide(
            client_host="100.112.255.106",
            ts_login="mallory@example.com",
            allowlist={"alice@example.com"},
            token=True,
        ) is False

    def test_cgnat_no_identity_with_tailnet_host_uses_token(self, loopback_bind):
        # WebSocket / Serve paths that strip identity headers but keep
        # the MagicDNS Host can still authenticate by token.
        assert _decide(
            client_host="100.112.255.106",
            ts_login="",
            host_header="machine.tailnet.ts.net",
            token=True,
        ) is True
        assert _decide(
            client_host="100.112.255.106",
            ts_login="",
            host_header="machine.tailnet.ts.net",
            token=False,
        ) is False

    def test_cgnat_no_identity_no_tailnet_host_denied(self, loopback_bind):
        # Trusted peer but neither identity nor tailnet host → deny,
        # regardless of token.
        assert _decide(
            client_host="100.112.255.106",
            ts_login="",
            host_header="evil.example",
            token=True,
        ) is False

    def test_cgnat_boundary_addresses(self, loopback_bind):
        # 100.64.0.0/10 spans 100.64.0.0–100.127.255.255.
        from hermes_cli.web_server import _is_tailscale_peer

        assert _is_tailscale_peer("100.64.0.0")
        assert _is_tailscale_peer("100.127.255.255")
        # Just outside the block.
        assert not _is_tailscale_peer("100.63.255.255")
        assert not _is_tailscale_peer("100.128.0.0")
        # Garbage.
        assert not _is_tailscale_peer("")
        assert not _is_tailscale_peer(None)
        assert not _is_tailscale_peer("not-an-ip")


class TestUntrustedPeers:
    """Anything that isn't loopback, CGNAT, or a public-bind operator opt-in."""

    def test_random_public_peer_always_denied(self, loopback_bind):
        for token in (True, False):
            assert _decide(client_host="203.0.113.5", token=token) is False

    def test_random_public_peer_with_identity_denied(self, loopback_bind):
        # Even with an allowlisted login, we don't trust the header from
        # an untrusted peer.
        assert _decide(
            client_host="203.0.113.5",
            ts_login="alice@example.com",
            allowlist={"alice@example.com"},
            token=True,
        ) is False

    def test_empty_client_host_denied(self, loopback_bind):
        assert _decide(client_host="", token=True) is False


# ---------------------------------------------------------------------------
# _is_loopback_http_request — peer + host must both be loopback
# ---------------------------------------------------------------------------


class TestIsLoopbackHttpRequest:
    @staticmethod
    def _req(peer, host_header):
        return SimpleNamespace(
            client=SimpleNamespace(host=peer) if peer is not None else None,
            headers={"host": host_header} if host_header is not None else {},
        )

    def test_localhost_peer_and_host_is_loopback(self):
        from hermes_cli.web_server import _is_loopback_http_request

        assert _is_loopback_http_request(self._req("127.0.0.1", "localhost"))
        assert _is_loopback_http_request(self._req("127.0.0.1", "127.0.0.1:9119"))
        assert _is_loopback_http_request(self._req("::1", "[::1]:9119"))

    def test_loopback_peer_with_tailnet_host_is_not_loopback(self):
        # tailscale serve preserves loopback peer but rewrites Host —
        # this is exactly the case the auth policy must NOT short-circuit.
        from hermes_cli.web_server import _is_loopback_http_request

        assert not _is_loopback_http_request(
            self._req("127.0.0.1", "machine.tailnet.ts.net")
        )

    def test_cgnat_peer_is_not_loopback(self):
        from hermes_cli.web_server import _is_loopback_http_request

        assert not _is_loopback_http_request(
            self._req("100.112.255.106", "localhost")
        )

    def test_testclient_with_no_host_is_loopback(self):
        # Pin the test-artifact path: TestClient may not set a Host.
        from hermes_cli.web_server import _is_loopback_http_request

        req = SimpleNamespace(client=SimpleNamespace(host="testclient"), headers={})
        assert _is_loopback_http_request(req)


# ---------------------------------------------------------------------------
# auth_middleware — end-to-end behaviour via FastAPI TestClient
# ---------------------------------------------------------------------------


class TestAuthMiddlewareEndToEnd:
    """TestClient peers report as 'testclient' (loopback alias). Override
    Host so requests pass the host-header middleware too."""

    def _client(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        return TestClient(app)

    def test_public_api_path_does_not_require_auth(self):
        client = self._client()
        resp = client.get("/api/status", headers={"Host": "localhost"})
        # /api/status is public — should not be 401.
        assert resp.status_code != 401

    def test_loopback_html_load_without_token_allowed(self):
        # Local browser hits /chat with no token: prior behaviour, must
        # keep working so the SPA can bootstrap its embedded token.
        client = self._client()
        resp = client.get("/", headers={"Host": "localhost"})
        # Whatever the route returns, it must not be 401 from the auth
        # middleware (the route itself may 404 etc., that's fine).
        assert resp.status_code != 401

    def test_loopback_html_with_tailscale_header_requires_auth(self):
        # A loopback peer that *also* sends a Tailscale-User-Login can no
        # longer ride the loopback carve-out — must satisfy the policy.
        client = self._client()
        resp = client.get(
            "/",
            headers={
                "Host": "localhost",
                "Tailscale-User-Login": "alice@example.com",
            },
        )
        # No allowlist configured → fail-closed.
        assert resp.status_code == 401

    def test_loopback_html_with_allowlisted_identity_allowed(self):
        from hermes_cli import web_server

        client = self._client()
        with patch.object(
            web_server,
            "_dashboard_tailscale_allowlist",
            return_value=frozenset({"alice@example.com"}),
        ):
            resp = client.get(
                "/",
                headers={
                    "Host": "localhost",
                    "Tailscale-User-Login": "alice@example.com",
                },
            )
        assert resp.status_code != 401

    def test_protected_api_without_token_rejected(self):
        # /api/config is not in _PUBLIC_API_PATHS — must require token.
        client = self._client()
        resp = client.get("/api/config", headers={"Host": "localhost"})
        assert resp.status_code == 401

    def test_protected_api_with_token_allowed(self):
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

        client = self._client()
        resp = client.get(
            "/api/config",
            headers={
                "Host": "localhost",
                _SESSION_HEADER_NAME: _SESSION_TOKEN,
            },
        )
        assert resp.status_code != 401

    def test_protected_api_with_bearer_token_allowed(self):
        # Backward-compat path for older dashboard bundles.
        from hermes_cli.web_server import _SESSION_TOKEN

        client = self._client()
        resp = client.get(
            "/api/config",
            headers={
                "Host": "localhost",
                "Authorization": f"Bearer {_SESSION_TOKEN}",
            },
        )
        assert resp.status_code != 401

    def test_unauthorized_html_response_for_browser_paths(self):
        # Browser-facing 401s must return HTML (matches what the page
        # renders); JSON 401s are reserved for API clients.
        client = self._client()
        resp = client.get(
            "/",
            headers={
                "Host": "localhost",
                "Tailscale-User-Login": "mallory@example.com",
            },
        )
        assert resp.status_code == 401
        assert "text/html" in resp.headers.get("content-type", "").lower()

    def test_unauthorized_json_response_for_api_paths(self):
        client = self._client()
        resp = client.get("/api/config", headers={"Host": "localhost"})
        assert resp.status_code == 401
        assert "application/json" in resp.headers.get("content-type", "").lower()
