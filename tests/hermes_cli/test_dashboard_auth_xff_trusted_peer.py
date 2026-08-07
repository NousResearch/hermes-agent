"""X-Forwarded-For is only trusted from a loopback peer, and only its LAST hop.

``_client_ip`` is the sole bucket key for the ``/auth/password-login``
brute-force throttle (``routes._password_rate_limited``) and the ``ip=`` field
on every auth audit event. It previously took ``fwd.split(",")[0]`` — the
first element, which is entirely client-supplied — so rotating the header
handed out a fresh 10-per-60s bucket per request and forged the audit trail.

Both copies of the helper (``routes`` and ``token_auth``) must:
  * ignore X-Forwarded-For entirely when the connection peer is NOT loopback;
  * take the LAST hop when the peer IS loopback (the single local reverse
    proxy appends it, so it is the one element a client cannot forge).
"""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient
from starlette.requests import Request

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers
from hermes_cli.dashboard_auth import middleware, token_auth
from hermes_cli.dashboard_auth.routes import (
    _PW_RATE_MAX_ATTEMPTS,
    _client_ip,
    _reset_password_rate_limit,
)


# THREE call sites carry an identical helper; every case below must hold for
# each of them, so they are parametrized rather than duplicated.
#
# middleware._client_ip was missing from this list while this PR patched it,
# so the one copy that feeds the audit log's ip= field on every authenticated
# request was changed with nothing exercising it. A helper tuple that silently
# lags the call sites is worse than no parametrization, because the coverage it
# implies is the reason nobody looks again.
_HELPERS = (_client_ip, token_auth._client_ip, middleware._client_ip)


def _req(peer, xff=None) -> Request:
    """A real Starlette Request with the given connection peer and XFF."""
    headers = []
    if xff is not None:
        headers.append((b"x-forwarded-for", xff.encode()))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": "/auth/password-login",
            "raw_path": b"/auth/password-login",
            "query_string": b"",
            "root_path": "",
            "scheme": "https",
            "headers": headers,
            "client": peer,
            "server": ("testserver", 443),
        }
    )


# ---------------------------------------------------------------------------
# Helper contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("helper", _HELPERS)
class TestClientIpTrustedPeer:
    def test_loopback_peer_takes_last_hop_of_spoofed_chain(self, helper):
        # The client sent two forged hops; the proxy appended the real one.
        # Only the last element may be believed.
        req = _req(("127.0.0.1", 51234), "192.0.2.1, 192.0.2.9, 198.51.100.7")
        assert helper(req) == "198.51.100.7"

    def test_loopback_peer_single_hop(self, helper):
        req = _req(("127.0.0.1", 51234), "198.51.100.7")
        assert helper(req) == "198.51.100.7"

    def test_loopback_peer_strips_whitespace(self, helper):
        req = _req(("127.0.0.1", 51234), "192.0.2.1,   198.51.100.7   ")
        assert helper(req) == "198.51.100.7"

    def test_loopback_peer_without_xff_falls_back_to_peer(self, helper):
        assert helper(_req(("127.0.0.1", 51234))) == "127.0.0.1"

    def test_empty_xff_from_loopback_falls_back_to_peer(self, helper):
        assert helper(_req(("127.0.0.1", 51234), "")) == "127.0.0.1"

    def test_trailing_separator_does_not_yield_empty_bucket_key(self, helper):
        # A proxy emitting "10.0.0.7," must not resolve to "" — that lands
        # every such caller in the throttle's shared "_unknown_" bucket.
        req = _req(("127.0.0.1", 51234), "10.0.0.7,")
        assert helper(req) == "10.0.0.7"

    def test_ipv6_loopback_is_trusted(self, helper):
        req = _req(("::1", 51234), "192.0.2.1, 198.51.100.7")
        assert helper(req) == "198.51.100.7"

    def test_ipv4_mapped_loopback_is_trusted(self, helper):
        # Under a dual-stack bind (--host ::) the IPv4 loopback proxy peer
        # arrives IPv4-mapped. A literal ("127.0.0.1", "::1") test rejects
        # it, discarding XFF and collapsing every client into one bucket.
        req = _req(("::ffff:127.0.0.1", 51234), "192.0.2.1, 198.51.100.7")
        assert helper(req) == "198.51.100.7"

    def test_non_loopback_peer_ignores_xff_entirely(self, helper):
        # Direct (non-proxied) connection: the header is pure client input.
        req = _req(("203.0.113.9", 44321), "192.0.2.1, 198.51.100.7")
        assert helper(req) == "203.0.113.9"

    def test_private_lan_peer_is_not_loopback(self, helper):
        # Only the local proxy is trusted — not "anything that looks internal".
        req = _req(("192.168.1.50", 44321), "192.0.2.1")
        assert helper(req) == "192.168.1.50"

    def test_no_client_returns_empty(self, helper):
        assert helper(_req(None, "192.0.2.1")) == ""

    def test_no_client_ignores_xff(self, helper):
        # No discernible peer → the throttle's shared "_unknown_" bucket,
        # never an attacker-chosen one.
        assert helper(_req(None, "192.0.2.1, 198.51.100.7")) == ""


# ---------------------------------------------------------------------------
# End-to-end: the login throttle can no longer be reset by header rotation
# ---------------------------------------------------------------------------


@pytest.fixture
def login_client():
    clear_providers()
    _reset_password_rate_limit()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    yield
    clear_providers()
    _reset_password_rate_limit()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _client(peer):
    return TestClient(
        web_server.app, base_url="https://fly-app.fly.dev", client=peer
    )


def _attempt(client, xff):
    # No provider is registered, so every allowed attempt 404s; the throttle
    # is evaluated before provider lookup, so 429 still marks bucket
    # exhaustion. Keeps the test about bucketing, not about credentials.
    return client.post(
        "/auth/password-login",
        json={"provider": "nope", "username": "admin", "password": "x"},
        headers={"X-Forwarded-For": xff},
    )


class TestThrottleBucketing:
    def test_rotating_spoofed_first_hop_shares_one_bucket(self, login_client):
        # Proxied request: attacker varies the forgeable prefix on every
        # attempt but the proxy-appended last hop is constant, so all
        # attempts must land in ONE bucket and exhaust it.
        client = _client(("127.0.0.1", 51234))
        codes = [
            _attempt(client, f"10.0.0.{i}, 198.51.100.5").status_code
            for i in range(_PW_RATE_MAX_ATTEMPTS + 2)
        ]
        assert codes[:_PW_RATE_MAX_ATTEMPTS] == [404] * _PW_RATE_MAX_ATTEMPTS
        assert codes[_PW_RATE_MAX_ATTEMPTS:] == [429, 429]

    def test_distinct_last_hops_get_distinct_buckets(self, login_client):
        # The flip side: two genuinely different clients behind the proxy
        # must not share a budget.
        client = _client(("127.0.0.1", 51234))
        for _ in range(_PW_RATE_MAX_ATTEMPTS):
            _attempt(client, "198.51.100.5")
        assert _attempt(client, "198.51.100.5").status_code == 429
        assert _attempt(client, "198.51.100.6").status_code == 404

    def test_non_loopback_peer_rotation_does_not_reset_bucket(self, login_client):
        # Unproxied request: XFF is ignored outright, so a fully rotated
        # header still buckets on the real connection peer.
        client = _client(("203.0.113.9", 44321))
        codes = [
            _attempt(client, f"10.0.0.{i}, 172.16.0.{i}").status_code
            for i in range(_PW_RATE_MAX_ATTEMPTS + 2)
        ]
        assert codes[:_PW_RATE_MAX_ATTEMPTS] == [404] * _PW_RATE_MAX_ATTEMPTS
        assert codes[_PW_RATE_MAX_ATTEMPTS:] == [429, 429]
