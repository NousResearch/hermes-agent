"""Regression tests for A2A is_trusted_peer fail-closed behaviour (issue #126756).

These tests guard against two failure modes:
1. Non-loopback bind with empty allowlist admitting every identity (the original bug).
2. Loopback bind with a bearer token but no allowlist rejecting valid requests
   (the regression introduced by PR #127431 which used localhost_only() as a
   proxy for "bound to loopback").
"""
from __future__ import annotations
import pytest
from plugins.platforms.a2a.security import A2ASecurityContext


def _ctx(**overrides) -> A2ASecurityContext:
    defaults = dict(
        bearer_token="test-token",
        peer_tokens=(),
        trusted_peers=frozenset(),
        allow_all_users=False,
        requested_host="127.0.0.1",
        push_secret="",
    )
    defaults.update(overrides)
    return A2ASecurityContext(**defaults)


class TestIsTrustedPeerFailClosed:
    def test_non_loopback_empty_allowlist_fails_closed(self):
        """Non-loopback bind + empty allowlist must refuse every identity (§3.1)."""
        ctx = _ctx(requested_host="0.0.0.0")
        assert ctx.resolve_bind_host() == "0.0.0.0"
        assert ctx.is_trusted_peer("ip:10.0.0.1") is False
        assert ctx.is_trusted_peer("alice") is False
        assert ctx.is_trusted_peer("") is False

    def test_loopback_with_bearer_token_no_allowlist_stays_open(self):
        """Loopback bind + bearer token + empty allowlist must stay open (regression
        guard: PR #127431 broke this by using localhost_only() as a bind-address proxy)."""
        ctx = _ctx(bearer_token="tok", requested_host="127.0.0.1")
        assert ctx.resolve_bind_host() == "127.0.0.1"
        assert ctx.is_trusted_peer("ip:127.0.0.1") is True
        assert ctx.is_trusted_peer("some-peer") is True

    def test_non_loopback_populated_allowlist_checks_membership(self):
        """Non-loopback bind + populated allowlist: listed identity in, unlisted out."""
        ctx = _ctx(
            requested_host="0.0.0.0",
            trusted_peers=frozenset({"alice", "bob"}),
        )
        assert ctx.is_trusted_peer("alice") is True
        assert ctx.is_trusted_peer("bob") is True
        assert ctx.is_trusted_peer("carol") is False

    def test_allow_all_users_overrides_everything(self):
        """allow_all_users=True opens the gate regardless of allowlist or bind."""
        ctx = _ctx(requested_host="0.0.0.0", allow_all_users=True)
        assert ctx.is_trusted_peer("anyone") is True

    def test_localhost_only_no_tokens_stays_open(self):
        """localhost_only() path (no tokens at all) is unchanged."""
        ctx = _ctx(bearer_token="", peer_tokens=(), requested_host="0.0.0.0")
        assert ctx.localhost_only() is True
        assert ctx.is_trusted_peer("ip:local") is True
