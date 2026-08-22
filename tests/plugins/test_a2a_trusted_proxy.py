"""Trusted reverse-proxy identity resolution for the A2A plugin (Issue #80534).

Behavior contracts (not change-detectors):
  - With trusted_proxies + X-Forwarded-For + peer==proxy: identity is
    ip:<real_client>, never ip:<proxy>.
  - Without trusted_proxies: identity stays ip:<proxy> (the safe default).
  - Peer not in trusted_proxies: X-Forwarded-For is ignored (spoofing guard).
  - Multi-hop X-Forwarded-For: hops are walked right-to-left from the
    socket peer; the first hop not listed in trusted_proxies is the client
    (a caller-supplied allowed leftmost value cannot spoof identity).
  - Startup warnings fire for shared-token + non-loopback + no proxies, and
    for allow-all overriding trusted_peers; neither fires in the safe cases.
  - E2E through the real http.server: a proxied request carries the resolved
    identity into framing/audit.

Tests use a temp HERMES_HOME via the conftest autouse fixture and never write
to ~/.hermes/. Env is manipulated with monkeypatch so it resets per test.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request

import pytest

from plugins.platforms.a2a import protocol, security


def _hdr(xff: str = "") -> dict:
    return {"X-Forwarded-For": xff} if xff else {}


class _DupHeaders:
    """Minimal header object that preserves duplicate X-Forwarded-For fields.

    Mimics ``http.client.HTTPMessage`` / ``email.message.Message`` wire order:
    ``get("X-Forwarded-For")`` returns only the first occurrence (the old
    buggy path), while ``get_all("X-Forwarded-For")`` returns all values in
    wire order.  Also exposes ``_headers`` and ``items()`` for the fallback
    branches in ``security._get_xff_values``.
    """

    def __init__(self, values: list[str]):
        self._values = list(values)

    def get_all(self, name, default=None):  # noqa: D401
        if name.lower() == "x-forwarded-for":
            return list(self._values) if self._values else default
        return default

    def get(self, name, default=None):
        if name.lower() == "x-forwarded-for" and self._values:
            return self._values[0]
        return default

    @property
    def _headers(self):  # type: ignore[no-redef]
        return [("X-Forwarded-For", v) for v in self._values]

    def items(self):
        return [("X-Forwarded-For", v) for v in self._values]


def _email_dup_headers(values: list[str]):
    """Build a real ``email.message.Message`` with duplicate XFF fields."""
    from email.message import Message
    from email.policy import default

    msg = Message(policy=default)
    for v in values:
        msg["X-Forwarded-For"] = v
    return msg


# --------------------------------------------------------------------------
# resolve_client_identity invariants
# --------------------------------------------------------------------------

class TestResolveClientIdentity:
    def test_no_trusted_proxies_uses_socket_peer(self, monkeypatch):
        """Safe default: with no trusted_proxies configured, X-Forwarded-For
        is never consulted — the socket peer is the identity source."""
        monkeypatch.delenv("A2A_TRUSTED_PROXIES", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/nonexistent-hermes-home-no-config")
        assert security.resolve_client_identity(_hdr("203.0.113.9"), "10.0.0.1") == "10.0.0.1"

    def test_trusted_proxy_peer_uses_xff_leftmost(self, monkeypatch):
        """Peer is a trusted proxy + XFF present -> real client (leftmost hop)."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        assert security.resolve_client_identity(_hdr("203.0.113.9"), "10.0.0.1") == "203.0.113.9"

    def test_untrusted_peer_ignores_xff(self, monkeypatch):
        """Spoofing guard: peer NOT in trusted_proxies -> XFF ignored, socket used."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        assert security.resolve_client_identity(_hdr("203.0.113.9"), "198.51.100.7") == "198.51.100.7"

    def test_trusted_proxy_missing_xff_uses_socket(self, monkeypatch):
        """A trusted proxy that forgot to set XFF must not collapse to empty."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        assert security.resolve_client_identity({}, "10.0.0.1") == "10.0.0.1"

    def test_multihop_xff_picks_leftmost(self, monkeypatch):
        """Proxies append hops to the right; walking right-to-left through
        the listed trusted chain lands on the original client.

        X-Forwarded-For: client, proxy1, proxy2 -> identity is `client`,
        with both proxies listed in trusted_proxies (the correct real-world
        configuration for a multi-hop chain).
        """
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1,10.0.0.2")
        xff = "203.0.113.9, 10.0.0.1, 10.0.0.2"
        assert security.resolve_client_identity(_hdr(xff), "10.0.0.2") == "203.0.113.9"

    def test_caller_supplied_allowed_leftmost_hop_rejected(self, monkeypatch):
        """P1 regression: a caller prepends an allow-listed address before the
        proxy appends the real client (``X-Forwarded-For: 10.0.0.1, 203.0.113.9``
        with 10.0.0.1 in trusted_proxies). The right-to-left walk must resolve
        to the real client (203.0.113.9), never the attacker-chosen 10.0.0.1."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        xff = "10.0.0.1, 203.0.113.9"  # attacker-supplied leftmost, proxy-appended real client
        assert security.resolve_client_identity(_hdr(xff), "10.0.0.1") == "203.0.113.9"

    def test_all_trusted_hops_take_leftmost(self, monkeypatch):
        """Walk right-to-left: trusted hops are skipped; when every hop is a
        trusted proxy, the leftmost hop is the identity (whole chain is
        proxies, so the farthest one is the origin)."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1,10.0.0.2")
        xff = "10.0.0.1, 10.0.0.2"
        assert security.resolve_client_identity(_hdr(xff), "10.0.0.2") == "10.0.0.1"

    def test_unlisted_middle_hop_returns_that_hop(self, monkeypatch):
        """Conservative semantics: a hop NOT listed in trusted_proxies stops
        the walk (it is treated as the direct client) instead of being skipped,
        so an attacker cannot extend the trusted chain past its configuration."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.2")
        xff = "203.0.113.9, 10.0.0.1, 10.0.0.2"
        assert security.resolve_client_identity(_hdr(xff), "10.0.0.2") == "10.0.0.1"

    def test_cidr_trusted_proxy_matches(self, monkeypatch):
        """CIDR entries match any address in the range."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.0/8")
        assert security.resolve_client_identity(_hdr("203.0.113.9"), "10.255.1.2") == "203.0.113.9"
        # Outside the CIDR -> XFF ignored.
        assert security.resolve_client_identity(_hdr("203.0.113.9"), "11.0.0.1") == "11.0.0.1"

    def test_xff_with_empty_leading_hops_skips_to_first_nonempty(self, monkeypatch):
        """Garbage/empty leading hops don't yield an empty identity."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        assert security.resolve_client_identity(_hdr(" , 203.0.113.9"), "10.0.0.1") == "203.0.113.9"

    @pytest.mark.parametrize("xff", [
        "not-an-ip",                            # single malformed hop
        "10.0.0.1, not-an-ip",                  # malformed rightmost (direct upstream)
        "10.0.0.1, garbage, 203.0.113.9, not-an-ip",  # malformed rightmost in a longer chain
    ])
    def test_malformed_xff_hop_at_boundary_rejected_falls_back_to_socket(
        self, monkeypatch, xff
    ):
        """P1 hardening: an unvalidated X-Forwarded-For value at the walk
        boundary (the first non-trusted hop from the right) must never become
        the identity. The direct upstream of a trusted proxy is an IP by
        construction; anything else was forged by the caller, so the header
        is rejected and the socket peer is used."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        assert security.resolve_client_identity(_hdr(xff), "10.0.0.1") == "10.0.0.1"

    @pytest.mark.parametrize("xff", [
        "not-an-ip, 203.0.113.9",               # malformed leftmost, valid rightmost
        "10.0.0.1, garbage, 203.0.113.9",       # malformed middle hop
    ])
    def test_malformed_xff_hop_beyond_valid_client_ignored(self, monkeypatch, xff):
        """A malformed hop further LEFT than the first valid untrusted hop is
        never reached by the right-to-left walk — the proxy-appended real
        client wins, so forged garbage cannot displace it either way."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        assert security.resolve_client_identity(_hdr(xff), "10.0.0.1") == "203.0.113.9"

    def test_spoofed_bare_string_hop_cannot_match_allowlist_entry(self, monkeypatch):
        """A bare-string hop like ``ip:alice`` (or any non-IP) must not be
        usable to impersonate a configured trusted-peer identity."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        assert security.resolve_client_identity(_hdr("ip:alice"), "10.0.0.1") == "10.0.0.1"

    def test_end_to_end_identity_through_authenticate(self, monkeypatch):
        """authenticate() builds the final identity string from the resolved IP.
        With a shared token + trusted proxy, two distinct real clients get two
        distinct identities (the core invariant the fix restores)."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        ip_a = security.resolve_client_identity(_hdr("203.0.113.9"), "10.0.0.1")
        ip_b = security.resolve_client_identity(_hdr("198.51.100.7"), "10.0.0.1")
        assert ip_a == "203.0.113.9"
        assert ip_b == "198.51.100.7"
        assert security.authenticate("Bearer shared-tok", ip_a) == "ip:203.0.113.9"
        assert security.authenticate("Bearer shared-tok", ip_b) == "ip:198.51.100.7"
        # Without the fix both would be ip:10.0.0.1 — assert they differ.
        assert security.authenticate("Bearer shared-tok", ip_a) != security.authenticate("Bearer shared-tok", ip_b)

    def test_untrusted_peer_collapses_to_proxy_identity_unchanged(self, monkeypatch):
        """The bug this fixes: without trusted_proxies, both peers share ip:<proxy>."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.delenv("A2A_TRUSTED_PROXIES", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        ip_a = security.resolve_client_identity(_hdr("203.0.113.9"), "10.0.0.1")
        ip_b = security.resolve_client_identity(_hdr("198.51.100.7"), "10.0.0.1")
        assert ip_a == ip_b == "10.0.0.1"  # unchanged behavior — the documented risk

    def test_duplicate_xff_fields_wire_order_canonicalized(self, monkeypatch):
        """P1 follow-up (#80779): duplicate X-Forwarded-For fields must be
        canonicalized in wire order, not truncated to the first occurrence.

        Attacker injects ``X-Forwarded-For: 10.0.0.1`` (allow-listed) as the
        first field; the trusted proxy appends ``X-Forwarded-For:
        203.0.113.9`` (real client) as a second field.  ``headers.get()``
        returns only the first field, so the old code resolved to the
        attacker value.  After canonicalizing all field values the walk
        yields the real client.
        """
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1,203.0.113.1")
        dup = _DupHeaders(["10.0.0.1", "203.0.113.9"])
        # socket peer is the trusted proxy that appended the second field
        result = security.resolve_client_identity(dup, "203.0.113.1")
        assert result == "203.0.113.9"
        assert result != "10.0.0.1"

    def test_duplicate_xff_fields_via_email_message_canonicalized(self, monkeypatch):
        """Same bypass, but through a real stdlib Message with duplicate fields.

        ``email.message.Message`` preserves duplicates in ``_headers`` and
        ``get_all``; ``get`` returns the first.  The fix must not rely solely
        on ``get``.
        """
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        msg = _email_dup_headers(["10.0.0.1", "203.0.113.9"])
        # Socket peer is the trusted proxy; attacker allow-listed leftmost must not win.
        assert security.resolve_client_identity(msg, "10.0.0.1") == "203.0.113.9"
        # Also verify that a single combined field still works (no duplicate).
        from email.message import Message
        from email.policy import default

        single = Message(policy=default)
        single["X-Forwarded-For"] = "10.0.0.1, 203.0.113.9"
        assert security.resolve_client_identity(single, "10.0.0.1") == "203.0.113.9"

    def test_duplicate_xff_fields_with_three_fields_wire_order(self, monkeypatch):
        """Wire order is preserved across >2 duplicate fields and mixed commas."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1,10.0.0.2")
        # Attacker field 1, proxy1 field 2 with comma, proxy2 is socket peer
        dup = _DupHeaders(["10.0.0.1", "203.0.113.9, 10.0.0.1"])
        # hops wire order: [10.0.0.1, 203.0.113.9, 10.0.0.1]
        # walk from right: 10.0.0.1 trusted -> skip, 203.0.113.9 not trusted -> real client
        assert security.resolve_client_identity(dup, "10.0.0.2") == "203.0.113.9"

    def test_duplicate_xff_fields_untrusted_peer_still_ignored(self, monkeypatch):
        """Even with duplicate fields, an untrusted socket peer must not trust XFF."""
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        dup = _DupHeaders(["10.0.0.1", "203.0.113.9"])
        assert security.resolve_client_identity(dup, "198.51.100.7") == "198.51.100.7"


class TestGetTrustedProxiesConfig:
    def test_env_list_parsed(self, monkeypatch):
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1, 10.0.0.0/8, , 203.0.113.5")
        assert security.get_trusted_proxies() == {"10.0.0.1", "10.0.0.0/8", "203.0.113.5"}

    def test_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("A2A_TRUSTED_PROXIES", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/nonexistent-hermes-home-no-config")
        assert security.get_trusted_proxies() == set()

    def test_config_yaml_primary_over_env_when_env_unset(self, monkeypatch, tmp_path):
        """config.yaml a2a.trusted_proxies is the primary source (env is fallback),
        mirroring get_trusted_peers."""
        monkeypatch.delenv("A2A_TRUSTED_PROXIES", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("a2a:\n  trusted_proxies:\n    - 10.0.0.0/8\n    - 203.0.113.5\n")
        assert security.get_trusted_proxies() == {"10.0.0.0/8", "203.0.113.5"}


# --------------------------------------------------------------------------
# Startup warnings
# --------------------------------------------------------------------------

class TestStartupWarnings:
    def test_shared_token_nonloopback_no_proxies_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.delenv("A2A_TRUSTED_PROXIES", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/nonexistent-hermes-home-no-config")
        monkeypatch.setenv("A2A_HOST", "0.0.0.0")
        monkeypatch.delenv("A2A_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("A2A_TRUSTED_PEERS", raising=False)
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.a2a.security"):
            security.warn_on_insecure_identity_config()
        assert any("trusted_proxies" in r.message and "collapsing" in r.message for r in caplog.records)

    def test_no_warn_when_trusted_proxies_set(self, monkeypatch, caplog):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        monkeypatch.setenv("A2A_HOST", "0.0.0.0")
        monkeypatch.delenv("A2A_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("A2A_TRUSTED_PEERS", raising=False)
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.a2a.security"):
            security.warn_on_insecure_identity_config()
        assert not any("collapsing" in r.message for r in caplog.records)

    def test_no_warn_when_loopback_host(self, monkeypatch, caplog):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.delenv("A2A_TRUSTED_PROXIES", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/nonexistent-hermes-home-no-config")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        monkeypatch.delenv("A2A_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("A2A_TRUSTED_PEERS", raising=False)
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.a2a.security"):
            security.warn_on_insecure_identity_config()
        assert not any("collapsing" in r.message for r in caplog.records)

    def test_no_warn_with_peer_tokens_only(self, monkeypatch, caplog):
        """Per-peer tokens are spoof-proof; the shared-token warning doesn't apply."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.setenv("A2A_PEER_TOKENS", "alice:tok-a")
        monkeypatch.delenv("A2A_TRUSTED_PROXIES", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/nonexistent-hermes-home-no-config")
        monkeypatch.setenv("A2A_HOST", "0.0.0.0")
        monkeypatch.delenv("A2A_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("A2A_TRUSTED_PEERS", raising=False)
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.a2a.security"):
            security.warn_on_insecure_identity_config()
        assert not any("collapsing" in r.message for r in caplog.records)

    def test_allow_all_overrides_trusted_peers_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.setenv("A2A_ALLOW_ALL_USERS", "true")
        monkeypatch.setenv("A2A_TRUSTED_PEERS", "alice")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        # Keep the shared-token warning quiet so we can isolate this one.
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.a2a.security"):
            security.warn_on_insecure_identity_config()
        assert any("A2A_ALLOW_ALL_USERS" in r.message and "allow-list" in r.message for r in caplog.records)

    def test_allow_all_alone_does_not_warn(self, monkeypatch, caplog):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.setenv("A2A_ALLOW_ALL_USERS", "true")
        monkeypatch.delenv("A2A_TRUSTED_PEERS", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "10.0.0.1")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.a2a.security"):
            security.warn_on_insecure_identity_config()
        assert not any("allow-list" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# E2E: live http.server routes XFF identity into framing/audit
# --------------------------------------------------------------------------

def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_live_adapter(monkeypatch, reply_fn=None):
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig
    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    adapter = A2AAdapter(PlatformConfig(enabled=True))

    async def fake_handle_message(event):
        reply = "ECHO: " + event.text if reply_fn is None else reply_fn(event)
        if reply is not None:
            await adapter.send(event.source.chat_id, reply, metadata={"notify": True})

    adapter.handle_message = fake_handle_message  # type: ignore
    adapter._message_handler = object()
    return adapter, f"http://127.0.0.1:{port}"


def _post_json(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _send_body(text, ctx=""):
    return {
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {"message": protocol.text_message(protocol.ROLE_USER, text, context_id=ctx)},
    }


@pytest.mark.integration
class TestTrustedProxyE2E:
    def test_xff_identity_used_for_framing_through_live_server(self, monkeypatch):
        """A request arriving from a trusted-proxy peer with X-Forwarded-For
        set must carry the real client (not the proxy) into the agent's privacy
        frame — the full do_POST -> authenticate -> framing path."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.setenv("A2A_TRUSTED_PROXIES", "127.0.0.1")  # the test client is the "proxy"
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")

        seen = {}

        def reply_fn(event):
            seen["user"] = event.source.user_id
            seen["text"] = event.text
            return "ok"

        adapter, base = _make_live_adapter(monkeypatch, reply_fn=reply_fn)

        async def run():
            assert await adapter.connect() is True
            resp = await asyncio.to_thread(
                _post_json, base + "/", _send_body("hi", ctx="ctx-xff"),
                {"Authorization": "Bearer shared-tok", "X-Forwarded-For": "203.0.113.9"})
            assert resp["result"]["status"]["state"] == protocol.STATE_COMPLETED
            await adapter.disconnect()

        asyncio.run(run())
        # The agent saw the real client, not 127.0.0.1.
        assert seen["user"] == "ip:203.0.113.9"
        assert "'ip:203.0.113.9'" in seen["text"]
        assert "127.0.0.1" not in seen["text"]

    def test_no_trusted_proxies_e2e_uses_socket_identity(self, monkeypatch):
        """Without trusted_proxies, an XFF header is ignored end-to-end — the
        socket peer (the test client) is the identity. Safe default preserved."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.delenv("A2A_TRUSTED_PROXIES", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/nonexistent-hermes-home-no-config")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")

        seen = {}

        def reply_fn(event):
            seen["user"] = event.source.user_id
            return "ok"

        adapter, base = _make_live_adapter(monkeypatch, reply_fn=reply_fn)

        async def run():
            assert await adapter.connect() is True
            resp = await asyncio.to_thread(
                _post_json, base + "/", _send_body("hi", ctx="ctx-noxff"),
                {"Authorization": "Bearer shared-tok", "X-Forwarded-For": "203.0.113.9"})
            assert resp["result"]["status"]["state"] == protocol.STATE_COMPLETED
            await adapter.disconnect()

        asyncio.run(run())
        # XFF ignored; identity is the socket peer (loopback).
        assert seen["user"].startswith("ip:127.0.0.1") or seen["user"] == "ip:::1" or seen["user"].startswith("ip:127.")
        assert seen["user"] != "ip:203.0.113.9"
