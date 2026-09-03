"""Regression tests for the 2026-08-27 community review on #76661 (guglxni).

Two real defects found by running the branch on a two-node setup:

1. ``_ping_loop`` sent PEER_PING heartbeats via ``ws.send()`` directly,
   bypassing ``send()`` — the only place ``message.sign()`` runs. Every
   ping went out unsigned, the peer's receive loop dropped them at the
   ``verify()`` gate, and auth-enabled (default) links never stayed alive.

2. ``SecretStore`` (HIGH-2 keychain integration) had no callers outside
   its own module — the factory passed ``auth_token`` straight from
   ``config.yaml`` with no keychain fallback, contradicting the PR's
   "macOS Keychain ✅" claim. ``resolve_auth_token`` + the factory wiring
   put the store on the live read path.
"""
from __future__ import annotations

import asyncio

import pytest

from gateway.federation.federation_protocol import FedMessage, MessageType


# ========================================================================
# Finding 1: _ping_loop must sign heartbeats
# ========================================================================


class TestPingLoopSignsHeartbeats:
    """Regression: PEER_PING must carry a valid signature when auth is on.

    Under the old code the ping's ``signature`` was empty, so these
    assertions were impossible to satisfy — the peer dropped every ping
    with "bad signature" (federation_connection.py receive loop).
    """

    @pytest.mark.asyncio
    async def test_ping_is_signed_and_verifiable(self, monkeypatch):
        import gateway.federation.federation_connection as fc

        mgr = fc.FederationConnectionManager(
            device_id="dev-a", auth_token="tok-123",
        )
        sent: list[str] = []

        class FakeWS:
            async def send(self, raw: str) -> None:
                sent.append(raw)

        mgr._ws_connections["dev-b"] = FakeWS()
        mgr._running = True
        monkeypatch.setattr(fc, "PING_INTERVAL", 0)

        task = asyncio.create_task(mgr._ping_loop())
        try:
            for _ in range(500):
                if sent:
                    break
                await asyncio.sleep(0.01)
        finally:
            mgr._running = False
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        assert sent, "ping loop never sent a heartbeat"
        msg = FedMessage.from_json(sent[0])
        assert msg.msg_type == MessageType.PEER_PING.value
        assert msg.sender_id == "dev-a"
        assert msg.target_id == "dev-b"
        # The actual regression: signature must exist AND verify.
        assert msg.signature, "ping went out unsigned — peer would drop it"
        assert msg.verify("tok-123")
        assert not msg.verify("wrong-token")

    @pytest.mark.asyncio
    async def test_ping_unsigned_but_sent_without_auth(self, monkeypatch):
        """No-auth local path must keep working (the sign gate is skipped)."""
        import gateway.federation.federation_connection as fc

        mgr = fc.FederationConnectionManager(device_id="dev-a", auth_token=None)
        sent: list[str] = []

        class FakeWS:
            async def send(self, raw: str) -> None:
                sent.append(raw)

        mgr._ws_connections["dev-b"] = FakeWS()
        mgr._running = True
        monkeypatch.setattr(fc, "PING_INTERVAL", 0)

        task = asyncio.create_task(mgr._ping_loop())
        try:
            for _ in range(500):
                if sent:
                    break
                await asyncio.sleep(0.01)
        finally:
            mgr._running = False
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        assert sent, "ping must still be sent on the no-auth path"
        msg = FedMessage.from_json(sent[0])
        assert msg.signature == ""


# ========================================================================
# Finding 2: SecretStore must be on the live auth-token read path
# ========================================================================


def _make_store(tmp_path, monkeypatch):
    """SecretStore backed ONLY by the encrypted-file backend (no keychain)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from gateway.federation.secret_store import EncryptedFileBackend, SecretStore

    store = SecretStore()
    store._backends = [EncryptedFileBackend()]
    return store


class TestResolveAuthToken:
    """resolve_auth_token: explicit config wins, store is the fallback."""

    def test_explicit_token_wins_without_store_lookup(self, tmp_path, monkeypatch):
        import gateway.federation.secret_store as ss

        def _explode(explicit=None):
            raise AssertionError("store must not be consulted when explicit")

        monkeypatch.setattr(ss, "get_default_store", _explode)
        assert ss.resolve_auth_token("config-token") == "config-token"

    def test_fallback_reads_store_bare_key(self, tmp_path, monkeypatch):
        import gateway.federation.secret_store as ss

        store = _make_store(tmp_path, monkeypatch)
        store.set("auth_token", "keychain-secret-1234567890")
        monkeypatch.setattr(ss, "get_default_store", lambda: store)
        assert ss.resolve_auth_token(None) == "keychain-secret-1234567890"

    def test_fallback_reads_store_dotted_key(self, tmp_path, monkeypatch):
        import gateway.federation.secret_store as ss

        store = _make_store(tmp_path, monkeypatch)
        store.set("federation.cluster_secret", "dotted-secret-1234567890")
        monkeypatch.setattr(ss, "get_default_store", lambda: store)
        assert ss.resolve_auth_token(None) == "dotted-secret-1234567890"

    def test_no_token_anywhere_returns_none(self, tmp_path, monkeypatch):
        import gateway.federation.secret_store as ss

        store = _make_store(tmp_path, monkeypatch)
        monkeypatch.setattr(ss, "get_default_store", lambda: store)
        assert ss.resolve_auth_token(None) is None

    def test_store_unavailable_returns_none(self, monkeypatch):
        import gateway.federation.secret_store as ss

        def _explode():
            raise RuntimeError("no secure secret backend available")

        monkeypatch.setattr(ss, "get_default_store", _explode)
        assert ss.resolve_auth_token(None) is None


class TestAdapterFactoryTokenWiring:
    """create_federation_adapter must consult the secret store.

    Before the fix, building an adapter with auth_token=None produced
    config.auth_token=None even with a token in the store — exactly the
    experiment the reviewer ran on a two-machine setup.
    """

    def test_store_token_reaches_adapter(self, tmp_path, monkeypatch):
        import gateway.federation.secret_store as ss
        from gateway.federation.federation_adapter import create_federation_adapter

        store = _make_store(tmp_path, monkeypatch)
        store.set("cluster_secret", "stored-token-1234567890")
        monkeypatch.setattr(ss, "get_default_store", lambda: store)

        adapter = create_federation_adapter(
            enabled=True, mode="lan", auth_token=None, require_auth=True,
        )
        assert adapter.config.auth_token == "stored-token-1234567890"

    def test_explicit_token_beats_store(self, tmp_path, monkeypatch):
        import gateway.federation.secret_store as ss
        from gateway.federation.federation_adapter import create_federation_adapter

        store = _make_store(tmp_path, monkeypatch)
        store.set("cluster_secret", "stored-token-1234567890")
        monkeypatch.setattr(ss, "get_default_store", lambda: store)

        adapter = create_federation_adapter(
            enabled=True, mode="lan", auth_token="config-token",
            require_auth=True,
        )
        assert adapter.config.auth_token == "config-token"

    def test_no_auth_mode_does_not_probe_store(self, tmp_path, monkeypatch):
        """require_auth=False must not silently adopt a stray keychain token."""
        import gateway.federation.secret_store as ss
        from gateway.federation.federation_adapter import create_federation_adapter

        store = _make_store(tmp_path, monkeypatch)
        store.set("cluster_secret", "stored-token-1234567890")
        monkeypatch.setattr(ss, "get_default_store", lambda: store)

        adapter = create_federation_adapter(
            enabled=True, mode="lan", auth_token=None, require_auth=False,
        )
        assert adapter.config.auth_token is None
