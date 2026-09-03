import hashlib
import hmac
import time
import sqlite3

import pytest

from plugins.builder_adapter.auth import HMACAuthenticator, PrincipalKey
from plugins.builder_adapter.canonical import signed_material
from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.store import DispatchStore


def signer(secret, method, path, timestamp, nonce, digest):
    return hmac.new(
        secret,
        signed_material(method, path, timestamp, nonce, digest),
        hashlib.sha256,
    ).hexdigest()


def test_hmac_identity_and_durable_replay_rejection(tmp_path):
    secret = b"test-secret-not-production".ljust(32, b"x")
    store = DispatchStore(tmp_path / "journal.db")
    auth = HMACAuthenticator(
        [PrincipalKey("orchestrator-mcp", "key-1", secret, 501, 20)],
        store,
        clock=lambda: 1000,
    )
    timestamp = "1000"
    nonce = "nonce-1234567890abcdef"
    digest = "a" * 64
    signature = signer(secret, "POST", "/v1/dispatches", timestamp, nonce, digest)
    assert (
        auth.verify(
            method="POST",
            path="/v1/dispatches",
            timestamp=timestamp,
            nonce=nonce,
            request_sha256=digest,
            key_id="key-1",
            signature=signature,
            peer_uid=501,
            peer_gid=20,
        )
        == "orchestrator-mcp"
    )
    restarted = HMACAuthenticator(
        [PrincipalKey("orchestrator-mcp", "key-1", secret, 501, 20)],
        DispatchStore(tmp_path / "journal.db"),
        clock=lambda: 1000,
    )
    with pytest.raises(AdapterError) as replay:
        restarted.verify(
            method="POST",
            path="/v1/dispatches",
            timestamp=timestamp,
            nonce=nonce,
            request_sha256=digest,
            key_id="key-1",
            signature=signature,
            peer_uid=501,
            peer_gid=20,
        )
    assert replay.value.code == "REPLAY_REJECTED"


@pytest.mark.parametrize(
    ("timestamp", "uid", "signature", "code"),
    [
        ("1", 501, "bad", "REPLAY_REJECTED"),
        ("1000", 999, "bad", "AUTHORIZATION_FAILED"),
        ("1000", 501, "bad", "AUTHENTICATION_FAILED"),
    ],
)
def test_auth_failures_precede_side_effects(tmp_path, timestamp, uid, signature, code):
    auth = HMACAuthenticator(
        [PrincipalKey("principal", "key", b"test-secret".ljust(32, b"x"), 501)],
        DispatchStore(tmp_path / "journal.db"),
        clock=lambda: 1000,
    )
    with pytest.raises(AdapterError) as raised:
        auth.verify(
            method="POST",
            path="/v1/dispatches",
            timestamp=timestamp,
            nonce="nonce-1234567890abcdef",
            request_sha256="a" * 64,
            key_id="key",
            signature=signature,
            peer_uid=uid,
            peer_gid=20,
        )
    assert raised.value.code == code


def test_expired_nonce_cleanup_is_bounded_to_expired_rows(tmp_path):
    store = DispatchStore(tmp_path / "journal.db")
    store.consume_nonce("key", "old-nonce-123456", 10, now=1)
    store.consume_nonce("key", "new-nonce-123456", 30, now=20)
    with sqlite3.connect(store.path) as conn:
        rows = conn.execute("SELECT nonce FROM nonces ORDER BY nonce").fetchall()
    assert rows == [("new-nonce-123456",)]


def test_revoked_key_fails_even_with_a_valid_signature(tmp_path):
    secret = b"revoked-test-secret".ljust(32, b"x")
    auth = HMACAuthenticator(
        [PrincipalKey("principal", "revoked", secret, 501, active=False)],
        DispatchStore(tmp_path / "journal.db"),
        clock=lambda: 1000,
    )
    digest = "a" * 64
    signature = signer(
        secret,
        "POST",
        "/v1/dispatches",
        "1000",
        "nonce-1234567890abcdef",
        digest,
    )
    with pytest.raises(AdapterError) as raised:
        auth.verify(
            method="POST",
            path="/v1/dispatches",
            timestamp="1000",
            nonce="nonce-1234567890abcdef",
            request_sha256=digest,
            key_id="revoked",
            signature=signature,
            peer_uid=501,
            peer_gid=20,
        )
    assert raised.value.code == "AUTHENTICATION_FAILED"


def test_duplicate_or_short_hmac_keys_fail_at_startup(tmp_path):
    store = DispatchStore(tmp_path / "journal.db")
    with pytest.raises(AdapterError):
        HMACAuthenticator(
            [
                PrincipalKey("a", "duplicate", b"a" * 32, 501),
                PrincipalKey("b", "duplicate", b"b" * 32, 501),
            ],
            store,
        )
    with pytest.raises(AdapterError):
        HMACAuthenticator(
            [PrincipalKey("a", "short", b"too-short", 501)],
            store,
        )
