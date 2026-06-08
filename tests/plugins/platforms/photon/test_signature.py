"""Signature verification tests for the Photon webhook receiver."""
from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from plugins.platforms.photon.adapter import verify_signature


def _sign(secret: str, body: bytes, ts: int) -> str:
    return "v0=" + hmac.new(
        secret.encode(), f"v0:{ts}:".encode() + body, hashlib.sha256,
    ).hexdigest()


def test_accepts_valid_signature() -> None:
    secret = "topsecret-32chars-or-whatever"
    body = b'{"event":"messages"}'
    ts = int(time.time())
    sig = _sign(secret, body, ts)
    assert verify_signature(
        body=body, timestamp_header=str(ts), signature_header=sig,
        signing_secret=secret,
    )


def test_rejects_tampered_body() -> None:
    secret = "s"
    body = b'{"event":"messages"}'
    ts = int(time.time())
    sig = _sign(secret, body, ts)
    assert not verify_signature(
        body=body + b" tamper", timestamp_header=str(ts),
        signature_header=sig, signing_secret=secret,
    )


def test_rejects_wrong_secret() -> None:
    body = b"x"
    ts = int(time.time())
    sig = _sign("right", body, ts)
    assert not verify_signature(
        body=body, timestamp_header=str(ts), signature_header=sig,
        signing_secret="wrong",
    )


def test_rejects_drifted_timestamp() -> None:
    secret = "s"
    body = b"x"
    ts = int(time.time()) - 3600  # 1h old; drift window is 5 min
    sig = _sign(secret, body, ts)
    assert not verify_signature(
        body=body, timestamp_header=str(ts), signature_header=sig,
        signing_secret=secret,
    )


def test_rejects_missing_v0_prefix() -> None:
    secret = "s"
    body = b"x"
    ts = int(time.time())
    raw_hex = hmac.new(
        secret.encode(), f"v0:{ts}:".encode() + body, hashlib.sha256,
    ).hexdigest()
    # Strip the "v0=" prefix — verify_signature must reject.
    assert not verify_signature(
        body=body, timestamp_header=str(ts), signature_header=raw_hex,
        signing_secret=secret,
    )


def test_rejects_empty_inputs() -> None:
    assert not verify_signature(
        body=b"x", timestamp_header="", signature_header="v0=abc",
        signing_secret="s",
    )
    assert not verify_signature(
        body=b"x", timestamp_header="123", signature_header="",
        signing_secret="s",
    )
    assert not verify_signature(
        body=b"x", timestamp_header="123", signature_header="v0=abc",
        signing_secret="",
    )


def test_rejects_non_integer_timestamp() -> None:
    assert not verify_signature(
        body=b"x", timestamp_header="not-an-int",
        signature_header="v0=abc", signing_secret="s",
    )
