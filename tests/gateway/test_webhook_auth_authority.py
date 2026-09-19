"""Focused contract tests for explicit webhook signature authority."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from types import SimpleNamespace

import pytest

from gateway.platforms import webhook_auth as auth


class _Headers(dict):
    def get(self, key, default=""):
        for candidate in (key, key.lower(), key.upper()):
            if candidate in self:
                return super().get(candidate, default)
        return default


class _Request:
    def __init__(self, headers=None, route="alerts"):
        self.headers = _Headers(headers or {})
        self.match_info = {"route_name": route}


class _Verifier(auth.WebhookAuthMixin):
    pass


def _verifier():
    return _Verifier()


def test_unknown_signature_mode_fails_closed():
    request = _Request({"X-Hub-Signature-256": "sha256=deadbeef"})
    assert _verifier()._validate_signature(request, b"body", "secret", "wat") is False


def test_github_mode_does_not_accept_gitlab_header():
    request = _Request({"X-Gitlab-Token": "secret"})
    assert _verifier()._validate_signature(request, b"body", "secret", "github") is False


def test_github_mode_accepts_exact_body_hmac():
    body = b'{"ok":true}'
    signature = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    request = _Request({"X-Hub-Signature-256": signature})
    assert _verifier()._validate_signature(request, body, "secret", "github") is True


def test_generic_v2_requires_timestamp_even_when_v1_is_valid():
    body = b"payload"
    v1 = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    request = _Request(
        {
            "X-Webhook-Signature-V2": "present-but-invalid-without-timestamp",
            "X-Webhook-Signature": v1,
        }
    )
    assert _verifier()._validate_signature(request, body, "secret", "generic_v2") is False


def test_generic_v2_binds_timestamp_and_body(monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr(auth.time, "time", lambda: now)
    body = b"payload"
    timestamp = str(now)
    signed = timestamp.encode() + b"." + body
    signature = hmac.new(b"secret", signed, hashlib.sha256).hexdigest()
    request = _Request(
        {
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Signature-V2": signature,
        }
    )
    assert _verifier()._validate_signature(request, body, "secret", "generic_v2") is True
    assert _verifier()._validate_signature(request, body + b"!", "secret", "generic_v2") is False


def test_generic_v2_rejects_expired_timestamp(monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr(auth.time, "time", lambda: now)
    timestamp = str(now - auth.DEFAULT_REPLAY_TOLERANCE_SECONDS - 1)
    body = b"payload"
    signature = hmac.new(
        b"secret", timestamp.encode() + b"." + body, hashlib.sha256
    ).hexdigest()
    request = _Request(
        {
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Signature-V2": signature,
        }
    )
    assert _verifier()._validate_signature(request, body, "secret", "generic_v2") is False


def test_svix_uses_shared_replay_tolerance(monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr(auth.time, "time", lambda: now)
    monkeypatch.setattr(auth, "DEFAULT_REPLAY_TOLERANCE_SECONDS", 10)
    body = b"payload"
    msg_id = "msg_1"
    timestamp = str(now - 11)
    signed = msg_id.encode() + b"." + timestamp.encode() + b"." + body
    signature = base64.b64encode(hmac.new(b"secret", signed, hashlib.sha256).digest()).decode()
    request = _Request(
        {
            "svix-id": msg_id,
            "svix-timestamp": timestamp,
            "svix-signature": f"v1,{signature}",
        }
    )
    # Default arguments are bound at function definition, so pass the shared
    # module value explicitly to pin the public authority contract.
    assert (
        _verifier()._validate_svix_signature(
            body,
            "secret",
            msg_id,
            timestamp,
            f"v1,{signature}",
            tolerance_seconds=auth.DEFAULT_REPLAY_TOLERANCE_SECONDS,
        )
        is False
    )


def test_non_ascii_attacker_header_fails_cleanly():
    request = _Request({"X-Gitlab-Token": "sécret"})
    assert _verifier()._validate_signature(request, b"", "secret", "gitlab") is False
