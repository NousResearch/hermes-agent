"""Pocket HMAC and replay-window regressions; no listener or live delivery.

Exercise the imported adapter with case-insensitive HTTP headers and a fixed
clock. Secrets and payloads are synthetic, not captured production traffic.
"""

import base64
import hashlib
import hmac
from types import SimpleNamespace

import pytest
from multidict import CIMultiDict

from gateway.config import PlatformConfig
from gateway.platforms import webhook
from gateway.platforms.webhook import WebhookAdapter


NOW = 1_800_000_000
SECRET = "pocket-unit-test-secret"
BODY = '{ "event": "transcript.edited", "text": "café" }\n'.encode()
SIGNATURE = "X-HeyPocket-Signature"
TIMESTAMP = "X-HeyPocket-Timestamp"


def _digest(content, secret=SECRET):
    return hmac.new(secret.encode(), content, hashlib.sha256).hexdigest()


def _signed_headers(timestamp=None, body=BODY):
    timestamp = str(NOW * 1000) if timestamp is None else timestamp
    return {
        TIMESTAMP: timestamp,
        SIGNATURE: _digest(timestamp.encode() + b"." + body),
    }


def _request(headers):
    return SimpleNamespace(
        headers=CIMultiDict(headers), match_info={"route_name": "pocket-test"}
    )


@pytest.fixture
def adapter(monkeypatch):
    instance = WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {}}))
    monkeypatch.setattr(webhook, "time", SimpleNamespace(time=lambda: NOW))
    return instance


def test_valid_raw_body_signature_accepts(adapter):
    assert adapter._validate_signature(_request(_signed_headers()), BODY, SECRET) is True


@pytest.mark.parametrize("header_case", [str.lower, str.upper])
def test_http_header_names_are_case_insensitive(adapter, header_case):
    headers = {header_case(k): v for k, v in _signed_headers().items()}
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is True


@pytest.mark.parametrize("body", [b"", b"{}", b"\x00\xff\xfe"])
def test_signature_uses_raw_bytes_without_json_decoding(adapter, body):
    headers = _signed_headers(body=body)
    assert adapter._validate_signature(_request(headers), body, SECRET) is True


@pytest.mark.parametrize("body", [BODY + b" ", BODY.replace(b"edited", b"created"), BODY.rstrip()])
def test_modified_body_rejects(adapter, body):
    assert adapter._validate_signature(_request(_signed_headers()), body, SECRET) is False


def test_wrong_secret_rejects(adapter):
    assert adapter._validate_signature(_request(_signed_headers()), BODY, "wrong-secret") is False


@pytest.mark.parametrize("signature", ["", "bad", "0" * 64, "sha256=" + "0" * 64, "café"])
def test_missing_or_malformed_signature_rejects(adapter, signature):
    headers = _signed_headers()
    headers[SIGNATURE] = signature
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is False


def test_absent_signature_rejects(adapter):
    assert adapter._validate_signature(_request({TIMESTAMP: str(NOW * 1000)}), BODY, SECRET) is False


@pytest.mark.parametrize("timestamp", [None, "", "not-a-number", "1800000000000.0"])
def test_missing_or_malformed_timestamp_rejects(adapter, timestamp):
    headers = _signed_headers()
    if timestamp is None:
        del headers[TIMESTAMP]
    else:
        headers[TIMESTAMP] = timestamp
    if timestamp is not None:
        headers[TIMESTAMP.lower()] = headers.pop(TIMESTAMP)
    headers[SIGNATURE.lower()] = headers.pop(SIGNATURE)
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is False


@pytest.mark.parametrize(
    "offset_ms, accepted",
    [(-300_001, False), (-300_000, True), (-299_999, True),
     (0, True), (299_999, True), (300_000, True), (300_001, False)],
)
def test_signed_replay_window_boundaries(adapter, offset_ms, accepted):
    headers = _signed_headers(str(NOW * 1000 + offset_ms))
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is accepted


def test_seconds_timestamp_rejects_even_with_matching_signature(adapter):
    headers = _signed_headers(str(NOW))
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is False


def test_timestamp_is_bound_to_signature(adapter):
    headers = _signed_headers()
    headers[TIMESTAMP] = str(NOW * 1000 + 1)
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is False


@pytest.mark.parametrize("timestamp", [None, "", "malformed", str(NOW * 1000 - 300_001)])
def test_invalid_pocket_timestamp_cannot_fall_back_to_valid_v1(adapter, timestamp):
    headers = _signed_headers()
    headers["X-Webhook-Signature"] = _digest(BODY)
    if timestamp is None:
        del headers[TIMESTAMP]
    else:
        headers[TIMESTAMP] = timestamp
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is False


@pytest.mark.parametrize("timestamp", ["9" * 400, "9" * 1000])
def test_unrepresentable_or_excessive_timestamp_rejects_without_overflow(adapter, timestamp):
    headers = _signed_headers(timestamp)
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is False


@pytest.mark.parametrize("clock", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_clock_rejects_pocket_timestamp(adapter, monkeypatch, clock):
    monkeypatch.setattr(webhook, "time", SimpleNamespace(time=lambda: clock))
    assert adapter._validate_signature(_request(_signed_headers()), BODY, SECRET) is False


@pytest.mark.parametrize("signature", ["bad", "café", "0" * 64])
def test_invalid_pocket_signature_cannot_fall_back_to_valid_v1(adapter, signature):
    headers = _signed_headers()
    headers[SIGNATURE] = signature
    headers["X-Webhook-Signature"] = _digest(BODY)
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is False


@pytest.mark.parametrize("signature", [None, ""])
def test_incomplete_pocket_headers_cannot_downgrade_to_valid_v1(adapter, signature):
    """Presence of Pocket headers must commit to Pocket authentication."""
    headers = {TIMESTAMP: str(NOW * 1000), "X-Webhook-Signature": _digest(BODY)}
    if signature is not None:
        headers[SIGNATURE] = signature
    assert adapter._validate_signature(_request(headers), BODY, SECRET) is False


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Hub-Signature-256": "sha256=" + _digest(BODY)},
        {"X-Gitlab-Token": SECRET},
        {"linear-signature": _digest(BODY)},
        {"X-Webhook-Signature": _digest(BODY)},
        {"X-Webhook-Timestamp": str(NOW),
         "X-Webhook-Signature-V2": _digest(str(NOW).encode() + b"." + BODY)},
        {"svix-id": "msg_test", "svix-timestamp": str(NOW),
         "svix-signature": "v1," + base64.b64encode(hmac.new(
             SECRET.encode(), b"msg_test." + str(NOW).encode() + b"." + BODY,
             hashlib.sha256).digest()).decode()},
    ],
    ids=["github", "gitlab", "linear", "generic-v1", "generic-v2", "svix"],
)
@pytest.mark.parametrize("secret, accepted", [(SECRET, True), ("wrong-secret", False)])
def test_other_provider_schemes_without_pocket_headers(adapter, headers, secret, accepted):
    assert adapter._validate_signature(_request(headers), BODY, secret) is accepted
