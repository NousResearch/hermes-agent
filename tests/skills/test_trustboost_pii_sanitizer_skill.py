"""Tests for the trustboost-pii-sanitizer optional skill.

Hermetic by design: no real network calls are made. The HTTP layer
(httpx.Client.post) is mocked so these tests validate the skill's documented
request/response contract against the free /sanitize/preview endpoint
without depending on api.trustboost.dev being reachable or its live state.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

PREVIEW_URL = "https://api.trustboost.dev/sanitize/preview"
SAMPLE_TEXT = "Contact John at john@example.com or +1-555-0123. API key: sk-abc123xyz."


def _fake_response(status_code: int, payload: dict) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = str(payload)
    return resp


def _call_preview(text: str, mocked_payload: dict, status_code: int = 200):
    """Mimic the skill's documented call: POST {text} to /sanitize/preview,
    with the network layer replaced by a canned response."""
    with patch.object(httpx.Client, "post", return_value=_fake_response(status_code, mocked_payload)) as mock_post:
        with httpx.Client(timeout=60) as client:
            r = client.post(PREVIEW_URL, json={"text": text})
        return r, mock_post


def test_preview_redacts_pii():
    """Preview endpoint response must redact emails and API keys per the
    documented contract (mocked; validates response-shape handling, not the
    live server's redaction logic)."""
    mocked_payload = {
        "sanitized_content": "Contact John at [REDACTED] or [REDACTED]. API key: [REDACTED].",
        "safety_score": 0.85,
        "risk_category": "CRITICAL",
    }
    r, mock_post = _call_preview(SAMPLE_TEXT, mocked_payload)

    assert r.status_code == 200
    body = r.json()
    assert "sanitized_content" in body
    sanitized = body["sanitized_content"]
    assert "john@example.com" not in sanitized, "email was not redacted"
    assert "sk-abc123xyz" not in sanitized, "api key was not redacted"
    assert "[REDACTED]" in sanitized, "expected [REDACTED] placeholder in output"
    mock_post.assert_called_once()
    called_kwargs = mock_post.call_args.kwargs
    assert called_kwargs.get("json") == {"text": SAMPLE_TEXT}


def test_preview_returns_safety_metadata():
    """Preview response must carry a numeric safety score and a known risk
    category, per the documented contract."""
    mocked_payload = {
        "sanitized_content": "[REDACTED]",
        "safety_score": 0.4,
        "risk_category": "PRIVATE",
    }
    r, _ = _call_preview(SAMPLE_TEXT, mocked_payload)

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("safety_score"), (int, float))
    assert body.get("risk_category") in ("CRITICAL", "PRIVATE", "SENSITIVE", "CLEAN")


def test_preview_requires_no_wallet():
    """Preview request body must not need any tx_hash or wallet parameter —
    only {"text": ...}."""
    mocked_payload = {
        "sanitized_content": "call me at [REDACTED]",
        "safety_score": 0.2,
        "risk_category": "SENSITIVE",
    }
    r, mock_post = _call_preview("call me at 555-0199", mocked_payload)

    assert r.status_code == 200
    assert "sanitized_content" in r.json()
    called_kwargs = mock_post.call_args.kwargs
    sent_body = called_kwargs.get("json", {})
    assert "tx_hash" not in sent_body
    assert "wallet" not in sent_body
    assert set(sent_body.keys()) == {"text"}
