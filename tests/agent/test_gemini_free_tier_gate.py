"""Tests for Gemini free-tier detection and blocking."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


from agent.gemini_native_adapter import (
    gemini_http_error,
    is_free_tier_quota_error,
    probe_gemini_tier,
)


def _mock_response(status: int, headers: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.text = text
    return resp


def _run_probe(resp: MagicMock) -> str:
    with patch("agent.gemini_native_adapter.httpx.Client") as MC:
        inst = MagicMock()
        inst.post.return_value = resp
        MC.return_value.__enter__.return_value = inst
        return probe_gemini_tier("fake-key")


class TestProbeGeminiTier:
    """Verify the tier probe classifies keys correctly."""





    def test_free_tier_via_429_body(self):
        body = (
            '{"error":{"code":429,"message":"Quota exceeded for metric: '
            'generativelanguage.googleapis.com/generate_content_free_tier_requests, '
            'limit: 20"}}'
        )
        resp = _mock_response(429, {}, body)
        assert _run_probe(resp) == "free"


    def test_successful_200_without_rpd_header_is_paid(self):
        resp = _mock_response(200, {}, '{"candidates":[]}')
        assert _run_probe(resp) == "paid"








class TestIsFreeTierQuotaError:
    def test_detects_free_tier_marker(self):
        assert is_free_tier_quota_error(
            "Quota exceeded for metric: generate_content_free_tier_requests"
        )


    def test_no_free_tier_marker(self):
        assert not is_free_tier_quota_error("rate limited")


    def test_none(self):
        assert not is_free_tier_quota_error(None)  # type: ignore[arg-type]


class TestGeminiHttpErrorFreeTierGuidance:
    """gemini_http_error should append free-tier guidance for free-tier 429s."""

    class _FakeResp:
        def __init__(self, status: int, text: str):
            self.status_code = status
            self.headers: dict = {}
            self.text = text

    def test_free_tier_429_appends_guidance(self):
        body = (
            '{"error":{"code":429,"message":"Quota exceeded for metric: '
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            'limit: 20","status":"RESOURCE_EXHAUSTED"}}'
        )
        err = gemini_http_error(self._FakeResp(429, body))
        msg = str(err)
        assert "free tier" in msg.lower()
        assert "aistudio.google.com/apikey" in msg

    def test_paid_429_has_no_billing_url(self):
        body = '{"error":{"code":429,"message":"Rate limited","status":"RESOURCE_EXHAUSTED"}}'
        err = gemini_http_error(self._FakeResp(429, body))
        assert "aistudio.google.com/apikey" not in str(err)


