"""Tests for Gemini free-tier detection and blocking."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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

    def test_free_tier_via_rpd_header_flash(self):
        # gemini-2.5-flash free tier: 250 RPD
        resp = _mock_response(200, {"x-ratelimit-limit-requests-per-day": "250"}, "{}")
        assert _run_probe(resp) == "free"

    def test_free_tier_via_rpd_header_pro(self):
        # gemini-2.5-pro free tier: 100 RPD
        resp = _mock_response(200, {"x-ratelimit-limit-requests-per-day": "100"}, "{}")
        assert _run_probe(resp) == "free"

    def test_free_tier_via_rpd_header_flash_lite(self):
        # flash-lite free tier: 1000 RPD (our upper bound)
        resp = _mock_response(200, {"x-ratelimit-limit-requests-per-day": "1000"}, "{}")
        assert _run_probe(resp) == "free"

    def test_paid_tier_via_rpd_header(self):
        # Tier 1 starts at 1500+ RPD
        resp = _mock_response(200, {"x-ratelimit-limit-requests-per-day": "1500"}, "{}")
        assert _run_probe(resp) == "paid"

    def test_free_tier_via_429_body(self):
        body = (
            '{"error":{"code":429,"message":"Quota exceeded for metric: '
            'generativelanguage.googleapis.com/generate_content_free_tier_requests, '
            'limit: 20"}}'
        )
        resp = _mock_response(429, {}, body)
        assert _run_probe(resp) == "free"

    def test_paid_429_has_no_free_tier_marker(self):
        body = '{"error":{"code":429,"message":"rate limited"}}'
        resp = _mock_response(429, {}, body)
        assert _run_probe(resp) == "paid"

    def test_successful_200_without_rpd_header_is_paid(self):
        resp = _mock_response(200, {}, '{"candidates":[]}')
        assert _run_probe(resp) == "paid"

    def test_401_returns_unknown(self):
        resp = _mock_response(401, {}, '{"error":{"code":401}}')
        assert _run_probe(resp) == "unknown"

    def test_404_returns_unknown(self):
        resp = _mock_response(404, {}, '{"error":{"code":404}}')
        assert _run_probe(resp) == "unknown"

    def test_network_error_returns_unknown(self):
        with patch(
            "agent.gemini_native_adapter.httpx.Client",
            side_effect=Exception("dns failure"),
        ):
            assert probe_gemini_tier("fake-key") == "unknown"

    def test_empty_key_returns_unknown(self):
        assert probe_gemini_tier("") == "unknown"
        assert probe_gemini_tier("   ") == "unknown"
        assert probe_gemini_tier(None) == "unknown"  # type: ignore[arg-type]

    def test_malformed_rpd_header_falls_through(self):
        # Non-integer header value shouldn't crash; 200 with no usable header -> paid.
        resp = _mock_response(200, {"x-ratelimit-limit-requests-per-day": "abc"}, "{}")
        assert _run_probe(resp) == "paid"

    def test_openai_compat_suffix_stripped(self):
        """Base URLs ending in /openai get normalized to the native endpoint."""
        resp = _mock_response(200, {"x-ratelimit-limit-requests-per-day": "1500"}, "{}")
        with patch("agent.gemini_native_adapter.httpx.Client") as MC:
            inst = MagicMock()
            inst.post.return_value = resp
            MC.return_value.__enter__.return_value = inst
            probe_gemini_tier(
                "fake",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            )
            # Verify the post URL does NOT contain /openai
            called_url = inst.post.call_args[0][0]
            assert "/openai/" not in called_url
            assert called_url.endswith(":generateContent")


class TestIsFreeTierQuotaError:
    def test_detects_free_tier_marker(self):
        assert is_free_tier_quota_error(
            "Quota exceeded for metric: generate_content_free_tier_requests"
        )

    def test_case_insensitive(self):
        assert is_free_tier_quota_error("QUOTA: FREE_TIER_REQUESTS")

    def test_no_free_tier_marker(self):
        assert not is_free_tier_quota_error("rate limited")

    def test_empty_string(self):
        assert not is_free_tier_quota_error("")

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

    def test_non_429_has_no_billing_url(self):
        body = '{"error":{"code":400,"message":"bad request","status":"INVALID_ARGUMENT"}}'
        err = gemini_http_error(self._FakeResp(400, body))
        assert "aistudio.google.com/apikey" not in str(err)

    def test_401_has_no_billing_url(self):
        body = '{"error":{"code":401,"message":"API key invalid","status":"UNAUTHENTICATED"}}'
        err = gemini_http_error(self._FakeResp(401, body))
        assert "aistudio.google.com/apikey" not in str(err)
