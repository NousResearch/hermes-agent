"""Tests for agent.azure_content_safety — stdlib HTTP shim."""
from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from agent import azure_content_safety as acs


# --------------------------------------------------------------------------- helpers


def _resp(body: dict, status: int = 200, headers: dict | None = None):
    """Build a mock urlopen response context manager."""
    raw = json.dumps(body).encode("utf-8")
    m = MagicMock()
    m.read.return_value = raw
    m.status = status
    m.getcode.return_value = status
    m.headers = headers or {}
    cm = MagicMock()
    cm.__enter__.return_value = m
    cm.__exit__.return_value = False
    return cm


# --------------------------------------------------------------------------- analyze_text


class TestAnalyzeText:
    def test_request_shape_endpoint_and_path(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text(
                "hello",
                endpoint="https://cs.example.com",
                key="K",
            )
            req = up.call_args[0][0]
            assert req.full_url.startswith(
                "https://cs.example.com/contentsafety/text:analyze?api-version="
            )

    def test_request_strips_trailing_slash_on_endpoint(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text("hi", endpoint="https://cs.example.com/", key="K")
            req = up.call_args[0][0]
            assert "//contentsafety" not in req.full_url

    def test_uses_post_method(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")
            req = up.call_args[0][0]
            assert req.get_method() == "POST"

    def test_sends_subscription_key_header(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text("hi", endpoint="https://cs.example.com", key="SECRET")
            req = up.call_args[0][0]
            assert req.headers["Ocp-apim-subscription-key"] == "SECRET"

    def test_sends_json_content_type(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")
            req = up.call_args[0][0]
            assert "application/json" in req.headers["Content-type"]

    def test_body_contains_text(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text("hello world", endpoint="https://cs.example.com", key="K")
            req = up.call_args[0][0]
            body = json.loads(req.data.decode("utf-8"))
            assert body["text"] == "hello world"

    def test_default_categories(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")
            body = json.loads(up.call_args[0][0].data.decode("utf-8"))
            assert "Hate" in body["categories"]
            assert "Violence" in body["categories"]

    def test_custom_categories(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text(
                "hi",
                endpoint="https://cs.example.com",
                key="K",
                categories=["Hate"],
            )
            body = json.loads(up.call_args[0][0].data.decode("utf-8"))
            assert body["categories"] == ["Hate"]

    def test_returns_parsed_json(self):
        payload = {"categoriesAnalysis": [{"category": "Hate", "severity": 2}]}
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp(payload)
            out = acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")
            assert out == payload

    def test_passes_timeout(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text(
                "hi",
                endpoint="https://cs.example.com",
                key="K",
                timeout=7,
            )
            assert up.call_args.kwargs["timeout"] == 7

    def test_429_retry_with_retry_after(self):
        # First call: 429 with Retry-After. Second call: success.
        err = urllib.error.HTTPError(
            "https://cs.example.com",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            io.BytesIO(b'{"error":{"code":"TooManyRequests"}}'),
        )
        ok = _resp({"categoriesAnalysis": []})
        with patch("urllib.request.urlopen") as up, patch.object(acs.time, "sleep"):
            up.side_effect = [err, ok]
            out = acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")
            assert out == {"categoriesAnalysis": []}
            assert up.call_count == 2

    def test_429_raises_after_one_retry(self):
        err = urllib.error.HTTPError(
            "https://cs.example.com",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            io.BytesIO(b'{}'),
        )
        with patch("urllib.request.urlopen") as up, patch.object(acs.time, "sleep"):
            up.side_effect = [err, err]
            with pytest.raises(acs.ContentSafetyError) as exc_info:
                acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")
            assert "rate" in str(exc_info.value).lower() or "429" in str(exc_info.value)
            assert up.call_count == 2

    def test_http_error_redacts_key(self):
        err = urllib.error.HTTPError(
            "https://cs.example.com?key=SUPER_SECRET",
            500,
            "Server Error",
            {},
            io.BytesIO(b'auth failed for SUPER_SECRET'),
        )
        with patch("urllib.request.urlopen") as up:
            up.side_effect = err
            with pytest.raises(acs.ContentSafetyError) as exc_info:
                acs.analyze_text(
                    "hi",
                    endpoint="https://cs.example.com",
                    key="SUPER_SECRET",
                )
            assert "SUPER_SECRET" not in str(exc_info.value)

    def test_url_error_wrapped(self):
        with patch("urllib.request.urlopen") as up:
            up.side_effect = urllib.error.URLError("network down")
            with pytest.raises(acs.ContentSafetyError):
                acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")

    def test_missing_endpoint_raises(self):
        with pytest.raises(ValueError):
            acs.analyze_text("hi", endpoint="", key="K")

    def test_missing_key_raises(self):
        with pytest.raises(ValueError):
            acs.analyze_text("hi", endpoint="https://cs.example.com", key="")

    def test_api_version_in_url(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")
            req = up.call_args[0][0]
            assert "api-version=2024-09-01" in req.full_url


# --------------------------------------------------------------------------- shield_prompt


class TestShieldPrompt:
    def test_endpoint_and_path(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"userPromptAnalysis": {"attackDetected": False}})
            acs.shield_prompt(
                user_prompt="ignore previous",
                documents=[],
                endpoint="https://cs.example.com",
                key="K",
            )
            req = up.call_args[0][0]
            assert "/contentsafety/text:shieldPrompt?api-version=" in req.full_url

    def test_body_shape(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"userPromptAnalysis": {"attackDetected": False}})
            acs.shield_prompt(
                user_prompt="ignore previous",
                documents=["doc1", "doc2"],
                endpoint="https://cs.example.com",
                key="K",
            )
            body = json.loads(up.call_args[0][0].data.decode("utf-8"))
            assert body["userPrompt"] == "ignore previous"
            assert body["documents"] == ["doc1", "doc2"]

    def test_subscription_key_header(self):
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"userPromptAnalysis": {"attackDetected": False}})
            acs.shield_prompt(
                user_prompt="x",
                documents=[],
                endpoint="https://cs.example.com",
                key="SHIELD_KEY",
            )
            req = up.call_args[0][0]
            assert req.headers["Ocp-apim-subscription-key"] == "SHIELD_KEY"

    def test_returns_parsed_json(self):
        # When no attack is detected, shield_prompt returns the parsed payload.
        payload = {"userPromptAnalysis": {"attackDetected": False}}
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp(payload)
            out = acs.shield_prompt(
                user_prompt="benign",
                documents=[],
                endpoint="https://cs.example.com",
                key="K",
            )
            assert out["userPromptAnalysis"]["attackDetected"] is False

    def test_raises_on_attack_detected(self):
        payload = {"userPromptAnalysis": {"attackDetected": True}}
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp(payload)
            with pytest.raises(acs.ContentSafetyBlocked):
                acs.shield_prompt(
                    user_prompt="bad",
                    documents=[],
                    endpoint="https://cs.example.com",
                    key="K",
                )

    def test_429_retry(self):
        err = urllib.error.HTTPError(
            "https://cs.example.com",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            io.BytesIO(b'{}'),
        )
        ok = _resp({"userPromptAnalysis": {"attackDetected": False}})
        with patch("urllib.request.urlopen") as up, patch.object(acs.time, "sleep"):
            up.side_effect = [err, ok]
            out = acs.shield_prompt(
                user_prompt="x",
                documents=[],
                endpoint="https://cs.example.com",
                key="K",
            )
            assert out["userPromptAnalysis"]["attackDetected"] is False
            assert up.call_count == 2

    def test_redacts_key_in_error(self):
        err = urllib.error.HTTPError(
            "https://cs.example.com",
            500,
            "Boom",
            {},
            io.BytesIO(b'token=KKK exposed'),
        )
        with patch("urllib.request.urlopen") as up:
            up.side_effect = err
            with pytest.raises(acs.ContentSafetyError) as exc_info:
                acs.shield_prompt(
                    user_prompt="x",
                    documents=[],
                    endpoint="https://cs.example.com",
                    key="KKK",
                )
            assert "KKK" not in str(exc_info.value)


# --------------------------------------------------------------------------- helpers


class TestHelpers:
    def test_violation_severity_threshold(self):
        result = {
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 2},
                {"category": "Violence", "severity": 6},
            ]
        }
        assert acs.has_violation(result, ["Hate", "Violence"], threshold=4) is True

    def test_no_violation_below_threshold(self):
        result = {
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 1},
                {"category": "Violence", "severity": 2},
            ]
        }
        assert acs.has_violation(result, ["Hate", "Violence"], threshold=4) is False

    def test_no_violation_uncategorized(self):
        result = {"categoriesAnalysis": [{"category": "Sexual", "severity": 7}]}
        # Sexual not in block list → no violation
        assert acs.has_violation(result, ["Hate"], threshold=4) is False

    def test_redact_helper_strips_key(self):
        msg = "Authorization failed for key=ABCDEF123"
        out = acs._redact("ABCDEF123", msg)
        assert "ABCDEF123" not in out
        assert "***" in out

    def test_redact_helper_handles_empty_key(self):
        assert acs._redact("", "no key here") == "no key here"

    def test_lazy_sdk_import_no_op_when_missing(self):
        # The module must not require azure-* packages at import time.
        # Simply confirm we can call analyze_text without azure-* installed.
        with patch("urllib.request.urlopen") as up:
            up.return_value = _resp({"categoriesAnalysis": []})
            acs.analyze_text("hi", endpoint="https://cs.example.com", key="K")
