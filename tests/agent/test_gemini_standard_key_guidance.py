"""Tests for Gemini legacy Standard-key auth guidance.

Google began rejecting unrestricted legacy "Standard" Google Cloud API keys
on the Gemini API on June 19, 2026 (all Standard keys stop working in
September 2026). Two wire shapes exist:

- Original: 401 + ``ACCESS_TOKEN_TYPE_UNSUPPORTED`` / "expected OAuth 2 access
  token". ``gemini_http_error`` must append key-migration guidance.
- After the September cutoff: 400 ``API_KEY_INVALID`` ("API key not valid")
  for leftover AIza Standard keys. That shape must get the same guidance
  when the presented key is still AIza-shaped — otherwise vision/chat just
  says the key is invalid and hides the real fix. A mistyped Auth (``AQ.``)
  key must keep the raw invalid-key message.

Port of Kilo-Org/kilocode#12162, adapted to Hermes' GeminiAPIError surface.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from agent.gemini_native_adapter import (
    gemini_http_error,
    is_standard_key_auth_error,
)


GOOGLE_AUTH_MESSAGE = (
    "Request had invalid authentication credentials. Expected OAuth 2 access "
    "token, login cookie or other valid authentication credential. See "
    "https://developers.google.com/identity/sign-in/web/devconsole-project."
)

GUIDANCE_MARKER = "rejected this API key's type"

API_KEY_INVALID_MESSAGE = "API key not valid. Please pass a valid API key."
# Concatenated so the file never contains a contiguous AIzaSy… token.
_AIZA_STANDARD_KEY = "AIza" + "Sy_TEST_" + "x" * 24
_AQ_AUTH_KEY = "AQ." + "B" * 48


def _mock_response(status: int, body: str, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.text = body
    return resp


def _google_error_body(
    status_code: int,
    message: str,
    status: str = "UNAUTHENTICATED",
    reason: str | None = None,
) -> str:
    err: dict = {"code": status_code, "message": message, "status": status}
    if reason is not None:
        err["details"] = [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": reason,
                "domain": "googleapis.com",
                "metadata": {"service": "generativelanguage.googleapis.com"},
            }
        ]
    return json.dumps({"error": err})


class TestIsStandardKeyAuthError:


    def test_oauth_message_requires_401(self):
        assert not is_standard_key_auth_error(400, GOOGLE_AUTH_MESSAGE)
        assert not is_standard_key_auth_error(403, GOOGLE_AUTH_MESSAGE)
        # The 400 path stays narrowed to API_KEY_INVALID: a generic 400 on an AIza-shaped key
        # (malformed payload, unknown model) must not claim the key TYPE was rejected.
        assert not is_standard_key_auth_error(
            400, "Invalid JSON payload received. Unknown name \"x\".", "INVALID_ARGUMENT",
            api_key=_AIZA_STANDARD_KEY,
        )


    def test_empty_message_is_safe(self):
        assert not is_standard_key_auth_error(401, "")
        assert not is_standard_key_auth_error(401, None)  # type: ignore[arg-type]


    def test_400_api_key_invalid_with_aiza_key_is_standard_key(self):
        assert is_standard_key_auth_error(
            400, API_KEY_INVALID_MESSAGE, "API_KEY_INVALID", api_key=_AIZA_STANDARD_KEY
        )
        assert is_standard_key_auth_error(
            400, API_KEY_INVALID_MESSAGE, api_key=_AIZA_STANDARD_KEY
        )


    def test_400_api_key_invalid_without_aiza_key_is_not_standard_key(self):
        assert not is_standard_key_auth_error(
            400, API_KEY_INVALID_MESSAGE, "API_KEY_INVALID"
        )
        assert not is_standard_key_auth_error(
            400, API_KEY_INVALID_MESSAGE, "API_KEY_INVALID", api_key=_AQ_AUTH_KEY
        )


class TestGeminiHttpErrorGuidance:
    def test_guidance_appended_on_oauth_401_with_reason(self):
        body = _google_error_body(
            401, GOOGLE_AUTH_MESSAGE, reason="ACCESS_TOKEN_TYPE_UNSUPPORTED"
        )
        err = gemini_http_error(_mock_response(401, body))
        text = str(err)
        assert GUIDANCE_MARKER in text
        assert "aistudio.google.com/api-keys" in text
        assert "ai.google.dev/gemini-api/docs/api-key" in text
        assert err.code == "gemini_unauthorized"


    def test_guidance_appended_on_400_api_key_invalid_for_aiza_key(self):
        body = _google_error_body(
            400, API_KEY_INVALID_MESSAGE, status="INVALID_ARGUMENT",
            reason="API_KEY_INVALID",
        )
        err = gemini_http_error(_mock_response(400, body), api_key=_AIZA_STANDARD_KEY)
        text = str(err)
        assert GUIDANCE_MARKER in text
        assert "aistudio.google.com/api-keys" in text
        assert err.code == "gemini_http_400"


    def test_400_api_key_invalid_for_auth_key_keeps_raw_message(self):
        body = _google_error_body(
            400, API_KEY_INVALID_MESSAGE, status="INVALID_ARGUMENT",
            reason="API_KEY_INVALID",
        )
        err = gemini_http_error(_mock_response(400, body), api_key=_AQ_AUTH_KEY)
        assert GUIDANCE_MARKER not in str(err)


    def test_403_with_oauth_message_gets_no_guidance(self):
        body = _google_error_body(403, GOOGLE_AUTH_MESSAGE, status="PERMISSION_DENIED")
        err = gemini_http_error(_mock_response(403, body))
        assert GUIDANCE_MARKER not in str(err)

    def test_free_tier_429_unaffected(self):
        body = json.dumps(
            {
                "error": {
                    "code": 429,
                    "message": (
                        "Quota exceeded for metric: generativelanguage.googleapis.com/"
                        "generate_content_free_tier_requests, limit: 20"
                    ),
                }
            }
        )
        err = gemini_http_error(_mock_response(429, body))
        text = str(err)
        assert "free tier" in text
        assert GUIDANCE_MARKER not in text



class TestSummarizerPreservesGuidance:
    """_summarize_api_error must not strip adapter-composed guidance.

    GeminiAPIError carries ``.response``; without the GeminiAPIError branch,
    the summarizer re-extracts the raw body's error.message (capped at 300
    chars), silently discarding both the Standard-key 401 guidance and the
    pre-existing free-tier 429 guidance.
    """

    def test_standard_key_guidance_survives_summarizer(self):
        from run_agent import AIAgent

        body = _google_error_body(
            401, GOOGLE_AUTH_MESSAGE, reason="ACCESS_TOKEN_TYPE_UNSUPPORTED"
        )
        err = gemini_http_error(_mock_response(401, body))
        summary = AIAgent._summarize_api_error(err)
        assert GUIDANCE_MARKER in summary
        assert "aistudio.google.com/api-keys" in summary

    def test_400_standard_key_guidance_survives_summarizer(self):
        from run_agent import AIAgent

        body = _google_error_body(
            400, API_KEY_INVALID_MESSAGE, status="INVALID_ARGUMENT",
            reason="API_KEY_INVALID",
        )
        err = gemini_http_error(_mock_response(400, body), api_key=_AIZA_STANDARD_KEY)
        summary = AIAgent._summarize_api_error(err)
        assert GUIDANCE_MARKER in summary

    def test_free_tier_guidance_survives_summarizer(self):
        from run_agent import AIAgent

        body = json.dumps(
            {
                "error": {
                    "code": 429,
                    "message": (
                        "Quota exceeded for metric: "
                        "generativelanguage.googleapis.com/"
                        "generate_content_free_tier_requests, limit: 20"
                    ),
                }
            }
        )
        err = gemini_http_error(_mock_response(429, body))
        summary = AIAgent._summarize_api_error(err)
        assert "free tier" in summary

    def test_non_gemini_errors_keep_response_body_extraction(self):
        from types import SimpleNamespace

        from run_agent import AIAgent

        err = Exception("")
        err.status_code = 400
        err.body = {}
        err.response = SimpleNamespace(
            text='{"error": {"message": "model `foo` does not exist"}}'
        )
        summary = AIAgent._summarize_api_error(err)
        assert "model `foo` does not exist" in summary
