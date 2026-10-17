"""Tests for the provider-scoped 403 relay code in agent.error_classifier.

OpenCode Go/Console wrap an internal upstream failure (an upstream body that is not valid JSON)
in HTTP 403 with ``error.code=server_error``. The relay's own code is read LAST in
``_status_403`` — after the global transient table, billing/credit-exhaustion evidence, and
WAF/CDN markers — so every row below is the contract that something more specific keeps
winning. Lives in its own file: ``tests/agent/test_error_classifier.py`` is already over the
repo's file-lines target and may only shrink.
"""

from types import SimpleNamespace

import pytest

from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
)


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


class TestClassify403RelayCode:
    @pytest.mark.parametrize(
        "provider, body, expected",
        [
            pytest.param(
                "opencode-go",
                {"error": {"message": "Upstream request failed: Upstream response was not valid JSON",
                           "type": "server_error", "code": "server_error"}},
                FailoverReason.overloaded,
                id="relay-code-is-transient",
            ),
            pytest.param(
                # The opencode-go profile declares this alias; the slug is canonicalised by
                # asking the provider registry, so no second alias table lives in the classifier.
                "opencode_go",
                {"error": {"message": "Upstream request failed: Upstream response was not valid JSON",
                           "type": "server_error", "code": "server_error"}},
                FailoverReason.overloaded,
                id="declared-alias-resolves",
            ),
            pytest.param(
                # ``server_error`` is generic — re-globalising the code would move this verdict.
                "anthropic",
                {"error": {"message": "Upstream request failed: Upstream response was not valid JSON",
                           "type": "server_error", "code": "server_error"}},
                FailoverReason.auth,
                id="other-provider-keeps-its-verdict",
            ),
            pytest.param(
                "opencode-go",
                {"error": {"message": "Insufficient credits", "type": "server_error",
                           "code": "server_error"}},
                FailoverReason.billing,
                id="billing-outranks-the-relay-code",
            ),
            pytest.param(
                "opencode-go",
                {"error": {"message": "Attention Required! | Cloudflare", "code": "server_error"}},
                FailoverReason.upstream_blocked,
                id="waf-marker-outranks-the-relay-code",
            ),
            pytest.param(
                "opencode-go",
                {"error": {"message": "Invalid API key provided", "code": "invalid_api_key"}},
                FailoverReason.auth,
                id="no-relay-code-stays-auth",
            ),
        ],
    )
    def test_403_relay_code_is_read_last_and_per_provider(self, provider, body, expected):
        """One 403 table: the relay's own code is a transient upstream failure on its route only.

        OpenCode Go/Console wrap an internal upstream failure (an upstream body that is not valid
        JSON) in HTTP 403 with ``error.code=server_error``. Read as the auth default, that verdict
        benches the sole credential and cascades the fallback chain for the rest of the cooldown
        window — the field report in #117869.

        The provider-scoped check is read LAST in ``_status_403``, so every row below is the
        contract that something more specific keeps winning: a global transient code, billing
        evidence, a WAF/CDN marker, and a body with no relay-internal code at all.
        """
        result = classify_api_error(
            MockAPIError("Forbidden", status_code=403, body=body),
            provider=provider, model="deepseek-v4.1-flash",
        )
        assert result.reason == expected
        assert result.is_auth is (expected is FailoverReason.auth)
