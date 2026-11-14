"""A Gemini HTTP 400 "API key not valid" must rotate the credential (#150942).

Google AI Studio reports a revoked or invalid key as **400 INVALID_ARGUMENT**, not 401/403:

    Gemini HTTP 400 (INVALID_ARGUMENT): API key not valid. Please pass a valid API key.

``_STATUS_TO_FAILOVER_REASON`` maps only 401/403 to auth, so a 400 went to
``_classify_400()`` — which had no auth branch at all — and credential-pool rotation never
ran. The dead key was never benched and the next live key in the pool was never tried; the
turn only "succeeded" because ``fallback_providers`` caught it, hiding the gap.

A key-validity 400 cannot be repaired by retrying the same credential, so rotating is
strictly better than benching the whole pool.
"""

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


@pytest.mark.parametrize("message", [
    # The wording Google actually sends.
    "Gemini HTTP 400 (INVALID_ARGUMENT): API key not valid. Please pass a valid API key.",
    # The same failure without the surrounding provider envelope.
    "API key not valid",
    # A provider that only says what to do.
    "Please pass a valid API key",
    # The phrasings that already rotate on a 401/403 must rotate on a 400 too.
    "Invalid API key",
    "invalid_api_key",
])
def test_a_key_validity_400_rotates_the_credential(message):
    result = classify_api_error(MockAPIError(message, status_code=400), provider="gemini")
    assert result.reason == FailoverReason.auth
    assert result.should_rotate_credential is True


@pytest.mark.parametrize("message", [
    # An ordinary bad request must stay an ordinary bad request.
    "Bad Request",
    "Invalid JSON payload received. Unknown name \"foo\"",
    # A 400 that merely MENTIONS a key must not be read as a key failure.
    "Invalid request: the api_key field must be a string",
])
def test_an_ordinary_400_still_does_not_rotate(message):
    result = classify_api_error(MockAPIError(message, status_code=400), provider="gemini")
    assert result.should_rotate_credential is False
    assert result.reason != FailoverReason.auth


def test_a_403_key_failure_still_classifies_as_auth():
    """The guard: the fix adds a 400 branch, it does not change the 401/403 path.

    A bare 403 classifies as ``auth`` without the ``should_rotate_credential`` hint — the
    runtime rotates on the ``auth`` reason itself (see the ``effective_reason ==
    FailoverReason.auth`` branch in ``agent/agent_runtime_helpers``), so the hint is a
    turn-path detail, not the rotation trigger.
    """
    result = classify_api_error(MockAPIError("Forbidden", status_code=403), provider="gemini")
    assert result.reason == FailoverReason.auth
