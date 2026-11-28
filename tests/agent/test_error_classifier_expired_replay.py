"""A replayed reasoning blob past its TTL routes to the replay-strip recovery (#156189)."""

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code, self.body = status_code, body or {}


# Verbatim OpenCode Go body for an expired blob.
_EXPIRED = {"model": "muse-spark-1.3-contributor", "error": {
    "code": None, "message": "reasoning `encrypted_content` has expired",
    "param": None, "type": "invalid_request_error"}}


def test_opencode_expired_replay_blob_reaches_replay_strip():
    error = _APIError(f"Error code: 400 - {_EXPIRED}", status_code=400, body=_EXPIRED)
    result = classify_api_error(error, provider="opencode-go", model="muse-spark-1.3-contributor")
    assert result.reason == FailoverReason.invalid_encrypted_content
    assert result.retryable is True
    assert result.should_fallback is False
