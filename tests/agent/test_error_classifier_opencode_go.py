"""OpenCode Go relay error classification tests."""

from types import SimpleNamespace

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    def __init__(self, message, *, status_code, body):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = SimpleNamespace(headers={})


def _opencode_error(status_code, body):
    return MockAPIError(
        f"HTTP {status_code}: {body}",
        status_code=status_code,
        body=body,
    )


def test_opencode_go_message_less_400_is_retryable_server_error():
    error = _opencode_error(400, {"object": "error", "model": "gpt-5"})

    result = classify_api_error(error, provider="opencode-go", model="gpt-5")

    assert result.reason == FailoverReason.server_error
    assert result.retryable is True


def test_opencode_go_message_bearing_400_remains_non_retryable():
    error = _opencode_error(
        400,
        {
            "object": "error",
            "model": "gpt-5",
            "message": "Unsupported parameter: temperature",
        },
    )

    result = classify_api_error(error, provider="opencode-go", model="gpt-5")

    assert result.reason == FailoverReason.format_error
    assert result.retryable is False


def test_opencode_go_message_less_401_keeps_auth_classification():
    error = _opencode_error(401, {"object": "error", "model": "gpt-5"})

    result = classify_api_error(error, provider="opencode-go", model="gpt-5")

    assert result.reason == FailoverReason.auth
    assert result.retryable is False


def test_other_provider_message_less_400_remains_non_retryable():
    error = _opencode_error(400, {"object": "error", "model": "gpt-5"})

    result = classify_api_error(error, provider="custom", model="gpt-5")

    assert result.reason == FailoverReason.format_error
    assert result.retryable is False
