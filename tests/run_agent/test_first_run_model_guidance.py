from types import SimpleNamespace

from agent.conversation_loop import _friendly_first_run_api_error
from tui_gateway.server import _probe_credentials


class _ErrorWithResponse(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def test_bedrock_unrecognized_client_gets_setup_guidance():
    raw = (
        "An error occurred (UnrecognizedClientException) when calling the "
        "ConverseStream operation: The security token included in the request is invalid."
    )

    friendly = _friendly_first_run_api_error(
        raw,
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        error=_ErrorWithResponse(raw),
    )

    assert friendly is not None
    assert "No usable model credentials are configured" in friendly
    assert "hermes setup" in friendly
    assert "hermes model" in friendly
    assert "UnrecognizedClientException" not in friendly
    assert "ConverseStream" not in friendly


def test_ordinary_provider_errors_stay_raw():
    raw = "HTTP 500: upstream unavailable"

    assert (
        _friendly_first_run_api_error(
            raw,
            provider="openai",
            model="gpt-4o",
            error=_ErrorWithResponse(raw, status_code=500),
        )
        is None
    )


def test_tui_gateway_missing_key_warning_is_actionable():
    agent = SimpleNamespace(provider="bedrock", model="anthropic.claude", api_key="no-key-required")

    warning = _probe_credentials(agent)

    assert "No usable model credentials are configured" in warning
    assert "hermes setup" in warning
    assert "hermes model" in warning
    assert "First message will fail" not in warning
