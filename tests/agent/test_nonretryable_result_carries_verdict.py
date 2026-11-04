"""A non-retryable 4xx (rejected OAuth token, bad key) must reach UI clients with the
classifier's verdict: without ``failure_reason`` the desktop error card read a 401 as a
retryable "Provider error" and offered Retry instead of a re-login."""
from __future__ import annotations

from agent.error_classifier import classify_api_error
from agent.error_surface import LAYER_AUTH, build_error_surface_from_result
from agent.turn_recovery import nonretryable_client_error_result


class _Rejected(Exception):
    status_code = 401

    def __init__(self) -> None:
        super().__init__("HTTP 401: User not found.")


class _Agent:
    log_prefix = ""
    verbose = False

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):  # status/persist/print helpers the terminal path calls
        return lambda *args, **kwargs: None


def test_nonretryable_401_result_classifies_as_auth_for_the_ui():
    error = _Rejected()
    classified = classify_api_error(error, provider="nous", model="m")
    result = nonretryable_client_error_result(
        _Agent(), error, classified, status_code=401, api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=1, approx_tokens=10, provider="nous",
        base_url="https://inference-api.nousresearch.com/v1", model="m",
    )
    assert result["failure_reason"] == classified.reason.value
    assert result["failure_retryable"] is classified.retryable is False

    surface = build_error_surface_from_result(result, provider="nous", model="m")
    assert surface["layer"] == LAYER_AUTH
    assert surface["retryable"] is False
    assert surface["auth_kind"] == "oauth"
