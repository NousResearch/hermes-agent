"""A non-retryable 4xx (rejected OAuth token, bad key) must reach UI clients with the
classifier's verdict: without ``failure_reason`` the desktop error card read a 401 as a
retryable "Provider error" and offered Retry instead of a re-login."""
from __future__ import annotations

import pytest

from agent.agent_runtime_helpers import extract_api_error_context
from agent.error_classifier import classify_api_error
from agent.error_surface import LAYER_AUTH, build_error_surface_from_result
from agent.turn_recovery import nonretryable_client_error_result
from hermes_cli.cli_single_query import _result_reset_at, _single_query_exit_code
from hermes_cli.quiet_single_query import exit_single_query


class _Rejected(Exception):
    status_code = 401

    def __init__(self) -> None:
        super().__init__("HTTP 401: User not found.")


class _Agent:
    log_prefix = ""
    verbose = False

    def _summarize_api_error(self, error):
        return str(error)

    _extract_api_error_context = staticmethod(extract_api_error_context)

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


@pytest.mark.parametrize("reset_at", [None, 1_900_000_000])
def test_billing_402_only_reports_provider_reset_when_present(monkeypatch, capsys, reset_at):
    class _QuotaError(Exception):
        status_code = 402
        body = {"error": {"message": "Credits exhausted", "resets_at": reset_at}}

    error = _QuotaError("Credits exhausted")
    classified = classify_api_error(error, provider="custom", model="m")
    result = nonretryable_client_error_result(
        _Agent(), error, classified, status_code=402, api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=1, approx_tokens=10, provider="custom",
        base_url="https://example.test/v1", model="m",
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", "card-1")
    code = _single_query_exit_code(result)
    with pytest.raises(SystemExit) as exited:
        exit_single_query(code, reset_at=_result_reset_at(result))
    assert exited.value.code == 75
    trailer = capsys.readouterr().err
    assert ("reset_at=1900000000" in trailer) is (reset_at is not None)
